"""Fine-tune a pretrained encoder for 217-label skill classification.

The encoder (e.g. roberta-base, microsoft/codebert-base) gets a linear head
with one output per label, a sigmoid per label, and binary cross-entropy loss.
Input is the issue title and description joined with a space, optionally
prefixed with the repository name (--with-repo).

Training uses the "train" split. After every epoch the model is scored on
"val", and training stops early when validation stops improving. Two
checkpoints are kept, both chosen on "val" only:

- default: the epoch with the best val micro-F1 at threshold 0.5
- tuned: the epoch and one threshold for all labels with the best val micro-F1

Each is then scored once on "test".

Usage:
  python src/finetune.py --model roberta-base
  python src/finetune.py --model microsoft/codebert-base
  python src/finetune.py --model roberta-base --with-repo
  python src/finetune.py --sample 200 --epochs 1      # quick check, no results written
"""

import argparse
import os
import random
import time

# Needed for deterministic cuBLAS; must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import N_LABELS, binary_labels, issue_text, load_split_data
from evaluate import micro_metrics, save_result, seed_tag, tune_threshold


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def model_inputs(df, with_repo):
    texts = issue_text(df)
    if with_repo:
        texts = [f"{repo}: {text}" for repo, text in zip(df["Repo Name"], texts)]
    return texts


def make_loader(tokenizer, texts, labels, max_length, batch_size, shuffle, seed):
    encoded = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def prior_bce(labels):
    """Mean BCE of always predicting each label's frequency. A model whose
    train loss does not go below this has learned nothing from the text."""
    p = np.clip(labels.mean(axis=0), 1e-7, 1 - 1e-7)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


@torch.no_grad()
def predict(model, loader, device, use_bf16):
    """Return sigmoid probabilities, (n_rows x 217)."""
    model.eval()
    outputs = []
    for input_ids, attention_mask, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an encoder with sigmoid outputs and BCE loss")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate for the pretrained encoder")
    parser.add_argument("--head-lr", type=float, help="learning rate for the new classification head (default: --lr)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10, help="maximum number of epochs")
    parser.add_argument("--patience", type=int, default=2, help="stop after this many epochs without val improvement")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prior-bias-init", action="store_true",
                        help="initialise the output bias to each label's log-odds in the train split")
    parser.add_argument("--with-repo", action="store_true", help="prefix the input text with the repository name")
    parser.add_argument("--dataset", choices=["official", "corrected"], default="official",
                        help="labels as distributed, or with the repository join fixed")
    parser.add_argument("--sample", type=int, help="use only this many rows of each split")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}; bf16: {use_bf16}")

    start = time.time()
    splits = load_split_data(args.dataset)
    if args.sample:
        splits = {name: df.sample(n=min(args.sample, len(df)), random_state=args.seed)
                  for name, df in splits.items()}
    print({name: len(df) for name, df in splits.items()})
    y = {name: binary_labels(df) for name, df in splits.items()}
    print(f"Reference: train BCE of predicting each label's train frequency = {prior_bce(y['train']):.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    loaders = {
        name: make_loader(tokenizer, model_inputs(df, args.with_repo), y[name], args.max_length,
                          args.batch_size, shuffle=(name == "train"), seed=args.seed)
        for name, df in splits.items()
    }

    # Eager attention: the default fused attention has a non-deterministic backward pass on GPU.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=N_LABELS, attn_implementation="eager"
    ).to(device)
    if args.prior_bias_init:
        # Start the output layer at the label-frequency prediction: bias = log-odds
        # of each label in the training split. Uses training labels only.
        p = np.clip(y["train"].mean(axis=0), 1e-4, 1 - 1e-4)
        with torch.no_grad():
            model.classifier.out_proj.bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
    head_lr = args.head_lr or args.lr
    head_params = [p for name, p in model.named_parameters() if name.startswith("classifier.")]
    encoder_params = [p for name, p in model.named_parameters() if not name.startswith("classifier.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=args.weight_decay,
    )
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_fraction * total_steps), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = {"default": {"val_f1": -1.0}, "tuned": {"val_f1": -1.0}}
    history = []
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in loaders["train"]:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            loss = loss_fn(logits.float(), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)

        val_scores = predict(model, loaders["val"], device, use_bf16)
        val_f1_at_half = micro_metrics(y["val"], (val_scores >= 0.5).astype(int))["micro_f1"]
        threshold, val_f1_tuned = tune_threshold(y["val"], val_scores)
        train_loss = total_loss / len(y["train"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1_at_half": val_f1_at_half,
                        "val_f1_tuned": val_f1_tuned, "threshold": threshold,
                        "seconds": time.time() - epoch_start})
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val micro-F1 {val_f1_at_half:.4f} at 0.5, "
              f"{val_f1_tuned:.4f} at tuned {threshold} ({time.time() - epoch_start:.0f}s)")

        improved = False
        for key, f1, thr in (("default", val_f1_at_half, 0.5), ("tuned", val_f1_tuned, threshold)):
            if f1 > best[key]["val_f1"]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best[key] = {"val_f1": f1, "threshold": thr, "epoch": epoch, "state": state}
                improved = True
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if epochs_without_improvement >= args.patience:
            print(f"Stopping early: no val improvement for {args.patience} epochs")
            break

    # Score the two chosen checkpoints on test, once each.
    test_metrics, test_outputs = {}, {}
    for key in ("default", "tuned"):
        model.load_state_dict(best[key]["state"])
        test_scores = predict(model, loaders["test"], device, use_bf16)
        test_predictions = (test_scores >= best[key]["threshold"]).astype(int)
        test_metrics[key] = micro_metrics(y["test"], test_predictions)
        test_outputs[key] = {"test_predictions": test_predictions, "test_scores": test_scores}
        print(f"{key} (epoch {best[key]['epoch']}, threshold {best[key]['threshold']}):", test_metrics[key])
    runtime = time.time() - start
    print(f"total {runtime:.0f}s")

    if args.sample:
        return
    slug = args.model.split("/")[-1] + ("_repo" if args.with_repo else "") + f"_lr{args.lr:g}" + seed_tag(args.seed)
    common = {
        "model": f"Fine-tuned {args.model}" + (" (text + repo name)" if args.with_repo else ""),
        "dataset": args.dataset,
        "seed": args.seed,
        "hyperparameters": {
            "pretrained_model": args.model,
            "lr": args.lr,
            "head_lr": head_lr,
            "attn_implementation": "eager",
            "prior_bias_init": args.prior_bias_init,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "max_length": args.max_length,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "with_repo": args.with_repo,
            "loss": "BCEWithLogitsLoss",
            "bf16_autocast": use_bf16,
            "train_split": "train",
        },
        "history": history,
        "runtime_seconds": runtime,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_train": len(y["train"]),
        "n_val": len(y["val"]),
        "n_test": len(y["test"]),
    }
    save_result(f"ft_{slug}", {
        **common, "threshold": "default (0.5)", "chosen_epoch": best["default"]["epoch"],
        "val_micro_f1": best["default"]["val_f1"], **test_metrics["default"],
    }, **test_outputs["default"])
    save_result(f"ft_{slug}_tuned", {
        **common, "threshold": f"tuned on val: {best['tuned']['threshold']}",
        "chosen_epoch": best["tuned"]["epoch"], "val_micro_f1": best["tuned"]["val_f1"], **test_metrics["tuned"],
    }, **test_outputs["tuned"])


if __name__ == "__main__":
    main()
