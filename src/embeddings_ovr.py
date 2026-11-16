"""Frozen sentence embeddings + one-vs-rest logistic regression.

Each issue's title and description (raw text, joined with a space) are encoded
once with a pretrained sentence-transformers model, which is not fine-tuned.
One logistic regression per label is trained on the "train" split embeddings,
for each regularisation strength C in LOGREG_C_GRID. Reports micro
precision/recall/F1 on "test" over all 217 labels, twice:

- default: probability threshold 0.5; C chosen by val micro-F1 at 0.5
- tuned: C and one probability threshold for all labels, chosen together
  by val micro-F1

Only the "val" split is used to choose C and the threshold.

Embeddings for full runs are cached in data/embeddings/ so the classifier can be
rerun without re-encoding.

Usage:
  python src/embeddings_ovr.py                       # default model: all-MiniLM-L6-v2
  python src/embeddings_ovr.py --model sentence-transformers/all-mpnet-base-v2
  python src/embeddings_ovr.py --sample 300          # quick check, no cache or results written
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from data import ROOT, binary_labels, issue_text, load_split_data
from evaluate import micro_metrics, save_result, tune_threshold

EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
LOGREG_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LOGREG_MAX_ITER = 1000


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_splits(model_name, splits, batch_size, use_cache, dataset):
    """Return {split name: embedding array}, reading/writing the cache if use_cache."""
    slug = model_name.split("/")[-1]
    model = None
    embeddings = {}
    for name, df in splits.items():
        cache = EMBEDDINGS_DIR / f"{slug}_{dataset}_{name}.npy"
        if use_cache and cache.exists():
            embeddings[name] = np.load(cache)
            print(f"Loaded cached {cache}")
            continue
        if model is None:
            model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
            print(f"Encoding with {model_name} on {model.device} (max_seq_length={model.max_seq_length})")
        embeddings[name] = model.encode(
            issue_text(df), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )
        if use_cache:
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, embeddings[name])
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Frozen sentence embeddings + one-vs-rest logistic regression")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset", choices=["official", "corrected"], default="official",
                        help="labels as distributed, or with the repository join fixed")
    parser.add_argument("--sample", type=int, help="use only this many rows of each split")
    args = parser.parse_args()

    set_seeds(args.seed)
    start = time.time()
    splits = load_split_data(args.dataset)
    if args.sample:
        splits = {name: df.sample(n=min(args.sample, len(df)), random_state=args.seed)
                  for name, df in splits.items()}
    print({name: len(df) for name, df in splits.items()})

    embeddings = encode_splits(args.model, splits, args.batch_size, use_cache=not args.sample, dataset=args.dataset)
    encode_seconds = time.time() - start
    y = {name: binary_labels(df) for name, df in splits.items()}

    # Labels with no positive training example get a constant "absent" predictor
    # from OneVsRestClassifier, which warns once per such label. That is expected.
    # The fits run in worker processes, so the filter is passed via the environment.
    os.environ["PYTHONWARNINGS"] = "ignore:Label not:UserWarning"

    # For each C: fit on train, score val. Keep test scores aside for the chosen C only.
    candidates = []
    for c in LOGREG_C_GRID:
        classifier = OneVsRestClassifier(
            LogisticRegression(C=c, max_iter=LOGREG_MAX_ITER, random_state=args.seed),
            n_jobs=-1,
        )
        classifier.fit(embeddings["train"], y["train"])
        val_scores = classifier.predict_proba(embeddings["val"])
        val_f1_at_half = micro_metrics(y["val"], (val_scores >= 0.5).astype(int))["micro_f1"]
        threshold, val_f1_tuned = tune_threshold(y["val"], val_scores)
        print(f"C={c}: val micro-F1 {val_f1_at_half:.4f} at 0.5, {val_f1_tuned:.4f} at tuned {threshold}")
        candidates.append({
            "C": c,
            "val_f1_at_half": val_f1_at_half,
            "threshold": threshold,
            "val_f1_tuned": val_f1_tuned,
            "test_scores": classifier.predict_proba(embeddings["test"]),
        })

    # Ties go to the smaller C (first in the grid).
    default_pick = max(candidates, key=lambda r: r["val_f1_at_half"])
    tuned_pick = max(candidates, key=lambda r: r["val_f1_tuned"])
    default_predictions = (default_pick["test_scores"] >= 0.5).astype(int)
    default_metrics = micro_metrics(y["test"], default_predictions)
    threshold = tuned_pick["threshold"]
    tuned_predictions = (tuned_pick["test_scores"] >= threshold).astype(int)
    tuned_metrics = micro_metrics(y["test"], tuned_predictions)
    runtime = time.time() - start
    print(f"default (C={default_pick['C']}):", default_metrics)
    print(f"tuned (C={tuned_pick['C']}, threshold {threshold}, "
          f"val micro-F1 {tuned_pick['val_f1_tuned']:.4f}):", tuned_metrics)
    print(f"encode {encode_seconds:.1f}s, total {runtime:.1f}s")

    if args.sample:
        return
    slug = args.model.split("/")[-1]
    common = {
        "model": f"Frozen {slug} embeddings + one-vs-rest logistic regression",
        "dataset": args.dataset,
        "seed": args.seed,
        "hyperparameters": {
            "embedding_model": args.model,
            "normalize_embeddings": True,
            "embedding_dim": int(embeddings["train"].shape[1]),
            "logreg_C_grid": LOGREG_C_GRID,
            "val_search": [{k: v for k, v in r.items() if k != "test_scores"} for r in candidates],
            "logreg_max_iter": LOGREG_MAX_ITER,
            "train_split": "train",
        },
        "runtime_seconds": runtime,
        "encode_seconds_this_run": encode_seconds,
        "n_train": len(y["train"]),
        "n_val": len(y["val"]),
        "n_test": len(y["test"]),
    }
    save_result(f"emb_{slug}_ovr", {
        **common, "threshold": "default (0.5)", "chosen_C": default_pick["C"],
        "val_micro_f1": default_pick["val_f1_at_half"], **default_metrics,
    }, test_predictions=default_predictions, test_scores=default_pick["test_scores"])
    save_result(f"emb_{slug}_ovr_tuned", {
        **common, "threshold": f"tuned on val: {threshold}", "chosen_C": tuned_pick["C"],
        "val_micro_f1": tuned_pick["val_f1_tuned"], **tuned_metrics,
    }, test_predictions=tuned_predictions, test_scores=tuned_pick["test_scores"])


if __name__ == "__main__":
    main()
