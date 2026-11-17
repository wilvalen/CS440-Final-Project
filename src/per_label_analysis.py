"""Per-label F1 on the test split: for which labels does a model learn anything?

Micro-F1 is dominated by the most frequent labels, so it can hide gains on
rarer ones. This script reads the saved test predictions in
results/predictions/ and computes F1 per label (averaged over every seed that
was run).

Each label is compared with its best constant prediction, "always predict
this label", whose F1 is 2p / (1 + p) for test prevalence p (predicting it
never scores 0). This is a fixed rule, not tuned. The micro-F1 floor is not
used as the per-label reference: with one global threshold it predicts each
label always or never, so labels just below the threshold get F1 0 and any
model looks like it gains on them.

Because F1 depends on the threshold, the script also reports a threshold-free
measure from the saved scores: ROC-AUC per label (0.5 = no signal; the floor
is exactly 0.5). Only labels with both positive and negative test examples
get an AUC.

Only labels with at least one positive example in the test split are scored
(per-label F1 is undefined otherwise).

Outputs (in report/, <name> = <dataset>_<variant>):
  per_label_f1_<name>.csv          one row per label, F1 and AUC per model, gain over "always"
  per_label_summary_<name>.md      macro-F1, mean AUC and win/loss counts per model
  per_label_gain_<name>.png / .pdf F1 gain over "always" against label frequency

Usage: python src/per_label_analysis.py [--dataset official|corrected] [--variant tuned|default]
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from data import ROOT, binary_labels, label_columns, load_split_data
from evaluate import predictions_dir

REPORT_DIR = ROOT / "report"
FLOOR = "floor_label_frequency"
# (result name without the "_tuned" suffix, short name for tables and figures).
# Fine-tuned runs use the learning rate chosen on validation for each dataset.
COMMON_MODELS = [
    ("baseline_rf", "TF-IDF + RF"),
    ("emb_all-MiniLM-L6-v2_ovr", "MiniLM + LR"),
    ("emb_all-mpnet-base-v2_ovr", "mpnet + LR"),
]
CHOSEN_LR = {"official": "2e-05", "corrected": "5e-05"}


def models_for(dataset):
    lr = CHOSEN_LR[dataset]
    return COMMON_MODELS + [
        (f"ft_roberta-base_lr{lr}", "RoBERTa (ft)"),
        (f"ft_codebert-base_lr{lr}", "CodeBERT (ft)"),
        (f"ft_codebert-base_repo_lr{lr}", "CodeBERT + repo (ft)"),
    ]
FIGURE_MODELS = ["TF-IDF + RF", "mpnet + LR", "CodeBERT (ft)"]
AUC_SIGNAL = 0.6  # labels with AUC above this are counted as "some signal"
GAIN_MARGIN = 0.05  # a label counts as a win/loss when F1 differs from "always" by more than this

# Reference palette, light mode: slot 1 for the data, text and grid in neutral inks.
SERIES_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def prediction_files(name, variant, dataset):
    """All seeds' prediction files for one run name, e.g. ft_x, ft_x_seed43."""
    suffix = "_tuned" if variant == "tuned" else ""
    pattern = re.compile(rf"^{re.escape(name)}(_seed\d+)?{suffix}\.npz$")
    return sorted(p for p in predictions_dir(dataset).glob("*.npz") if pattern.match(p.name))


def per_label_f1(y_true, predictions):
    return f1_score(y_true, predictions, average=None, zero_division=0)


def per_label_auc(y_true, scores, columns):
    return np.array([roc_auc_score(y_true[:, j], scores[:, j]) for j in columns])


def main():
    parser = argparse.ArgumentParser(description="Per-label F1 compared with the floor")
    parser.add_argument("--variant", choices=["tuned", "default"], default="tuned")
    parser.add_argument("--dataset", choices=["official", "corrected"], default="official",
                        help="labels as distributed, or with the repository join fixed")
    args = parser.parse_args()

    splits = load_split_data(args.dataset)
    name_suffix = f"{args.dataset}_{args.variant}"
    labels = label_columns(splits["test"])
    y_train, y_test = binary_labels(splits["train"]), binary_labels(splits["test"])
    scored = y_test.sum(axis=0) > 0
    # AUC needs both classes in test.
    auc_columns = np.flatnonzero(scored & (y_test.sum(axis=0) < len(y_test)))
    print(f"Scoring {scored.sum()} labels with at least one positive in test")

    table = pd.DataFrame({
        "label": np.array(labels)[scored],
        "train_frequency": y_train.mean(axis=0)[scored],
        "test_positives": y_test.sum(axis=0)[scored],
    })
    table["Always"] = per_label_f1(y_test, np.ones_like(y_test))[scored]
    summary = [{"model": "Always predict the label", "seeds": "–", "macro_f1": table["Always"].mean()}]
    models = models_for(args.dataset)
    for name, short in [(FLOOR, "Floor")] + models:
        files = prediction_files(name, args.variant, args.dataset)
        if not files:
            print(f"Skipping {short}: no predictions for {name}")
            continue
        runs = [np.load(f) for f in files]
        f1 = np.mean([per_label_f1(y_test, run["predictions"]) for run in runs], axis=0)[scored]
        auc = np.mean([per_label_auc(y_test, run["scores"].astype(np.float32), auc_columns) for run in runs], axis=0)
        table[short] = f1
        auc_by_label = pd.Series(np.nan, index=np.flatnonzero(scored))
        auc_by_label[auc_columns] = auc
        table[f"{short} AUC"] = auc_by_label.to_numpy()
        row = {"model": short, "seeds": len(files), "macro_f1": f1.mean(),
               "mean_auc": auc.mean(), "labels_auc_signal": int((auc > AUC_SIGNAL).sum()),
               "max_gain": float((f1 - table["Always"]).max())}
        gain = f1 - table["Always"]
        table[f"{short} gain"] = gain
        row["labels_better"] = int((gain > GAIN_MARGIN).sum())
        row["labels_worse"] = int((gain < -GAIN_MARGIN).sum())
        summary.append(row)

    REPORT_DIR.mkdir(exist_ok=True)
    table = table.sort_values("train_frequency", ascending=False)
    table.to_csv(REPORT_DIR / f"per_label_f1_{name_suffix}.csv", index=False, float_format="%.4f")
    write_summary(table, summary, name_suffix, n_auc=len(auc_columns), models=models)
    plot_gains(table, name_suffix)


def write_summary(table, summary, name_suffix, n_auc, models):
    lines = [
        f"# Per-label F1 on test ({name_suffix.replace('_', ' labels, ')} threshold)",
        "",
        f"Generated by `src/per_label_analysis.py`. {len(table)} labels with at least one",
        "positive test example. Per-label F1 is averaged over seeds where several were run.",
        "Reference per label: always predicting that label (its best constant prediction).",
        f"\"Better\"/\"worse\" count labels whose F1 differs from that by more than {GAIN_MARGIN}.",
        "",
        f"Mean AUC is over the {n_auc} labels with both classes in test; the last AUC column counts",
        f"labels with AUC above {AUC_SIGNAL}.",
        "",
        "| model | seeds | macro-F1 | labels better than always | labels worse than always "
        f"| largest F1 gain | mean AUC | labels with AUC > {AUC_SIGNAL} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        fmt = lambda key, spec: format(row[key], spec) if key in row else "–"
        lines.append(f"| {row['model']} | {row['seeds']} | {row['macro_f1']:.4f} "
                     f"| {row.get('labels_better', '–')} | {row.get('labels_worse', '–')} "
                     f"| {fmt('max_gain', '+.3f')} | {fmt('mean_auc', '.3f')} | {row.get('labels_auc_signal', '–')} |")
    for short in [s for _, s in models if f"{s} gain" in table]:
        top = table.nlargest(10, f"{short} gain")
        lines += ["", f"## Largest gains over always-predict: {short}", "",
                  "| label | train frequency | test positives | always F1 | model F1 | gain |",
                  "|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(f"| {r['label']} | {r['train_frequency']:.3f} | {r['test_positives']} "
                         f"| {r['Always']:.3f} | {r[short]:.3f} | {r[f'{short} gain']:+.3f} |")
    (REPORT_DIR / f"per_label_summary_{name_suffix}.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_DIR / f'per_label_summary_{name_suffix}.md'}")


def plot_gains(table, name_suffix):
    panels = [m for m in FIGURE_MODELS if f"{m} gain" in table]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), sharey=True, squeeze=False)
    for ax, short in zip(axes[0], panels):
        gain = table[f"{short} gain"]
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.scatter(table["train_frequency"], gain, s=14, color=SERIES_COLOR,
                   edgecolors="white", linewidths=0.6, zorder=3)
        # Name only the two largest gains, placed to the left so they stay inside the panel.
        for i, (_, r) in enumerate(table.loc[gain.nlargest(2).index].iterrows()):
            ax.annotate(r["label"], (r["train_frequency"], r[f"{short} gain"]), fontsize=6,
                        color=TEXT_PRIMARY, xytext=(-6, -10 * i - 2), textcoords="offset points", ha="right")
        ax.set_title(short, fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlabel("label frequency in train", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xscale("log")
        ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
    axes[0][0].set_ylabel("per-label F1 minus always-predict F1", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = REPORT_DIR / f"per_label_gain_{name_suffix}.{ext}"
        fig.savefig(path, dpi=200)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
