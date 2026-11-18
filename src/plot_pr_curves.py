"""Micro-averaged precision-recall curves on test, with each run's operating points.

If the baseline's operating points (default and tuned threshold) lie below a
model's curve, that model is better at every precision/recall trade-off, not
only at the threshold we happened to choose. The curves are drawn from saved
test scores; nothing here chooses anything on test.

Each curve uses the scores of the run's "_tuned" checkpoint. Operating points
are read from the result JSON files; a run's default point can come from a
different epoch than its tuned checkpoint, so it may sit slightly off the curve.

Usage: python src/plot_pr_curves.py [--dataset v1|v2]
Writes report/pr_curves_<dataset>.png and .pdf.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from data import ROOT, binary_labels, load_split_data
from evaluate import predictions_dir, results_dir

REPORT_DIR = ROOT / "report"
# Reference palette, light mode: the first three categorical slots (validated all-pairs).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
# Curves per dataset: (run name without suffix, legend name). The best model by
# validation micro-F1 on each dataset, the baseline, and the floor.
CURVES = {
    "v1": [
        ("ft_codebert-base_lr2e-05", "CodeBERT (fine-tuned)"),
        ("baseline_rf", "TF-IDF + RF (baseline)"),
        ("floor_label_frequency", "Label-frequency floor"),
    ],
    "v2": [
        ("ft_codebert-base_repo_lr5e-05", "CodeBERT + repo (fine-tuned)"),
        ("baseline_rf", "TF-IDF + RF (baseline)"),
        ("floor_label_frequency", "Label-frequency floor"),
    ],
}
# Operating points to mark: result-name suffix -> short label.
POINTS = {"": "default", "_tuned": "tuned", "_minprec0.75": "precision ≥ 0.75 on val"}


def main():
    parser = argparse.ArgumentParser(description="Micro precision-recall curves on test")
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    y_test = binary_labels(load_split_data(args.dataset)["test"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    # Floor labels go below their points so they don't collide with the baseline's.
    offsets = [(6, 4), (6, 4), (6, -12)]
    for (name, legend), color, offset in zip(CURVES[args.dataset], COLORS, offsets):
        scores = np.load(predictions_dir(args.dataset) / f"{name}_tuned.npz")["scores"].astype(np.float32)
        precision, recall, _ = precision_recall_curve(y_test.ravel(), scores.ravel())
        ax.plot(recall, precision, color=color, linewidth=2, label=legend, zorder=2)
        for suffix, point_label in POINTS.items():
            path = results_dir(args.dataset) / f"{name}{suffix}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            ax.scatter(r["recall"], r["precision"], s=40, color=color, edgecolors="white",
                       linewidths=2, zorder=3)
            ax.annotate(point_label, (r["recall"], r["precision"]), fontsize=7, color=TEXT_PRIMARY,
                        xytext=offset, textcoords="offset points")

    ax.set_xlabel("micro recall", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("micro precision", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(f"Precision-recall on test ({args.dataset} labels)", fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    REPORT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        path = REPORT_DIR / f"pr_curves_{args.dataset}.{ext}"
        fig.savefig(path, dpi=200)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
