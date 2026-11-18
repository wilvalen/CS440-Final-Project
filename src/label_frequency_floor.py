"""Sanity floor: ignore the text and predict the same labels for every issue.

Each label's score is its frequency in the training split. Reported twice:

- default: predict every label that is present in more than half of training issues
- tuned: one frequency cutoff, chosen on the "val" split

Any real model should beat this.

Usage: python src/label_frequency_floor.py [--dataset v1|v2]
"""

import argparse
import time

import numpy as np

from data import binary_labels, load_split_data
from evaluate import micro_metrics, save_result, tune_threshold


def main():
    parser = argparse.ArgumentParser(description="Label-frequency sanity floor")
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2",
                        help="v1: dataset as first distributed; v2: organizers' release with the repository join fixed")
    args = parser.parse_args()

    start = time.time()
    splits = load_split_data(args.dataset)
    y_train, y_val, y_test = (binary_labels(splits[name]) for name in ("train", "val", "test"))

    frequency = y_train.mean(axis=0)
    val_scores = np.tile(frequency, (len(y_val), 1))
    test_scores = np.tile(frequency, (len(y_test), 1))

    default_predictions = (test_scores > 0.5).astype(int)
    default_metrics = micro_metrics(y_test, default_predictions)
    threshold, val_f1 = tune_threshold(y_val, val_scores)
    tuned_predictions = (test_scores >= threshold).astype(int)
    tuned_metrics = micro_metrics(y_test, tuned_predictions)
    runtime = time.time() - start
    print("default:", default_metrics)
    print(f"tuned (threshold {threshold}, val micro-F1 {val_f1:.4f}):", tuned_metrics)

    common = {
        "model": "Label-frequency floor (no text input)",
        "dataset": args.dataset,
        "seed": "n/a (deterministic)",
        "hyperparameters": {"score": "label frequency in train split"},
        "runtime_seconds": runtime,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }
    save_result("floor_label_frequency", {
        **common,
        "threshold": "default (frequency > 0.5)",
        "predicted_labels": int((frequency > 0.5).sum()),
        **default_metrics,
    }, test_predictions=default_predictions, test_scores=test_scores)
    save_result("floor_label_frequency_tuned", {
        **common,
        "threshold": f"tuned on val: {threshold}",
        "predicted_labels": int((frequency >= threshold).sum()),
        "val_micro_f1": val_f1,
        **tuned_metrics,
    }, test_predictions=tuned_predictions, test_scores=test_scores)


if __name__ == "__main__":
    main()
