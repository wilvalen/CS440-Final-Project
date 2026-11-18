"""SkillScope baseline: TF-IDF (1,000 features) + Random Forest.

Follows competition_baseline/main.py from github.com/nlbse2027/skill-classification
(same text cleaning, TfidfVectorizer(max_features=1000), dense features, and
RandomForestClassifier(random_state=42) with default parameters), with two
changes so it fits our evaluation protocol and runs cheaply:

- TF-IDF is fit on the training split only, not on the whole dataset.
- MLSMOTE oversampling is skipped. The original asks for ~758,000 synthetic
  rows and mixes synthetic rows into the test set.

Trains on the "train" split and reports micro precision/recall/F1 on "test"
over all 217 labels, twice:

- default: the Random Forest's own prediction (majority vote of the trees)
- tuned: one probability threshold for all labels, chosen on the "val" split

Usage:
  python src/baseline_rf.py              # full run, writes results/baseline_rf.json and baseline_rf_tuned.json
  python src/baseline_rf.py --sample 300 # quick check on 300 train/val/test rows, no files written
"""

import argparse
import re
import string
import time

import emoji
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from data import binary_labels, load_split_data
from evaluate import micro_metrics, save_result, seed_tag, tune_threshold

MAX_FEATURES = 1000


def clean_text(text):
    """Text cleaning copied from the official baseline (clean_text)."""
    text = text.replace('"', "")
    text = re.sub(r"DevTools.*?\(automated\)", "", text)
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    words = [word for word in text.split() if len(word) <= 20]
    return " ".join(words)


def baseline_text(df):
    """Clean title and description separately, then join them, as the original does."""
    titles = df["issue text"].fillna("").map(clean_text)
    descriptions = df["issue description"].fillna("").map(clean_text)
    return (titles + " " + descriptions).tolist()


def positive_probabilities(classifier, x):
    """Probability of label present, as an (n_rows x n_labels) array.

    For a label with no positive training example the forest only knows
    class 0, so its probability of being present is 0.
    """
    columns = []
    for classes, proba in zip(classifier.classes_, classifier.predict_proba(x)):
        has_positive = 1 in classes
        columns.append(proba[:, list(classes).index(1)] if has_positive else np.zeros(len(x)))
    return np.column_stack(columns)


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + Random Forest baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2",
                        help="v1: dataset as first distributed; v2: organizers' release with the repository join fixed")
    parser.add_argument("--sample", type=int, help="use only this many rows of each split")
    args = parser.parse_args()

    start = time.time()
    splits = load_split_data(args.dataset)
    if args.sample:
        splits = {name: df.sample(n=min(args.sample, len(df)), random_state=args.seed)
                  for name, df in splits.items()}
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"Train rows: {len(train)}; val rows: {len(val)}; test rows: {len(test)}")

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    x_train = tfidf.fit_transform(baseline_text(train)).toarray()
    x_val = tfidf.transform(baseline_text(val)).toarray()
    x_test = tfidf.transform(baseline_text(test)).toarray()
    y_train, y_val, y_test = binary_labels(train), binary_labels(val), binary_labels(test)

    # n_jobs only parallelises tree building; results are identical to n_jobs=None.
    classifier = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    classifier.fit(x_train, y_train)
    train_seconds = time.time() - start

    default_predictions = classifier.predict(x_test)
    default_metrics = micro_metrics(y_test, default_predictions)
    print("default:", default_metrics)

    threshold, val_f1 = tune_threshold(y_val, positive_probabilities(classifier, x_val))
    test_scores = positive_probabilities(classifier, x_test)
    tuned_predictions = (test_scores >= threshold).astype(int)
    tuned_metrics = micro_metrics(y_test, tuned_predictions)
    runtime = time.time() - start
    print(f"tuned (threshold {threshold}, val micro-F1 {val_f1:.4f}):", tuned_metrics)
    print(f"runtime {runtime:.1f}s")

    if args.sample:
        return
    common = {
        "dataset": args.dataset,
        "seed": args.seed,
        "hyperparameters": {
            "tfidf_max_features": MAX_FEATURES,
            "random_forest": "scikit-learn defaults (n_estimators=100)",
            "mlsmote": False,
            "train_split": "train",
        },
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    tag = seed_tag(args.seed)
    save_result(f"baseline_rf{tag}", {
        "model": "TF-IDF + Random Forest (SkillScope baseline, no MLSMOTE)",
        "threshold": "default (RF majority vote)",
        **default_metrics,
        "runtime_seconds": train_seconds,
        **common,
    }, test_predictions=default_predictions, test_scores=test_scores)
    save_result(f"baseline_rf{tag}_tuned", {
        "model": "TF-IDF + Random Forest (SkillScope baseline, no MLSMOTE)",
        "threshold": f"tuned on val: {threshold}",
        **tuned_metrics,
        "val_micro_f1": val_f1,
        "runtime_seconds": runtime,
        **common,
    }, test_predictions=tuned_predictions, test_scores=test_scores)


if __name__ == "__main__":
    main()
