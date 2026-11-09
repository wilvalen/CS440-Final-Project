"""SkillScope baseline: TF-IDF (1,000 features) + Random Forest.

Follows competition_baseline/main.py from github.com/nlbse2027/skill-classification
(same text cleaning, TfidfVectorizer(max_features=1000), dense features, and
RandomForestClassifier(random_state=42) with default parameters), with two
changes so it fits our evaluation protocol and runs cheaply:

- TF-IDF is fit on the training split only, not on the whole dataset.
- MLSMOTE oversampling is skipped. The original asks for ~758,000 synthetic
  rows and mixes synthetic rows into the test set.

Trains on the "train" split and reports micro precision/recall/F1 on "test"
over all 217 labels.

Usage:
  python src/baseline_rf.py              # full run, writes results/baseline_rf.json
  python src/baseline_rf.py --sample 300 # quick check on 300 train/test rows, no file written
"""

import argparse
import re
import string
import time

import emoji
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from data import binary_labels, load_split_data
from evaluate import micro_metrics, save_result

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


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + Random Forest baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, help="use only this many train and test rows")
    args = parser.parse_args()

    start = time.time()
    splits = load_split_data()
    train, test = splits["train"], splits["test"]
    if args.sample:
        train = train.sample(n=min(args.sample, len(train)), random_state=args.seed)
        test = test.sample(n=min(args.sample, len(test)), random_state=args.seed)
    print(f"Train rows: {len(train)}; test rows: {len(test)}")

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    x_train = tfidf.fit_transform(baseline_text(train)).toarray()
    x_test = tfidf.transform(baseline_text(test)).toarray()
    y_train, y_test = binary_labels(train), binary_labels(test)

    # n_jobs only parallelises tree building; results are identical to n_jobs=None.
    classifier = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    classifier.fit(x_train, y_train)
    metrics = micro_metrics(y_test, classifier.predict(x_test))
    runtime = time.time() - start
    print(metrics, f"runtime {runtime:.1f}s")

    if args.sample:
        return
    save_result("baseline_rf", {
        "model": "TF-IDF + Random Forest (SkillScope baseline, no MLSMOTE)",
        "seed": args.seed,
        "hyperparameters": {
            "tfidf_max_features": MAX_FEATURES,
            "random_forest": "scikit-learn defaults (n_estimators=100)",
            "mlsmote": False,
            "train_split": "train",
        },
        "threshold": "default (RF majority vote)",
        **metrics,
        "runtime_seconds": runtime,
        "n_train": len(train),
        "n_test": len(test),
    })


if __name__ == "__main__":
    main()
