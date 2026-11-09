"""Create the fixed train/val/test split used by every model.

80% train / 20% test, chosen uniformly at random with a fixed seed. A
validation set (10% of the training portion) is carved from train for any
tuning. The result is saved to data/split.csv, keyed by (Repo Name, PR #).

Usage: python src/make_split.py
"""

import argparse

import numpy as np

from data import DB_PATH, KEY_COLUMNS, SPLIT_PATH, load_issues

TEST_FRACTION = 0.2
VAL_FRACTION_OF_TRAIN = 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_issues(DB_PATH)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))

    n_test = round(len(df) * TEST_FRACTION)
    n_val = round((len(df) - n_test) * VAL_FRACTION_OF_TRAIN)
    split = np.empty(len(df), dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_val]] = "val"
    split[order[n_test + n_val:]] = "train"

    out = df[KEY_COLUMNS].copy()
    out["split"] = split
    out = out.sort_values(KEY_COLUMNS).reset_index(drop=True)
    out.to_csv(SPLIT_PATH, index=False)
    print(out["split"].value_counts().to_string())
    print(f"Wrote {SPLIT_PATH}")


if __name__ == "__main__":
    main()
