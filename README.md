# CS440 Final Project: NLBSE 2027 Skill Classification (Group 9)

GitHub: https://github.com/wilvalen/CS440-Final-Project

Multi-label classification of GitHub pull requests into 217 skill labels
(domains and sub-domains), based on the
[NLBSE 2027 Skill Classification competition](https://nlbse2027.github.io/tools/).
Models use only the PR title (`issue text`) and description (`issue description`)
as input. Metrics are micro-averaged precision, recall and F1 over all 217 labels.

## Install

Python 3.11.

```console
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The database (441 MB) is not committed. Download it from the official
Hugging Face dataset (checksum is verified):

```console
python src/download_data.py      # -> data/skillscope_data.db
```

## Split

The competition does not provide a train/test split, so we make one:
a seeded random 80/20 train/test split of the 7,154 issues, with 10% of
the training portion held out as validation for tuning. The file
`data/split.csv` is committed. To regenerate it (same output for the same seed):

```console
python src/make_split.py         # train 5151, val 572, test 1431
```

## Runs

| step | command |
|---|---|
| Baseline: TF-IDF + Random Forest | `python src/baseline_rf.py` |

Each run writes `results/<run>.json` and rebuilds `results/summary.md`.
Add `--sample 300` to any run for a quick check on a small subset (no results file written).

### Baseline notes

`src/baseline_rf.py` follows `competition_baseline/main.py` from the
[official repo](https://github.com/nlbse2027/skill-classification) (same text
cleaning, 1,000 TF-IDF features, default `RandomForestClassifier(random_state=42)`),
except that TF-IDF is fit on the training split only and MLSMOTE oversampling
is skipped. The original fits TF-IDF on all data and oversamples before
splitting, so synthetic rows leak into its test set, and it needs about 758,000
dense synthetic rows.
