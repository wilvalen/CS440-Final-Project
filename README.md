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
# PyTorch first, the build for your machine (pick one):
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu     # CPU only
pip install -r requirements.txt
```

Fine-tuning was run on an NVIDIA RTX 4000 Ada (20 GB). Everything else runs on CPU.

## Data

The database (441 MB) is not committed. Download it from the official
Hugging Face dataset (checksum is verified):

```console
python src/download_data.py      # -> data/skillscope_data.db
```

### Corrected labels

The official labels are broken. The view that builds them
(`vw_classification_helper`) joins pull requests to their changed files on
the PR number only (`a.pullNumber = b.pullNumber`), without matching the
repository. PR numbers repeat across the 12 repositories, so each issue gets
the labels of every repository's PR with the same number: all issues sharing
a PR number have identical label vectors (only 980 distinct vectors for 7,154
issues), and the labels carry no information about the issue's own text.

`src/build_corrected_labels.py` reruns the official view definitions with one
added condition, `AND a.repoNum = b.repoNum`, and writes
`data/skillscope_corrected.db` (same table name and columns; the official
database is opened read-only). Only PRs with at least one labelled API in
their own changed files remain: 3,070 issues (kafka has almost no recorded
files and drops out).

```console
python src/build_corrected_labels.py   # -> data/skillscope_corrected.db (~15 s)
```

Every script takes `--dataset official` (default) or `--dataset corrected`.

## Split

The competition does not provide a train/test split, so we make one:
a seeded random 80/20 train/test split, with 10% of the training portion
held out as validation for tuning. The split files are committed. To
regenerate them (same output for the same seed):

```console
python src/make_split.py                        # data/split.csv: train 5151, val 572, test 1431
python src/make_split.py --dataset corrected    # data/split_corrected.csv: train 2210, val 246, test 614
```

## Runs

| step | command |
|---|---|
| Sanity floor: same labels for every issue, by training frequency | `python src/label_frequency_floor.py` |
| Baseline: TF-IDF + Random Forest | `python src/baseline_rf.py` |
| Frozen sentence embeddings + one-vs-rest logistic regression | `python src/embeddings_ovr.py` (`--model` picks the encoder; default `all-MiniLM-L6-v2`) |
| Fine-tuned encoder (general), lr candidates | `python src/finetune.py --model roberta-base --prior-bias-init --lr 2e-5` and `--lr 5e-5` |
| Fine-tuned encoder (code-aware) | `python src/finetune.py --model microsoft/codebert-base --prior-bias-init --lr 2e-5` |
| Fine-tuned encoder, text + repo name | `python src/finetune.py --model microsoft/codebert-base --prior-bias-init --lr 2e-5 --with-repo` |
| Per-label F1 and AUC against always-predicting each label (after the runs above) | `python src/per_label_analysis.py` (writes to `report/`) |

Each run writes `results/<run>.json` and rebuilds `results/summary.md`
(with `--dataset corrected`: `results/corrected/`). Runs with a seed other
than 42 get a `_seed<N>` suffix, and the summary adds a mean ± standard
deviation table for runs repeated over seeds.
Each command reports two runs: the model's default decision rule, and a
`_tuned` run that uses one probability threshold for all labels, chosen to
maximise micro-F1 on the validation split (grid 0.01 to 0.99). The test
split is only used for the final numbers.
Add `--sample 300` to a model run for a quick check on a small subset (no results file written).

### Baseline notes

`src/baseline_rf.py` follows `competition_baseline/main.py` from the
[official repo](https://github.com/nlbse2027/skill-classification) (same text
cleaning, 1,000 TF-IDF features, default `RandomForestClassifier(random_state=42)`),
except that TF-IDF is fit on the training split only and MLSMOTE oversampling
is skipped. The original fits TF-IDF on all data and oversamples before
splitting, so synthetic rows leak into its test set, and it needs about 758,000
dense synthetic rows.

### Embedding notes

`src/embeddings_ovr.py` encodes the raw title + description with a frozen
sentence-transformers model (normalised embeddings) and trains one
`LogisticRegression` per label for each C in {0.01, 0.03, 0.1, 0.3, 1, 3, 10}.
C is chosen on the validation split: by micro-F1 at a 0.5 threshold for the
default run, and jointly with the threshold for the `_tuned` run. Every
candidate's validation scores are stored in the result JSON (`val_search`).

Run both encoders with:

```console
python src/embeddings_ovr.py                                               # all-MiniLM-L6-v2, ~1 min on CPU
python src/embeddings_ovr.py --model sentence-transformers/all-mpnet-base-v2  # ~5 min on CPU
```

Embeddings are cached in `data/embeddings/` (not committed);
deleting the folder forces re-encoding, which gives identical results.

### Fine-tuning notes

`src/finetune.py` adds a 217-output classification head to the encoder
(sigmoid per label, `BCEWithLogitsLoss`) and trains with AdamW (weight decay
0.01, 10% linear warmup), batch size 16, max 256 tokens, bf16 autocast, eager
(deterministic) attention, for up to 10 epochs with early stopping (patience 2)
on validation micro-F1. The default run keeps the epoch with the best val
micro-F1 at 0.5; the `_tuned` run keeps the epoch and threshold with the best
tuned val micro-F1.

`--prior-bias-init` sets the output bias to each label's log-odds in the
training split, so the model starts at the label-frequency floor. Without it,
at lr 2e-5 the model spent its whole run learning the label frequencies and
never got past them (log in `report/ft_roberta_lr2e-5_collapsed.log`).

Learning rate was chosen on validation: roberta-base at 2e-5 and 5e-5, the
better (2e-5) reused for codebert-base. The `--with-repo` run uses whichever
of the two models had the higher validation micro-F1 (codebert-base).
Result files are named `ft_<model>[_repo]_lr<lr>[_tuned].json`.

Runs took 4–11 minutes each on an RTX 4000 Ada shared with another job
(50–100 s per epoch). Seeds and deterministic CUDA settings are set, so reruns
on the same GPU and software give the same numbers.
