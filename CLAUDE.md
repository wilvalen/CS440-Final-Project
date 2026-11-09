# Project: NLBSE 2027 Skill Classification (course assignment)

A 2-week course assignment based on the NLBSE 2027 Skill Classification tool
competition. This is for a class, not a competition entry.

- Group name: [GROUP NAME]
- Due date: [DATE]
- Compute: [e.g. "local NVIDIA GPU, 8 GB" or "no GPU, full training runs on Colab"]
- GitHub repo: [LINK]

## Task

Multi-label classification. For each issue/pull request, predict which of 217
skill labels (domains and sub-domains) apply.

- Competition page: https://nlbse2027.github.io/tools/
- Official repo (data + baseline): github.com/nlbse2027/skill-classification
- Data: `skillscope_data.db` (SQLite), table `nlbse_tool_competition_data_by_issue`
- Label columns hold counts of matching APIs; a count above zero means the
  label is present.
- Baseline: TF-IDF + Random Forest, in the `competition_baseline` folder.
- Metrics: precision, recall, micro-F1. Always report all three.

## Goal

Fine-tune a pretrained model that beats the baseline, and produce a clear
comparison for a written report:

1. Baseline (TF-IDF + Random Forest), reproduced
2. Frozen sentence embeddings + one-vs-rest classifier
3. Fine-tuned encoder (one general model, one code-aware model), sigmoid
   output per label, binary cross-entropy loss
4. Each of the above with and without decision-threshold tuning

## Hard rules

- Never use label columns, or anything derived from them, as model inputs.
  If unsure whether a column leaks labels, stop and ask.
- Use the provided train/test split. Tune only on a validation set carved
  from the training data. Never tune on, or peek at, the test set.
- Do not use a third-party classification service or API as the classifier.
- Set random seeds everywhere. Every reported number must be reproducible
  from a single documented command.
- Never invent or estimate results. Numbers in the README or results files
  come only from actual runs.

## Working style

- Work one step at a time and report back before moving to the next.
- Ask before starting any run expected to take more than about 10 minutes.
- Test new code on a small data sample before a full run.
- If there is no local GPU, keep training code runnable both locally (small
  sample) and on Colab (full run) without edits beyond a config flag.
- Prefer simple, readable code over clever code. A grader has to run it.

## Repo layout

```
data/            database file (not committed if large; README says how to get it)
src/             data loading, models, training, evaluation
notebooks/       exploration and Colab training notebook
results/         metrics as JSON/CSV, one file per run, plus a summary table
report/          figures and tables for the written report
README.md
requirements.txt
```

## Results logging

Every run writes a file to `results/` containing: model name, seed,
hyperparameters, threshold setting, precision, recall, micro-F1, and runtime.
Keep `results/summary.md` updated with one comparison table of all runs.

## Deliverables (keep these current as the project evolves)

- `README.md`: how to install and run everything, plus the GitHub repo link
- `requirements.txt`: updated whenever a dependency changes
- All code and files needed to test and run the project
- Final package: `[GROUP NAME].zip`
- The report is written separately (4+ pages, venue format, sections:
  Abstract, Introduction, Related Work, Method, Results, Discussion,
  Conclusion and Future Work). Code work should produce the tables and
  figures it needs in `report/`.
