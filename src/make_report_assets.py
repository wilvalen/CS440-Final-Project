"""Build the report's tables and figures from results/ and the data. No training.

Writes to report/ (the .tex tables need the LaTeX packages booktabs and graphicx):
  results_table.md / .tex     main results, v1 and v2, default and tuned threshold
                              (mean ± sample std over seeds where several were run)
  dataset_stats.md / .tex     size and label statistics of v1 and v2, with split sizes
  label_by_repo.md / .tex     share of issues with a label, by repository, v1 vs v2
  training_curves.png / .pdf  validation micro-F1 per epoch of the fine-tuned models

Usage: python src/make_report_assets.py
"""

import json
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ROOT, label_columns, load_issues, load_split_data
from evaluate import results_dir

REPORT_DIR = ROOT / "report"
DATASETS = ["v1", "v2"]
# Learning rate chosen on validation for the fine-tuned models on each dataset.
CHOSEN_LR = {"v1": "2e-05", "v2": "5e-05"}
METRICS = ["precision", "recall", "micro_f1"]

# Reference palette, light mode: first three categorical slots, neutral inks.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def models(dataset):
    """(result name without suffixes, display name) in table order."""
    lr = CHOSEN_LR[dataset]
    return [
        ("floor_label_frequency", "Label-frequency floor"),
        ("baseline_rf", "TF-IDF + Random Forest (baseline)"),
        ("emb_all-MiniLM-L6-v2_ovr", "MiniLM embeddings + LR"),
        ("emb_all-mpnet-base-v2_ovr", "mpnet embeddings + LR"),
        (f"ft_roberta-base_lr{lr}", "RoBERTa, fine-tuned"),
        (f"ft_codebert-base_lr{lr}", "CodeBERT, fine-tuned"),
        (f"ft_codebert-base_repo_lr{lr}", "CodeBERT + repo name, fine-tuned"),
    ]


def runs(dataset, name, variant):
    """All seeds' results for one run; variant is "", "_tuned" or "_minprec0.75"."""
    pattern = re.compile(rf"^{re.escape(name)}(_seed\d+)?{re.escape(variant)}\.json$")
    return [json.loads(p.read_text()) for p in sorted(results_dir(dataset).glob("*.json"))
            if pattern.match(p.name)]


def summarize(results):
    """{metric: (mean, std or None)} over seeds."""
    out = {}
    for m in METRICS:
        values = [r[m] for r in results]
        out[m] = (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None)
    return out


def fmt(mean_std, latex=False, with_std=True):
    mean, std = mean_std
    if std is None or not with_std:
        return f"{mean:.3f}"
    return f"{mean:.3f}$\\pm${std:.3f}" if latex else f"{mean:.3f} ± {std:.3f}"


def results_table():
    md = ["# Main results (test set, micro-averaged over all 217 labels)", "",
          "Mean ± sample standard deviation over seeds 42–44 where three seeds were run;",
          "single-seed runs show one value. P = precision, R = recall.", ""]
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Test-set results, micro-averaged over 217 labels. Default: the model's own "
           r"decision rule (0.5, or the Random Forest's vote); tuned: one threshold chosen on validation. "
           r"F1 is mean$\pm$std over three seeds where marked by $n$=3.}",
           r"\label{tab:results}", r"\begin{tabular}{llcccccc}", r"\toprule",
           r"& & \multicolumn{3}{c}{Default threshold} & \multicolumn{3}{c}{Tuned threshold} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"Data & Model & P & R & F1 & P & R & F1 \\", r"\midrule"]
    for dataset in DATASETS:
        md += [f"## {dataset}", "",
               "| model | seeds | default P | default R | default F1 | tuned P | tuned R | tuned F1 |",
               "|---|---|---|---|---|---|---|---|"]
        rows = []
        for name, display in models(dataset):
            default, tuned = runs(dataset, name, ""), runs(dataset, name, "_tuned")
            if not default or not tuned:
                continue
            rows.append((display, len(default), summarize(default), summarize(tuned)))
        constrained = runs(dataset, f"ft_codebert-base_repo_lr{CHOSEN_LR[dataset]}", "_minprec0.75")
        for display, n, d, t in rows:
            md.append(f"| {display} | {n} | " + " | ".join(fmt(d[m]) for m in METRICS) + " | "
                      + " | ".join(fmt(t[m]) for m in METRICS) + " |")
            tex_name = display + (f" ($n$={n})" if n > 1 else "")
            tex.append(f"{dataset} & {tex_name} & "
                       + " & ".join(fmt(d[m], latex=True, with_std=(m == "micro_f1")) for m in METRICS) + " & "
                       + " & ".join(fmt(t[m], latex=True, with_std=(m == "micro_f1")) for m in METRICS) + r" \\")
        if constrained:
            c = summarize(constrained)
            md += ["", f"Precision-constrained threshold (CodeBERT + repo name; best validation F1 with "
                       f"validation precision ≥ 0.75), {len(constrained)} seeds: "
                   + ", ".join(f"{ {'precision': 'precision', 'recall': 'recall', 'micro_f1': 'micro-F1'}[m]} {fmt(c[m])}" for m in METRICS) + "."]
            tex.append(f"{dataset} & CodeBERT + repo, val precision $\\geq$ 0.75 ($n$={len(constrained)}) & "
                       + " & ".join(fmt(c[m], latex=True, with_std=(m == "micro_f1")) for m in METRICS)
                       + r" & -- & -- & -- \\")
        md.append("")
        tex.append(r"\midrule" if dataset != DATASETS[-1] else r"\bottomrule")
    tex += [r"\end{tabular}", r"\end{table*}"]
    write("results_table", md, tex)


def dataset_stats():
    md = ["# Datasets", "",
          "| | issues | labels per issue (mean) | distinct label vectors | labels with a positive "
          "| train / val / test |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{The two dataset versions. Labels are the 217 domain and sub-domain columns.}",
           r"\label{tab:datasets}", r"\begin{tabular}{lrrrrl}", r"\toprule",
           r"& Issues & Labels/issue & Distinct vectors & Labels used & Train/val/test \\", r"\midrule"]
    for dataset in DATASETS:
        df = load_issues(dataset)
        y = (df[label_columns(df)].fillna(0) > 0).astype(int)
        splits = load_split_data(dataset)
        sizes = "/".join(str(len(splits[s])) for s in ("train", "val", "test"))
        cells = [f"{len(df):,}", f"{y.sum(axis=1).mean():.1f}",
                 f"{y.astype(str).agg(''.join, axis=1).nunique():,}", str(int((y.sum() > 0).sum())), sizes]
        md.append(f"| {dataset} | " + " | ".join(cells) + " |")
        tex.append(f"{dataset} & " + " & ".join(c.replace(",", "{,}") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("dataset_stats", md, tex)


def label_by_repo():
    labels = ["Machine Learning", "Computer Graphics", "User Interface", "Parser"]
    repos = ["deeplearning4j", "libgdx", "jabref", "javaparser"]
    shares = {}
    for dataset in DATASETS:
        df = load_issues(dataset)
        y = df[label_columns(df)].fillna(0) > 0
        shares[dataset] = y.groupby(df["Repo Name"]).mean()[labels]
    md = ["# Share of issues with a label, by repository (v1 → v2)", "",
          "In v1 every repository looks alike; in v2 each label follows its repository's domain.", "",
          "| repository | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Share of issues carrying a label, by repository, v1 $\rightarrow$ v2. "
           r"ML = Machine Learning, Graphics = Computer Graphics, UI = User Interface.}",
           r"\label{tab:labelbyrepo}", r"\begin{tabular}{l" + "c" * len(labels) + "}", r"\toprule",
           "Repository & ML & Graphics & UI & Parser" + r" \\", r"\midrule"]
    for repo in repos:
        cells = [f"{shares['v1'].loc[repo, l]:.2f} → {shares['v2'].loc[repo, l]:.2f}" for l in labels]
        md.append(f"| {repo} | " + " | ".join(cells) + " |")
        tex.append(f"{repo} & " + " & ".join(c.replace("→", r"$\rightarrow$") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("label_by_repo", md, tex)


def training_curves():
    # Same y-axis on both panels, so v1's small fluctuations are not magnified.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        lr = CHOSEN_LR[dataset]
        series = [(f"ft_roberta-base_lr{lr}", "RoBERTa"), (f"ft_codebert-base_lr{lr}", "CodeBERT"),
                  (f"ft_codebert-base_repo_lr{lr}", "CodeBERT + repo")]
        for (name, display), color in zip(series, COLORS):
            history = json.loads((results_dir(dataset) / f"{name}.json").read_text())["history"]
            epochs = [h["epoch"] for h in history]
            ax.plot(epochs, [h["val_f1_tuned"] for h in history], color=color, linewidth=2,
                    marker="o", markersize=4, markeredgecolor="white", label=display)
        floor = json.loads((results_dir(dataset) / "floor_label_frequency_tuned.json").read_text())
        ax.axhline(floor["val_micro_f1"], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.annotate("label-frequency floor", (1, floor["val_micro_f1"]), fontsize=7, color=TEXT_SECONDARY,
                    xytext=(0, -10), textcoords="offset points")
        ax.set_title(f"{dataset}", fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlabel("epoch", fontsize=8, color=TEXT_SECONDARY)
        ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
    axes[0].set_ylabel("validation micro-F1 (tuned threshold)", fontsize=8, color=TEXT_SECONDARY)
    axes[0].set_ylim(0.44, 0.70)
    axes[1].legend(fontsize=7, frameon=False, loc="center right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPORT_DIR / f"training_curves.{ext}", dpi=200)
    print(f"Wrote {REPORT_DIR / 'training_curves.png'} and .pdf")


def write(stem, md, tex):
    # Scale each table to the column (or page, for table*) width; needs \usepackage{graphicx}.
    start = next(i for i, line in enumerate(tex) if line.startswith(r"\begin{tabular}"))
    end = tex.index(r"\end{tabular}")
    tex = tex[:start] + [r"\resizebox{\linewidth}{!}{%"] + tex[start:end + 1] + ["}"] + tex[end + 1:]
    (REPORT_DIR / f"{stem}.md").write_text("\n".join(md) + "\n")
    (REPORT_DIR / f"{stem}.tex").write_text("\n".join(tex) + "\n")
    print(f"Wrote {REPORT_DIR / stem}.md and .tex")


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    results_table()
    dataset_stats()
    label_by_repo()
    training_curves()


if __name__ == "__main__":
    main()
