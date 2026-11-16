"""Loading the competition table, its inputs, its labels, and our fixed split.

Two datasets share the same table layout:
- "official":  data/skillscope_data.db as distributed by the competition
- "corrected": data/skillscope_corrected.db, built by src/build_corrected_labels.py
  (labels recomputed with the PR-to-file join fixed to match on repository)
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "skillscope_data.db"
SPLIT_PATH = ROOT / "data" / "split.csv"
CORRECTED_DB_PATH = ROOT / "data" / "skillscope_corrected.db"
CORRECTED_SPLIT_PATH = ROOT / "data" / "split_corrected.csv"
DATASETS = {
    "official": (DB_PATH, SPLIT_PATH),
    "corrected": (CORRECTED_DB_PATH, CORRECTED_SPLIT_PATH),
}
TABLE = "nlbse_tool_competition_data_by_issue"

# The first 9 columns of the table are metadata / inputs. Everything after
# them is a label column (217 domain and domain-subdomain API counts).
METADATA_COLUMNS = [
    "Repo Name",
    "PR #",
    "issue text",
    "issue description",
    "created_at",
    "closed_at",
    "userlogin",
    "author_name",
    "most_recent_commit",
]
KEY_COLUMNS = ["Repo Name", "PR #"]
N_LABELS = 217


def load_issues(dataset="official"):
    """Return the full competition table of one dataset as a DataFrame."""
    db_path = DATASETS[dataset][0]
    with sqlite3.connect(db_path) as connection:
        df = pd.read_sql_query(f'SELECT * FROM "{TABLE}"', connection)
    if list(df.columns[: len(METADATA_COLUMNS)]) != METADATA_COLUMNS:
        raise ValueError("Unexpected column layout in the competition table")
    return df


def label_columns(df):
    """All 217 label columns, in table order."""
    columns = list(df.columns[len(METADATA_COLUMNS):])
    if len(columns) != N_LABELS:
        raise ValueError(f"Expected {N_LABELS} label columns, found {len(columns)}")
    return columns


def binary_labels(df):
    """Label matrix (n_issues x 217) of 0/1: a count above zero means present."""
    return (df[label_columns(df)].fillna(0) > 0).astype(int).to_numpy()


def issue_text(df):
    """Title and description joined with a space. These are the only model inputs."""
    return (df["issue text"].fillna("") + " " + df["issue description"].fillna("")).tolist()


def load_split_data(dataset="official"):
    """Return {"train": df, "val": df, "test": df} using the dataset's saved split file."""
    df = load_issues(dataset)
    split = pd.read_csv(DATASETS[dataset][1])
    merged = df.merge(split, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(df):
        raise ValueError("Split file does not cover every issue in the table")
    return {
        name: part.drop(columns="split").reset_index(drop=True)
        for name, part in merged.groupby("split")
    }
