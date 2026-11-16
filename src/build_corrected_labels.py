"""Rebuild the competition table with the repository join fixed.

The official view vw_classification_helper, which produces the labels of
nlbse_tool_competition_data_by_issue, joins pull requests to their changed
files on the PR number only:

    WHERE a.pullNumber = b.pullNumber

PR numbers are unique only within a repository, so every issue gets the
changed files (and therefore the labels) of the PRs with the same number in
all 12 repositories. Issues that share a PR number have identical label vectors.

This script runs the official view definitions unchanged except for one added
condition, "AND a.repoNum = b.repoNum", and writes the result to
data/skillscope_corrected.db with the same table name and columns. PRs with no
labelled API in their own changed files drop out of the inner join, as in the
official view. The official database is opened read-only and not modified.

Usage: python src/build_corrected_labels.py
"""

import sqlite3

import pandas as pd

from data import CORRECTED_DB_PATH, DB_PATH, TABLE, label_columns

BUGGY_JOIN = "a.pullNumber = b.pullNumber"
FIXED_JOIN = "a.pullNumber = b.pullNumber\n  AND a.repoNum = b.repoNum"


def view_sql(connection, name):
    return connection.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()[0]


def main():
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    helper = view_sql(connection, "vw_classification_helper")
    if helper.count(BUGGY_JOIN) != 1:
        raise ValueError("Unexpected vw_classification_helper definition")
    helper = helper.replace("CREATE VIEW vw_classification_helper", "CREATE TEMP VIEW corrected_helper", 1)
    helper = helper.replace(BUGGY_JOIN, FIXED_JOIN)

    by_issue = view_sql(connection, "vw_nlbse_tool_competition_data_by_issue")
    by_issue = by_issue.replace(
        "CREATE VIEW vw_nlbse_tool_competition_data_by_issue", "CREATE TEMP VIEW corrected_by_issue", 1
    )
    by_issue = by_issue.replace("vw_classification_helper", "corrected_helper")

    connection.execute(helper)
    connection.execute(by_issue)
    print("Running the corrected views (takes a minute or two)...")
    corrected = pd.read_sql_query("SELECT * FROM corrected_by_issue", connection)
    official_columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{TABLE}")')]
    connection.close()
    if list(corrected.columns) != official_columns:
        raise ValueError("Corrected table columns differ from the official table")

    y = corrected[label_columns(corrected)] > 0
    print(f"Issues: {len(corrected)}")
    print(f"Labels per issue: mean {y.sum(axis=1).mean():.1f}, median {y.sum(axis=1).median():.0f}")
    print(f"Labels with at least one positive: {(y.sum() > 0).sum()}")
    print(f"Distinct label vectors: {y.astype(int).astype(str).agg(''.join, axis=1).nunique()}")
    print(corrected["Repo Name"].value_counts().to_string())

    CORRECTED_DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(CORRECTED_DB_PATH) as out:
        corrected.to_sql(TABLE, out, index=False)
    print(f"Wrote {CORRECTED_DB_PATH}")


if __name__ == "__main__":
    main()
