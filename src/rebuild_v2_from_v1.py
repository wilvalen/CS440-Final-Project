"""Recreate the v2 competition table from the v1 database, and check it against v2.

In v1, the view vw_classification_helper, which produces the labels of
nlbse_tool_competition_data_by_issue, joins pull requests to their changed
files on the PR number only:

    WHERE a.pullNumber = b.pullNumber

PR numbers are unique only within a repository, so every issue received the
changed files (and therefore the labels) of the PRs with the same number in
all 12 repositories. The organizers' v2 release fixes the join.

This script runs the v1 view definitions unchanged except for one added
condition, "AND a.repoNum = b.repoNum", and writes the result to
data/skillscope_data_v2_rebuilt.db. If the organizers' v2 database is present
at data/skillscope_data_v2.db, it then checks that the two competition tables
are identical (same rows, order, text and labels). Neither input database is
modified.

If you do not have the organizers' v2 file, the rebuilt database can be used
in its place: copy it to data/skillscope_data_v2.db.

Usage: python src/rebuild_v2_from_v1.py
"""

import sqlite3

import pandas as pd

from data import DB_PATH, REBUILT_V2_DB_PATH, TABLE, V2_DB_PATH, label_columns

REPO_BLIND_JOIN = "a.pullNumber = b.pullNumber"
REPO_MATCHED_JOIN = "a.pullNumber = b.pullNumber\n  AND a.repoNum = b.repoNum"


def view_sql(connection, name):
    return connection.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()[0]


def rebuild_from_v1():
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    helper = view_sql(connection, "vw_classification_helper")
    if helper.count(REPO_BLIND_JOIN) != 1:
        raise ValueError("Unexpected vw_classification_helper definition")
    helper = helper.replace("CREATE VIEW vw_classification_helper", "CREATE TEMP VIEW rebuilt_helper", 1)
    helper = helper.replace(REPO_BLIND_JOIN, REPO_MATCHED_JOIN)

    by_issue = view_sql(connection, "vw_nlbse_tool_competition_data_by_issue")
    by_issue = by_issue.replace(
        "CREATE VIEW vw_nlbse_tool_competition_data_by_issue", "CREATE TEMP VIEW rebuilt_by_issue", 1
    )
    by_issue = by_issue.replace("vw_classification_helper", "rebuilt_helper")

    connection.execute(helper)
    connection.execute(by_issue)
    print("Running the repository-matched views on v1 (about 15 seconds)...")
    rebuilt = pd.read_sql_query("SELECT * FROM rebuilt_by_issue", connection)
    v1_columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{TABLE}")')]
    connection.close()
    if list(rebuilt.columns) != v1_columns:
        raise ValueError("Rebuilt table columns differ from the v1 table")
    return rebuilt


def main():
    rebuilt = rebuild_from_v1()
    y = rebuilt[label_columns(rebuilt)] > 0
    print(f"Issues: {len(rebuilt)}")
    print(f"Labels per issue: mean {y.sum(axis=1).mean():.1f}, median {y.sum(axis=1).median():.0f}")
    print(f"Distinct label vectors: {y.astype(int).astype(str).agg(''.join, axis=1).nunique()}")

    REBUILT_V2_DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(REBUILT_V2_DB_PATH) as out:
        rebuilt.to_sql(TABLE, out, index=False)
    print(f"Wrote {REBUILT_V2_DB_PATH}")

    if not V2_DB_PATH.exists():
        print(f"No organizers' v2 database at {V2_DB_PATH}; skipping the comparison.")
        return
    with sqlite3.connect(f"file:{V2_DB_PATH}?mode=ro", uri=True) as connection:
        v2 = pd.read_sql_query(f'SELECT * FROM "{TABLE}"', connection)
    identical = list(v2.columns) == list(rebuilt.columns) and v2.astype(str).equals(rebuilt.astype(str))
    print(f"Identical to the organizers' v2 table ({len(v2)} issues): {identical}")
    if not identical:
        raise SystemExit("Rebuilt table differs from v2")


if __name__ == "__main__":
    main()
