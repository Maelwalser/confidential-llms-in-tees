"""Shared loading helpers for the CSV produced by ``run_parser.py``.

The plotting scripts select rows by exact string match on labels that come
from the result file names (``system``, ``numa``, ``dt``, model size).  A
label that does not match yields an empty data frame and therefore an empty
figure, with no error to explain it.  The helpers here turn both of those
failure modes into a message that names the mismatch.
"""

import pandas as pd

# ``run_parser.py`` writes the model size under "model"; the plotting scripts
# were written against a CSV that called the same column "size".  Both spellings
# are accepted so that either vintage of CSV can be plotted unchanged.
COLUMN_ALIASES = (("model", "size"), ("size", "model"))

REQUIRED_COLUMNS = ("system", "numa", "vCPU", "bs", "dt", "in_size", "out_size", "index", "time")


def load_results(path):
    """Read a results CSV and normalise its column names.

    Guarantees that both ``model`` and ``size`` are present, whichever one the
    file was written with.
    """
    df = pd.read_csv(path)

    for source, alias in COLUMN_ALIASES:
        if source in df.columns and alias not in df.columns:
            df[alias] = df[source]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if "model" not in df.columns:
        missing.append("model (or size)")
    if missing:
        raise SystemExit(
            f"{path} is missing the column(s) {missing}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Regenerate the CSV with processing/run_parser.py."
        )

    return df


def require_rows(df, source, path, **requested):
    """Fail with an explanation when filtering removed every row.

    ``source`` is the unfiltered frame, used to report which labels the CSV
    actually contains.  ``requested`` maps a column name to the value (or list
    of values) the caller filtered on.
    """
    if len(df) > 0:
        return df

    lines = [f"No rows left after filtering {path}. Requested vs. available:"]
    for column, wanted in requested.items():
        if column in source.columns:
            available = sorted(str(v) for v in source[column].unique())
        else:
            available = "<column not in CSV>"
        lines.append(f"  {column:<8} requested {wanted!r}")
        lines.append(f"  {'':<8} available {available}")
    lines.append(
        "Adjust the constants at the top of this script to match the labels in "
        "the CSV, or re-run the parser with the labels you want."
    )
    raise SystemExit("\n".join(lines))
