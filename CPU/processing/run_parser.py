"""Gather per-iteration token latencies from result files into results.csv.

Usage:  python run_parser.py <results-dir> [<results-dir> ...]

Every ``.txt`` file below a given directory is treated as a result file.  The
parameters of a run are taken from its file name, which ``run.sh`` builds as

    <system>-<in>in-<out>out-<vCPUs>vCPU-<numa>-<batch>bs-<model>-<dtype>.txt

Only the trailing seven fields are structured, so the system label may itself
contain dashes.  A label that cannot be expressed in a file name at all (one
with spaces or parentheses, e.g. "TDX (no AMX)") can be supplied by placing a
``system.txt`` file next to the results; its first line then overrides the
label for every result file in that directory.
"""

import glob
import os
import csv
import sys
import re

pattern = r"Iteration:\s*((?:[0-9]|\d{2,})),\s*Time:\s*([\d.]+)\s*sec"

SYSTEM_OVERRIDE_FILE = "system.txt"

# Number of dash-separated fields that follow the system label
TRAILING_FIELDS = 7


def read_system_override(directory):
    """Return the label from <directory>/system.txt, or None if absent."""
    override_path = os.path.join(directory, SYSTEM_OVERRIDE_FILE)
    if not os.path.isfile(override_path):
        return None
    with open(override_path, "r") as handle:
        label = handle.readline().strip()
    return label or None


def parse_file_name(file_name):
    """Split a result file name into its parameters.

    Splitting from the right keeps the system label intact when it contains
    dashes.  Returns None when the name does not follow the expected form.
    """
    fields = file_name.rsplit("-", TRAILING_FIELDS)
    if len(fields) != TRAILING_FIELDS + 1:
        return None

    system, in_size, out_size, vCPUs, numa, batch_size, model, data_type = fields
    if not (in_size.endswith("in") and out_size.endswith("out") and vCPUs.endswith("vCPU")):
        return None

    return {
        "system": system,
        "in_size": in_size[: -len("in")],
        "out_size": out_size[: -len("out")],
        "vCPUs": vCPUs[: -len("vCPU")],
        "numa": numa,
        "batch_size": batch_size,
        "model": model.upper(),
        "data_type": data_type,
    }


if len(sys.argv) < 2:
    raise SystemExit(f"usage: {sys.argv[0]} <results-dir> [<results-dir> ...]")

rows = []
skipped = []

# Process each directory provided in the arguments
for directory in sys.argv[1:]:
    system_override = read_system_override(directory)
    if system_override:
        print(f"{directory}: using system label {system_override!r} from {SYSTEM_OVERRIDE_FILE}")

    # Process each .txt file found, at any depth below the directory
    for txt_file in sorted(glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)):
        if os.path.basename(txt_file) == SYSTEM_OVERRIDE_FILE:
            continue

        # Get file name without .txt extension
        file_name = os.path.splitext(os.path.basename(txt_file))[0]
        parameters = parse_file_name(file_name)
        if parameters is None:
            skipped.append(txt_file)
            continue

        if system_override:
            parameters["system"] = system_override

        print(txt_file)
        with open(txt_file, "r") as file:
            for line in file:
                match = re.search(pattern, line.strip())
                if not match:
                    continue

                iteration = int(match.group(1))
                time = float(match.group(2))
                rows.append([
                    parameters["system"],
                    parameters["numa"],
                    parameters["vCPUs"],
                    parameters["batch_size"],
                    parameters["data_type"],
                    parameters["in_size"],
                    parameters["out_size"],
                    parameters["model"],
                    iteration,
                    time,
                ])

# Write header and truncate existing file (if it exists)
with open("./results.csv", "w", newline="") as results:
    writer = csv.writer(results)
    writer.writerow(["system", "numa", "vCPU", "bs", "dt", "in_size", "out_size", "model", "index", "time"])
    writer.writerows(rows)

for path in skipped:
    print(f"skipped (unexpected file name): {path}", file=sys.stderr)

print(f"wrote {len(rows)} rows to results.csv from {len(sys.argv) - 1} directory/-ies")
if not rows:
    print(
        "No iteration timings were found. Check that the directory contains "
        "result .txt files produced by run.sh.",
        file=sys.stderr,
    )
