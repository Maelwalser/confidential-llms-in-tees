"""Turn a vLLM latency sweep into cost per million generated tokens.

Usage:  python parse.py <results_directory> [--cost <usd_per_hour>] [--output-len N]

The printed dictionary is meant to be pasted into the gpu_cc_cost / gpu_raw_cost
constants of CPU/processing/vCPUs_batch_size.py and vCPUs_input.py.
"""

import argparse
import glob
import json
import os
import pprint
import re

# benchmark_latency.py measures the wall time of one generate() call that
# produces exactly --output-len tokens per sequence (ignore_eos=True), so the
# tokens produced per run are output_len * batch_size. The sweep does not record
# the output length in the file name, so it has to match benchmark_vllm.sh.
DEFAULT_OUTPUT_LEN = 128

# $/hour for the VM the results were collected on. The confidential and the
# non-confidential VM have different prices, so pass --cost for each.
DEFAULT_COST = 6.98

# Pattern: latency_in{input_len}_bs{batch}.json
PATTERN = re.compile(r"latency_in(\d+)_bs(\d+)\.json$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="directory produced by benchmark_vllm.sh")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST,
                        help=f"VM price in $/hour (default {DEFAULT_COST})")
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN,
                        help=f"tokens generated per sequence (default {DEFAULT_OUTPUT_LEN})")
    args = parser.parse_args()

    cost_by_input_bs = {}

    for path in sorted(glob.glob(os.path.join(args.results_dir, "latency_in*_bs*.json"))):
        match = PATTERN.search(os.path.basename(path))
        if not match:
            continue

        input_len = int(match.group(1))
        batch_size = int(match.group(2))

        with open(path, "r") as f:
            data = json.load(f)

        avg_latency = data.get("avg_latency")
        if avg_latency is None:
            print(f"skipping {path}: no avg_latency")
            continue

        # Throughput counts generated tokens, matching the CPU side
        # (bs / next-token-latency * output tokens) and plot_GPUs.py. Using the
        # input length here instead would overstate throughput by
        # input_len / output_len and understate the cost by the same factor.
        tokens_per_sec = args.output_len * batch_size / avg_latency
        cost_per_million = args.cost / (tokens_per_sec * 3600) * 1e6

        cost_by_input_bs[(input_len, batch_size)] = cost_per_million

    pprint.pprint(cost_by_input_bs)


if __name__ == "__main__":
    main()
