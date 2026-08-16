#!/usr/bin/env python3
"""Generate (or verify) the benchmark prompt pool that IPEX expects as prompt.json.

Upstream `tools/env_setup.sh` downloads this file from
`https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json`.
Intel end-of-lifed IPEX and that bucket now answers 403; the file is in no git
tag or branch of the 2.3.100+cpu source tree, so it is gone. The benchmark still
requires it: `run_generation_with_deepspeed.py` reads
`prompt_pool[model_type][str(input_tokens)]` and dies with FileNotFoundError
otherwise.

The generated pool is committed as `CPU/prompt.json` so that every VM in a
comparison runs byte-identical prompts. This script exists to document how that
file was produced and to check it, not because it needs regenerating.

Prompts are cut to an *exact* token count with the real model tokenizer, because
`--input-tokens N` is the independent variable of the input-length sweep: if the
prompt were merely approximately N tokens the x-axis would be wrong.

Both modes need the Llama-2 tokenizer, which is gated, so `HF_TOKEN` must be set
and the account must have been granted access to `meta-llama/Llama-2-7b-hf`. The
tokenizer lives inside the benchmark image, so run this there:

    docker run --rm -e HF_TOKEN="$HF_TOKEN" -v "$PWD:/work" -w /work ipex-llm:2.3.100 \
      bash -lc 'source /home/ubuntu/miniforge3/bin/activate && conda activate py310 && \
                pip install -q transformers && python make_prompts.py --verify prompt.json'

Note the absolute path to `activate`: the container's working directory is not
`/home/ubuntu`, so a relative `../miniforge3/bin/activate` does not resolve.
"""

import argparse
import json
import os
import sys

# The input lengths swept by run.sh / run_cpu_sweeps.sh.
SIZES = [32, 64, 128, 256, 512, 1024, 2048]

# The upstream file keyed the pool by model type. run_generation_with_deepspeed.py
# looks up the type it inferred from the model config, so every key has to exist;
# the prompts themselves are the same text for all of them.
MODEL_TYPES = ["llama", "gpt-j", "gpt-neox", "auto"]

# Filler text, taken from the prompt the upstream pool used, repeated to give the
# cutter more words than it can need for the longest size.
SEED = (
    "It is done, and submitted. You can play Survival of the Tastiest on Android "
    "and on the web. There is a lot I would like to talk about. I will go through "
    "every topic. "
)


def build(tokenizer, n):
    """Return text that tokenizes to exactly n tokens."""
    words = (SEED * 400).split()
    text = " ".join(words[: max(4, n)])
    ids = tokenizer(text).input_ids
    # Tokens are shorter than words, so grow until there is something to cut back
    # to. n + 1 because the cut below drops the BOS token that Llama prepends.
    while len(ids) < n + 1:
        text = text + " " + " ".join(words[:16])
        ids = tokenizer(text).input_ids
    return tokenizer.decode(ids[:n], skip_special_tokens=True)


def load_tokenizer(model):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers is not installed. Run this inside the benchmark image.")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN is not set. The Llama-2 tokenizer is gated; export a token "
            "with access to the model and pass it into the container with -e HF_TOKEN."
        )
    return AutoTokenizer.from_pretrained(model, token=token)


def report(tokenizer, pool):
    """Print the token count of every prompt. Returns True if all are exact."""
    ok = True
    for n in SIZES:
        text = pool.get(str(n))
        if text is None:
            print(f"{n:>5} -> MISSING")
            ok = False
            continue
        got = len(tokenizer(text).input_ids)
        print(f"{n:>5} -> {got} tokens{'' if got == n else '   <-- MISMATCH'}")
        ok = ok and got == n
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="tokenizer to cut prompts with (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default="prompt.json", help="where to write (default: %(default)s)"
    )
    parser.add_argument(
        "--verify",
        metavar="PATH",
        help="check an existing pool instead of generating one; "
        "exits non-zero unless every prompt is exactly its nominal token count",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)

    if args.verify:
        with open(args.verify, encoding="utf-8") as handle:
            pool = json.load(handle)
        missing = [t for t in MODEL_TYPES if t not in pool]
        if missing:
            sys.exit(f"{args.verify} is missing model types: {', '.join(missing)}")
        ok = True
        for model_type in MODEL_TYPES:
            print(f"[{model_type}]")
            ok = report(tokenizer, pool[model_type]) and ok
        if not ok:
            sys.exit(f"{args.verify} does not tokenize to the expected lengths")
        print(f"{args.verify}: all prompts are exact")
        return

    prompts = {str(n): build(tokenizer, n) for n in SIZES}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({t: prompts for t in MODEL_TYPES}, handle)
    print(f"wrote {args.out}")
    if not report(tokenizer, prompts):
        sys.exit("generated prompts are not exact")


if __name__ == "__main__":
    main()
