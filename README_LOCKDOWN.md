# Running the CPU benchmarks on a locked-down VM

The instructions in [`README.md`](README.md) assume unrestricted egress. This
document is the variant for a VM whose only route out is HTTPS, which is how the
Azure baseline (`D64s_v6`) and confidential (`DC64eds_v6`) machines used for the
CPU numbers are configured:

| NSG rule | Priority | Effect |
| --- | --- | --- |
| `AllowAzureDNS` | 300 | `168.63.129.16:53` |
| `AllowVnetOutbound` | 380 | `172.16.0.0/16` |
| `AllowHttpsOutbound` | 400 | TCP 443 |
| `DenyAllOutboundInternet` | 4096 | everything else |

Private subnet, NAT gateway for egress, Bastion for inbound. So: no port 80, no
`git://` (9418), no SSH clones (22). `CPU/bootstrap_build.sh` exists because the
stock build reaches for all three.

## Run order

```sh
git clone --recursive <fork-url>
cd confidential-llms-in-tees

export HF_TOKEN=hf_xxxx          # needs granted access to meta-llama/Llama-2-7b-hf

CPU/host_setup.sh                # apt mirrors, HF login, Docker
# log out and back in so the docker group membership applies

bash CPU/bootstrap_build.sh      # builds ipex-llm:2.3.100  (~1-2 h, compiles from source)

cd CPU
SWEEP=both VCPU_LIST=0-31 ./run_cpu_sweeps.sh "VM (AMX)"    # on the baseline VM
SWEEP=both VCPU_LIST=0-31 ./run_cpu_sweeps.sh "TDX (AMX)"   # on the confidential VM
```

`bootstrap_build.sh` is idempotent: re-run it after a failure instead of unpicking
state by hand. `run_cpu_sweeps.sh` tolerates a failing configuration, logging it to
`failed.txt` and continuing.

Optionally `SAVE_IMAGE=1 bash CPU/bootstrap_build.sh` also writes `~/ipex-llm.tgz`,
which the other VM can `docker load < ~/ipex-llm.tgz` instead of rebuilding. That
is only worth the tens of GB if you want the two VMs to run a bit-identical image;
prompt identity does not depend on it (see below).

## What the lockdown changes, and why

Each of these is a failure that the stock instructions hit on this network.

**apt talks to port 80.** Azure images point apt at `azure.archive.ubuntu.com`
over HTTP. Switching that host to HTTPS is not sufficient — the Azure-optimised
mirror is served over the Azure fabric, not through the NAT gateway, so it times
out too. `CPU/fix_apt_mirrors.sh` rewrites both apt source layouts to the generic
`archive.ubuntu.com` over HTTPS; `host_setup.sh` and `bootstrap_build.sh` both
call it.

**The container's apt has the same problem, plus no certificates.**
`ubuntu:22.04` ships no CA bundle, and `ca-certificates` is installed by the very
`apt` call that now needs to verify an HTTPS mirror. `bootstrap_build.sh` stages
`/etc/ssl/certs` from the host into the build context as `host-ca/`, and the
patched Dockerfile `COPY`s it in before that first `apt`.

**The IPEX submodule's `.git` is a pointer file.** It is 55 bytes reading
`gitdir: ../../.git/modules/intel-extension-for-pytorch`. The Dockerfile copies
the checkout into the build context, where the parent repo does not exist, so
`scripts/compile_bundle.sh` fails at `git submodule update` with *"fatal: not a
git repository"* during `env_setup.sh 2`. `bootstrap_build.sh` replaces the
pointer with a real `.git` directory, which makes the outer repo report the
submodule as modified — expected, and the reason this is done by a script rather
than committed.

**git may reach for blocked ports.** The patched Dockerfile sets
`url."https://github.com/".insteadOf` for both `git://github.com/` and
`git@github.com:` in the base stage, so every clone and transitive submodule URL
goes over 443.

**`prompt.json` no longer exists anywhere.** `tools/env_setup.sh` downloads it
from an Intel S3 bucket that now answers 403 (Intel end-of-lifed IPEX), then
symlinks `single_instance/prompt.json` and `distributed/prompt.json` at it. `ln -s`
succeeds against a missing target, so the failed download leaves two dangling
links and the benchmark dies with
`FileNotFoundError: /home/ubuntu/llm/distributed/prompt.json`. The file is in no
git tag or branch of the 2.3.100+cpu tree.

It is therefore committed here as [`CPU/prompt.json`](CPU/prompt.json)
(sha256 `8b8bdc933f314be4dd1db13285f57010bf99ca507d582e569321971861905a9a`), and
the patched Dockerfile copies it into the image after `env_setup.sh` has run. The
patch also makes the dead download non-fatal.

Committing it is what makes the comparison valid: both VMs get byte-identical
prompts from git, with no image transfer. [`CPU/make_prompts.py`](CPU/make_prompts.py)
documents how it was produced — prompts cut to an *exact* token count with the
Llama-2 tokenizer, because `--input-tokens N` is the independent variable of the
input-length sweep. To re-check the committed file:

```sh
cd CPU
docker run --rm -e HF_TOKEN -v "$PWD:/work" -w /work ipex-llm:2.3.100 \
  bash -lc 'source /home/ubuntu/miniforge3/bin/activate && conda activate py310 && \
            pip install -q transformers && python make_prompts.py --verify prompt.json'
```

It prints `32 -> 32 tokens` … `2048 -> 2048 tokens` and exits non-zero on any
mismatch. Note the absolute path to `activate`: the working directory is not
`/home/ubuntu`, so a relative `../miniforge3/bin/activate` does not resolve.

**The HF token never reached the container.** The models are gated and the
benchmark runs inside Docker, so a host-side `huggingface-cli login` is not
enough — every configuration 401s. Both runners now forward the token with
`docker run -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN`. The valueless form makes
Docker read the values from the runner's environment, which keeps the token out
of the command line, out of the command echoed into every result file, out of
`run.sh`'s `set -x` trace in `run.out`, and out of `ps`.

## Checking a run

The first configuration's result file should show:

- `avx512_core_amx` — AMX is actually being used,
- `*** Prompt size: 128` — matching the requested `--input-tokens`,
- no `FileNotFoundError`, no `401`,
- no `results/<date>/failed.txt`.

Then parse:

```sh
python CPU/processing/run_parser.py CPU/results/<date>-<time>
```

`sha256sum CPU/prompt.json` must match on both VMs; git guarantees it.
