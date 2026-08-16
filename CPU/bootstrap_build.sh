#!/usr/bin/env bash
# Build the CPU benchmark image (ipex-llm:2.3.100) from a fresh clone, on a VM
# whose only route to the internet is outbound 443.
#
#   bash CPU/bootstrap_build.sh
#
# Everything this script does is a workaround for something that is broken
# upstream or blocked by the network policy; each step says which. It is
# idempotent, so re-run it after a failure rather than unpicking state by hand.
#
# Environment:
#   IMAGE=ipex-llm:2.3.100   tag to build
#   SAVE_IMAGE=0             set to 1 to also write ~/ipex-llm.tgz (see below)
#
# Prerequisite: a working Docker. Run CPU/host_setup.sh first if you do not have
# one; it adds you to the `docker` group, which needs a re-login to take effect.
set -euo pipefail

IMAGE="${IMAGE:-ipex-llm:2.3.100}"
SAVE_IMAGE="${SAVE_IMAGE:-0}"

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
IPEX="$REPO/CPU/intel-extension-for-pytorch"
DOCKERFILE="$IPEX/examples/cpu/inference/python/llm/Dockerfile"

step() { echo; echo "=== $* ==="; }

# --- 1. apt over HTTPS on the generic mirror -------------------------------
# Azure's port-80 mirror is unreachable here. Shared with host_setup.sh.
step "1/8  apt mirrors"
bash "$REPO/CPU/fix_apt_mirrors.sh"

step "1b/8 docker check"
if ! docker info >/dev/null 2>&1; then
    echo "docker is not usable by $(whoami)." >&2
    echo "Run 'HF_TOKEN=<token> CPU/host_setup.sh', then log out and back in so the" >&2
    echo "docker group membership applies, then re-run this script." >&2
    exit 1
fi

# --- 2. IPEX submodule -----------------------------------------------------
step "2/8  IPEX submodule"
if [ ! -e "$IPEX/.git" ]; then
    git -C "$REPO" submodule update --init CPU/intel-extension-for-pytorch
else
    echo "already checked out"
fi

# --- 3. de-link the submodule ----------------------------------------------
# A submodule's .git is a 55-byte file pointing at the parent repo's
# .git/modules/... directory. The Dockerfile copies the checkout into the build
# context, where the parent repo does not exist, so every git call inside the
# build fails with "fatal: not a git repository". scripts/compile_bundle.sh runs
# `git submodule sync` / `git submodule update --init --recursive` during
# env_setup.sh 2, which is where the build dies. Replace the pointer with a real
# directory so the checkout is self-contained.
step "3/8  de-link submodule gitdir"
if [ -f "$IPEX/.git" ]; then
    gitdir="$(sed 's/^gitdir: //' "$IPEX/.git")"
    real="$(cd "$IPEX" && cd "$gitdir" && pwd)"
    echo "gitdir pointer -> $real"
    rm "$IPEX/.git"
    cp -a "$real" "$IPEX/.git"
    # core.worktree points back at the parent checkout and is meaningless now.
    git -C "$IPEX" config --unset core.worktree 2>/dev/null || true
    echo "replaced with a standalone .git directory"
    echo "note: the outer repo now reports this submodule as modified; that is expected."
else
    echo "already standalone"
fi

# IPEX's own third_party submodules (oneDNN, oneCCL, ...) are needed by the
# source build. Cloned over HTTPS; the Dockerfile also sets git's insteadOf
# rewrites so any git:// or SSH remote in the tree is redirected to HTTPS.
git -C "$IPEX" submodule update --init --recursive

# --- 4. apply the fork's patch ---------------------------------------------
step "4/8  apply CPU/ipex.patch"
if git -C "$IPEX" apply --reverse --check "$REPO/CPU/ipex.patch" 2>/dev/null; then
    echo "ipex.patch already applied"
else
    git -C "$IPEX" apply "$REPO/CPU/ipex.patch"
    echo "applied"
fi

# The patch is the only thing standing between a fresh clone and four separate
# build failures, so fail loudly rather than at minute 90 of the build.
for marker in 'COPY host-ca/certs' 'insteadOf' 'COPY --chown=ubuntu:ubuntu prompt.json'; do
    grep -q -- "$marker" "$DOCKERFILE" ||
        { echo "ipex.patch did not apply: '$marker' missing from the Dockerfile" >&2; exit 1; }
done

# --- 5. stage the host CA bundle -------------------------------------------
# ubuntu:22.04 has no CA certificates, and the patched Dockerfile makes the
# container's apt talk HTTPS because port 80 is blocked. That is a cycle: apt
# cannot verify the mirror it would install ca-certificates from. Copy the
# host's bundle into the build context to break it. -L turns the hash symlinks
# into real files so the COPY carries actual certificates.
step "5/8  stage CA certificates"
if [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
    echo "no CA bundle on the host; run: sudo apt-get install -y ca-certificates" >&2
    exit 1
fi
rm -rf "$IPEX/host-ca"
mkdir -p "$IPEX/host-ca"
cp -rL /etc/ssl/certs "$IPEX/host-ca/"
echo "staged $(find "$IPEX/host-ca/certs" -type f | wc -l) files"

# --- 6. stage prompt.json ---------------------------------------------------
# env_setup.sh fetches this from an Intel S3 bucket that now answers 403, then
# symlinks single_instance/ and distributed/ at it -- leaving dangling links and
# a FileNotFoundError at benchmark time. CPU/prompt.json is the committed
# replacement; committing it is also what makes the prompts byte-identical
# between the baseline and the confidential VM. See CPU/make_prompts.py.
step "6/8  stage prompt.json"
cp "$REPO/CPU/prompt.json" "$IPEX/prompt.json"
want="$(sha256sum "$REPO/CPU/prompt.json" | cut -d' ' -f1)"
echo "sha256 $want"

# --- 7. build ---------------------------------------------------------------
# Context is the IPEX checkout itself: the Dockerfile copies the context in and
# builds IPEX out of it. Compiles IPEX, PyTorch, LLVM, oneCCL and DeepSpeed from
# source -- roughly 1-2 hours on a Xeon.
step "7/8  docker build $IMAGE"
DOCKER_BUILDKIT=1 docker build -f "$DOCKERFILE" -t "$IMAGE" "$IPEX"

# --- 8. verify --------------------------------------------------------------
# Checks both that the file was baked in and that the symlinks env_setup.sh
# created resolve to it, which is the thing that actually failed at runtime.
step "8/8  verify prompts in the image"
for p in /home/ubuntu/llm/prompt.json \
         /home/ubuntu/llm/distributed/prompt.json \
         /home/ubuntu/llm/single_instance/prompt.json; do
    got="$(docker run --rm "$IMAGE" sha256sum "$p" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        echo "$p: sha256 $got != $want" >&2
        exit 1
    fi
    echo "ok  $p"
done

if [ "$SAVE_IMAGE" = "1" ]; then
    # Only needed if you want the two VMs to run a bit-identical image. Prompt
    # identity does not depend on it -- CPU/prompt.json is committed -- and the
    # tarball is tens of GB, so this is opt-in.
    step "saving $HOME/ipex-llm.tgz"
    docker save "$IMAGE" | gzip > "$HOME/ipex-llm.tgz"
    ls -lh "$HOME/ipex-llm.tgz"
fi

cat <<EOF

Done. $IMAGE is built and carries the committed prompt pool.

Next:
  cd $REPO/CPU
  export HF_TOKEN=<token>        # gated Llama-2 weights, forwarded into the container
  SWEEP=both VCPU_LIST=0-31 ./run_cpu_sweeps.sh "VM (AMX)"
EOF
