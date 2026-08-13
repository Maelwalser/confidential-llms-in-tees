#!/bin/bash
# Parameterized CPU (TDX/VM) sweep runner for the targeted experiments.
#
# Presets (SWEEP=):
#   batch  -> fixed input 128, batch 1..64, bf16+int8   (paper Fig 4/8/9)
#   input  -> fixed batch 64, input 32..2048, bf16+int8 (paper Fig 10)
#   both   -> batch then input (default)
#
# Usage:
#   SWEEP=input  ./run_cpu_sweeps.sh "TDX (AMX)"     # on the confidential VM
#   SWEEP=both   ./run_cpu_sweeps.sh "VM (AMX)"      # on the non-confidential VM
#
# Env overrides: MODELS, VCPU_LIST, QUANTS, IMAGE.
# Deliberately NOT set -e: one failing config is logged and skipped, not fatal.
set -uo pipefail

config="${1:?Pass a label, e.g. \"TDX (AMX)\" or \"VM (AMX)\"}"
SWEEP="${SWEEP:-both}"

# --- physical-core binding by default (avoids the hyperthread-contention regime) ---
NPROC="$(nproc --all)"
PHYS="$(lscpu | awk -F: '/^Core\(s\) per socket:/{gsub(/ /,"",$2);c=$2} /^Socket\(s\):/{gsub(/ /,"",$2);s=$2} END{if(c&&s)print c*s}')"
[[ -z "${PHYS:-}" ]] && PHYS=$((NPROC / 2))
VCPU_LIST="${VCPU_LIST:-0-$((PHYS - 1))}" # e.g. 0-31 on a 32-phys-core socket
vCPUs_num=$(awk -F- '{print (NF==1)?$1:($2-$1+1)}' <<<"$VCPU_LIST")

MODELS="${MODELS:-meta-llama/Llama-2-7b-hf}"
QUANTS="${QUANTS:-bf16 int8}"
IMAGE="${IMAGE:-ipex-llm:2.3.100}"
OUT_TOKEN=128

# --- sweep definitions -------------------------------------------------------
case "$SWEEP" in
batch) PAIRS=$(for b in 1 2 4 8 16 32 64; do echo "128:$b"; done) ;;
input) PAIRS=$(for i in 32 64 128 256 512 1024 2048; do echo "$i:64"; done) ;;
both) PAIRS=$({
    for b in 1 2 4 8 16 32 64; do echo "128:$b"; done
    for i in 32 64 128 256 512 1024 2048; do echo "$i:64"; done
}) ;;
*)
    echo "SWEEP must be batch|input|both" >&2
    exit 1
    ;;
esac

date=$(date +"%F-%H-%M")
directory="results/$date"
mkdir -p "$directory"
echo "label='$config' sweep=$SWEEP cores=$VCPU_LIST ($vCPUs_num vCPU) -> $directory"
lscpu &>"$directory/lscpu.out"
numactl --hardware &>"$directory/numactl-hw.out" || true

run_one() {
    local in_token="$1" batch_size="$2" model="$3" dt="$4"
    case "$model" in *7b*) mt=7b ;; *13b*) mt=13b ;; *70b*) mt=70b ;; *) mt=other ;; esac
    local name="$directory/${config}-${in_token}in-${OUT_TOKEN}out-${vCPUs_num}vCPU-1s-${batch_size}bs-${mt}-${dt}"
    [[ -s "$name.txt" ]] && grep -q "Finished" "$name.txt" 2>/dev/null && {
        echo "skip: $(basename "$name")"
        return
    }

    local quant=''
    [[ "$dt" == int8 ]] && quant='--ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp'
    local num_iter=50 num_warmup=10 greedy=''
    if [[ "$batch_size" -eq 1 ]]; then greedy='--greedy'; else num_iter=25 num_warmup=5; fi

    local cmd=(docker run --rm --privileged --shm-size=2gb -v "$HOME/.cache:/home/ubuntu/.cache" "$IMAGE" bash -c
        "cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --bind_cores_to_rank --bind_core_list $VCPU_LIST distributed/run_generation_with_deepspeed.py --deployment-mode --profile --benchmark -m $model $quant --ipex --batch-size $batch_size --num-iter $num_iter --num-warmup $num_warmup --max-new-tokens $OUT_TOKEN --input-tokens $in_token --token-latency $greedy")
    echo "${cmd[@]}" >"$name.txt"
    if "${cmd[@]}" &>>"$name.txt"; then
        echo "ok: $(basename "$name")"
    else echo "FAILED: $(basename "$name")" | tee -a "$directory/failed.txt" >&2; fi
}

for model in $MODELS; do
    for pair in $PAIRS; do
        in_token="${pair%%:*}"
        batch_size="${pair##*:}"
        for dt in $QUANTS; do run_one "$in_token" "$batch_size" "$model" "$dt"; done
    done
done
echo "Finished sweep=$SWEEP -> $directory"
