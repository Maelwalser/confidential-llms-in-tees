#!/bin/bash

# exit on failure
set -euxo pipefail

config=$1

# The Llama-2 weights are gated. Logging in on the host is not enough: the
# benchmark runs inside the container, so the token has to be forwarded there or
# every config fails with a 401. Both spellings are exported because transformers
# and huggingface_hub have disagreed about the name across versions.
# `docker run -e NAME` (no value) passes the value through from this environment,
# which keeps the token out of the command line -- and so out of the logged
# command, the `set -x` trace in run.out, and `ps`.
# xtrace is off across this block so the trace does not print the token itself.
set +x
: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "HF_TOKEN (or HUGGINGFACE_TOKEN) is not set; the gated models will 401." >&2
    exit 1
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
set -x

config_num_iter=50
config_num_warmup=10
config_out_token=128
config_in_token=1024

# Machine topology, detected at runtime so that the socket tag in the result
# file names describes the machine the sweep actually runs on. Override with
# CONFIG_PROCS / CONFIG_SOCKETS when detection is not possible or not wanted.
detect_sockets() {
    local sockets=""
    if command -v lscpu &> /dev/null; then
        sockets=$(lscpu | awk -F: '/^Socket\(s\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
    fi
    # Fall back to the number of NUMA nodes, then to a single socket
    if ! [[ $sockets =~ ^[0-9]+$ ]] || (( sockets < 1 )); then
        if command -v numactl &> /dev/null; then
            sockets=$(numactl --hardware | awk '/^available:/ {print $2; exit}')
        fi
    fi
    if ! [[ $sockets =~ ^[0-9]+$ ]] || (( sockets < 1 )); then
        sockets=1
    fi
    printf '%s' "$sockets"
}

config_procs=${CONFIG_PROCS:-$(nproc --all)}
config_sockets=${CONFIG_SOCKETS:-$(detect_sockets)}
# Logical CPUs per socket; on a single-socket machine this is the full CPU
# count, so no run can be tagged as spanning two sockets.
config_procs_per_socket=$(( config_procs / config_sockets ))

# per date folder
date=$(date +"%F-%H-%M")
directory=results/$date
mkdir -p $directory

echo "storing results in $directory"
echo "detected topology: $config_procs vCPUs, $config_sockets socket(s), $config_procs_per_socket vCPUs per socket"
echo "$1 stored in $directory" >> experiment.log

{
    lscpu &> $directory/lscpu.out
    lshw &> $directory/lshw.out
    numactl --hardware &> $directory/numactl-hw.out

    # initialize variables with different values
    for vCPUs in '1' '1-2' '1-4' '1-8' '1-16' '1-32' '1-48' '0-59'; do # if you want to use all cores available to the system just leave empty ''; if you want to use cores accross sockets, you can use `--num_accelerators 2` in the main command
        for batch_size in 1 2 4 8 16 32 64 128; do
            for in_token in 32 64 128 256 512 1024 2048; do
                for out_token in 128; do
                    for quant in '' '--ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp'; do # needs first quantizing
                        for model in 'meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Llama-2-70b-hf'; do # 
                            # other working options 'EleutherAI/gpt-j-6b' 'tiiuae/falcon-7b' 'baichuan-inc/Baichuan2-7B-Chat' 'Qwen/Qwen-7B-Chat' 'meta-llama/Meta-Llama-3-8B'
                            # for these you need to modify the name outputting
                            # not working 'mosaicml/mpt-7b' (error) 'liuhaotian/llava-v1.5-7b' (no class) 'mistralai/Mistral-7B-v0.1' (error)                    
                            num_iter=$config_num_iter
                            num_warmup=$config_num_warmup
                            # cmp output name
                            name=$directory/$1
                            name=$name-${in_token}in
                            name=$name-${out_token}out
                            # keep $vCPUs intact: it is the loop variable and is
                            # re-read by every following iteration
                            if [[ -n ${vCPUs//[[:space:]]/} ]]; then
                                vCPUs_num=$(awk -F- '{print (NF==1)?$1:($2-$1+1)}' <<< "$vCPUs")
                                bind_cores="--bind_core_list $vCPUs"
                            else
                                vCPUs_num=$config_procs
                                bind_cores=''
                            fi
                            name=$name-${vCPUs_num}vCPU
                            if (( vCPUs_num > config_procs_per_socket )); then
                                numa='2s'
                            else
                                numa='1s'
                            fi
                            name=$name-$numa
                            name=$name-${batch_size}bs
                            if [[ $model == *"7b"* ]]; then
                                name=$name-7b
                            elif [[ $model == *"13b"* ]]; then
                                name=$name-13b
                            elif [[ $model == *"70b"* ]]; then
                                name=$name-70b
                                num_iter=$(( $num_iter/2 ))
                                num_warmup=$(( $num_warmup/2 ))
                            fi
                            if [ ! -z "${quant}" ]; then
                                name=$name-int8
                            else
                                name=$name-bf16
                            fi
                            # set greedy if single batch
                            greedy=''
                            if [[ "$batch_size" -eq 1 ]]; then
                                greedy='--greedy'
                            else
                                num_iter=$(( $num_iter/2 ))
                                num_warmup=$(( $num_warmup/2 ))
                            fi

                            # safety
                            if [[ "$num_iter" -le 1 ]]; then
                                num_iter=5
                            fi
                            if [[ "$num_warmup" -le 1 ]]; then
                                num_warmup=2
                            fi

                            cmd=(docker run --rm --privileged --shm-size="2gb" \
                                -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN \
                                -v $HOME/.cache:/home/ubuntu/.cache ipex-llm:2.3.100 bash -c \
                                "cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --bind_cores_to_rank $bind_cores distributed/run_generation_with_deepspeed.py --deployment-mode --profile --benchmark -m $model $quant --ipex --batch-size $batch_size --num-iter $num_iter --num-warmup $num_warmup --max-new-tokens $out_token --input-tokens $in_token --token-latency $greedy" )

                            # log cmd
                            echo "${cmd[@]}" > $name.txt

                            # run cmd
                            "${cmd[@]}" &>> $name.txt
                        done
                    done
                done
            done
        done
    done

    # Finished run
    echo "Finished"
# store run log
} &> $directory/run.out


