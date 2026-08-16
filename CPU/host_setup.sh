#!/bin/bash

# stop on failure
set -euxo pipefail

# Accept either name: the sweep runners forward HF_TOKEN into the container,
# this script historically took HUGGINGFACE_TOKEN. Fail here rather than with a
# 401 an hour into a sweep.
: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
if [ -z "$HF_TOKEN" ]; then
    set +x
    echo "HF_TOKEN (or HUGGINGFACE_TOKEN) is not set." >&2
    echo "The Llama-2 models are gated; export a token with access to them." >&2
    exit 1
fi

# Azure's default apt mirror is port 80 only, which the benchmark VMs block.
# Also does the apt-get update that the installs below need.
bash "$(dirname "$0")/fix_apt_mirrors.sh"

# Create venv and login. lshw is used by run.sh to record machine provenance;
# under `set -e` a missing lshw aborts the sweep before it starts.
sudo apt install -y python3-venv numactl lshw
python3 -m venv .venv
source .venv/bin/activate
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HF_TOKEN

###### DOCKER #######
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER