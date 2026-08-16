#!/usr/bin/env bash
# Point apt at the generic Ubuntu mirrors over HTTPS, then refresh the index.
#
# Azure VM images ship apt sources pointing at azure.archive.ubuntu.com over
# port 80. On a locked-down VM (NSG allowing only DNS, VNet and outbound 443,
# with egress through a NAT gateway) port 80 is closed, so `apt-get update`
# stalls until it times out. Switching that same host to HTTPS is not enough
# either: the Azure-optimised mirror is served over the Azure fabric rather than
# through the NAT gateway, so it times out as well. The generic mirror over
# HTTPS is the combination that works.
#
# Both apt source layouts are handled: deb822 (`ubuntu.sources`, Ubuntu 24.04+)
# and the older one-line `sources.list`. Idempotent -- re-running on an
# already-correct machine only re-runs `apt-get update`.
set -euo pipefail

changed=""
for f in /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; do
    [ -f "$f" ] || continue
    before="$(mktemp)"
    cp "$f" "$before"
    sudo sed -i \
        -e 's|https\?://azure.archive.ubuntu.com|https://archive.ubuntu.com|g' \
        -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
        -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
        "$f"
    cmp -s "$before" "$f" || changed="$changed $f"
    rm -f "$before"
done

if [ -n "$changed" ]; then
    echo "apt mirrors rewritten to HTTPS in:$changed"
else
    echo "apt mirrors already on HTTPS generic mirrors"
fi

sudo apt-get update
