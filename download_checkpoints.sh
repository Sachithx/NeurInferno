#!/usr/bin/env bash
# Download seed-789 eval weights into checkpoints/seed789/.
#
# After you upload neurinferno_seed789_ckpts.tar.gz (see pack_checkpoints.sh),
# paste the public URL below, or pass it as CKPT_URL=... bash download_checkpoints.sh

set -euo pipefail
cd "$(dirname "$0")"

CKPT_URL="${CKPT_URL:-}"

if [[ -z "$CKPT_URL" ]]; then
    echo "Checkpoint archive URL is not set."
    echo "Upload neurinferno_seed789_ckpts.tar.gz (from pack_checkpoints.sh) to"
    echo "Zenodo or an anonymous GitHub Release, then either:"
    echo "  1. Put the URL in download_checkpoints.sh as CKPT_URL=..."
    echo "  2. Or run:  CKPT_URL=https://... bash download_checkpoints.sh"
    exit 1
fi

tmp="$(mktemp /tmp/neurinferno_ckpts.XXXXXX.tar.gz)"
echo "Downloading ${CKPT_URL}"
curl -L --fail "$CKPT_URL" -o "$tmp"
mkdir -p checkpoints
tar -xzf "$tmp" -C .
rm -f "$tmp"
echo "Weights are in checkpoints/seed789/"
echo "Next: CUDA_VISIBLE_DEVICES=0 bash eval.sh"
