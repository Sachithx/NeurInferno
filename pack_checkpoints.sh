#!/usr/bin/env bash
# Pack shipped eval weights into one archive for Zenodo / GitHub Release.
# Run from this directory. Does not upload anything.

set -euo pipefail
cd "$(dirname "$0")"

if ! find checkpoints/seed789 -name '*.ckpt' | grep -q .; then
    echo "No checkpoints under checkpoints/seed789/ — nothing to pack."
    exit 1
fi

out="neurinferno_seed789_ckpts.tar.gz"
tar -czf "$out" checkpoints/seed789
ls -lh "$out"
echo "Upload this file to Zenodo (recommended) or an anonymous GitHub Release,"
echo "then put the public URL in download_checkpoints.sh."
