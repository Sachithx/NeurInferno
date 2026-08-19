#!/usr/bin/env bash
# Download seed-789 eval weights from the shared Drive folder into
# checkpoints/seed789/.

set -euo pipefail
cd "$(dirname "$0")"

DRIVE_FOLDER="${DRIVE_FOLDER:-https://drive.google.com/drive/folders/1AR5maHr_DzH8yXj1jcKAJ-8z8lYk38nD}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "Installing gdown ..."
    pip install -q gdown
fi

tmp="$(mktemp -d /tmp/neurinferno_ckpts.XXXXXX)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

echo "Downloading ${DRIVE_FOLDER}"
gdown --folder "$DRIVE_FOLDER" -O "$tmp" --remaining-ok

# Accept any of:
#   .../checkpoints/seed789/{lopo,l4po}
#   .../seed789/{lopo,l4po}
#   .../{lopo,l4po}
src=""
if [[ -d "$tmp/checkpoints/seed789/lopo" ]]; then
    src="$tmp/checkpoints/seed789"
elif [[ -d "$tmp/seed789/lopo" ]]; then
    src="$tmp/seed789"
else
    src="$(find "$tmp" -type d -name lopo -print -quit 2>/dev/null || true)"
    if [[ -n "$src" ]]; then
        src="$(dirname "$src")"
    fi
fi

if [[ -z "$src" || ! -d "$src/lopo" ]]; then
    echo "Download succeeded but the folder layout was unexpected."
    echo "Expected lopo/ and l4po/ under checkpoints/seed789/."
    find "$tmp" -maxdepth 4 -type d | head -40
    exit 1
fi

mkdir -p checkpoints
rm -rf checkpoints/seed789
mkdir -p checkpoints/seed789
cp -a "$src/." checkpoints/seed789/
echo "Weights are in checkpoints/seed789/"
echo "Next: CUDA_VISIBLE_DEVICES=0 bash eval.sh"
