#!/usr/bin/env bash
# Reproduce NeurInferno LOPO and L4PO training (seed 789).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash train.sh
#
# Per held-out protocol / L4PO split (fully leakage-free):
#   Stage 1 : train ByteLM excluding the held-out protocol(s)
#   Stage 2 : train the main model on remaining data
#   Stage 3 : fine-tune, still excluding the held-out protocol(s)
#
# Existing checkpoints under checkpoints/seed789/ are reused.

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-.}"

ENC_D=128; ENC_H=4; ENC_FF=512;  ENC_L=4
LM_D=64;   LM_H=4;  LM_FF=256;   LM_L=2

TIER1="data/grammar"
TIER2="data/protocols"
SEED=789
CKPT_ROOT="checkpoints/seed${SEED}"
RESULTS_DIR="results/seed${SEED}"
N_MSG=500
MAIN_EPOCHS=1
MAX_STEPS=2000
LR=3e-4
THRESHOLD=0.75
MAX_MSGS=1000

LOPO_PROTOCOLS=(bgp_raw arp dhcp dns icmp igmp ip modbus ntp ospf tcp udp)

L4PO_SPLITS=(
  "arp bgp_raw igmp ip"
  "arp dhcp ntp ospf"
  "arp icmp igmp tcp"
  "arp ip ospf udp"
  "bgp_raw dns icmp ip"
  "bgp_raw igmp ip udp"
  "bgp_raw ip modbus tcp"
  "dhcp igmp ntp tcp"
  "dhcp igmp tcp udp"
  "dns igmp ntp ospf"
  "dns ip ntp udp"
  "icmp igmp ospf tcp"
  "ip ntp ospf tcp"
)

find_best_lm() {
    local dir="$1"
    [[ -d "$dir" ]] || { echo ""; return; }
    find "$dir" -name "*.ckpt" 2>/dev/null \
      | awk -F'val_loss=' '{if(NF>1) print $NF, $0}' \
      | sort -n | head -1 | awk '{print $2}'
}

has_ckpt() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    find "$dir" -name "*.ckpt" 2>/dev/null | grep -q .
}

mkdir -p "${CKPT_ROOT}" "${RESULTS_DIR}/l4po"

echo ""
echo "############################################################"
echo " SEED = ${SEED}"
echo "############################################################"

echo ""
echo "============================================================"
echo " LOPO  (${#LOPO_PROTOCOLS[@]} protocols)  seed=${SEED}"
echo "============================================================"

for PROTO in "${LOPO_PROTOCOLS[@]}"; do
    LOPO_FT_DIR="${CKPT_ROOT}/lopo/${PROTO}"

    if has_ckpt "${LOPO_FT_DIR}"; then
        echo "[${PROTO}] LOPO checkpoint exists — skipping"
        continue
    fi

    echo ""
    echo "========== LOPO / ${PROTO} =========="

    LM_DIR="${CKPT_ROOT}/lm/${PROTO}"
    LM_CKPT="$(find_best_lm "$LM_DIR")"

    if [[ -z "$LM_CKPT" ]]; then
        echo "  [Stage 1] Training ByteLM (excl=${PROTO}, seed=${SEED}) ..."
        python -m src.training.train_lm \
            --tier1_dir        "$TIER1" \
            --tier2_dir        "$TIER2" \
            --ckpt_dir         "$CKPT_ROOT" \
            --exclude_protocol "$PROTO" \
            --d_model          "$LM_D" \
            --n_heads          "$LM_H" \
            --d_ff             "$LM_FF" \
            --n_layers         "$LM_L" \
            --seed             "$SEED"
        LM_CKPT="$(find_best_lm "$LM_DIR")"
    else
        echo "  [Stage 1] ByteLM checkpoint exists — reusing ${LM_CKPT##*/}"
    fi

    echo "  [Stage 2+3] Training main + LOPO fine-tune (excl=${PROTO}, seed=${SEED}) ..."
    python -m src.training.train_lopo \
        --lm_ckpt          "$LM_CKPT" \
        --protocol         "$PROTO" \
        --tier1_dir        "$TIER1" \
        --tier2_dir        "$TIER2" \
        --ckpt_dir         "${CKPT_ROOT}/lopo" \
        --main_epochs      "$MAIN_EPOCHS" \
        --max_steps        "$MAX_STEPS" \
        --n_msgs           "$N_MSG" \
        --lr               "$LR" \
        --d_model          "$ENC_D" \
        --n_heads          "$ENC_H" \
        --d_ff             "$ENC_FF" \
        --n_layers         "$ENC_L" \
        --lm_d_model       "$LM_D" \
        --lm_n_heads       "$LM_H" \
        --lm_d_ff          "$LM_FF" \
        --lm_n_layers      "$LM_L" \
        --seed             "$SEED"

    echo "  [${PROTO}] done."
done

echo ""
echo "============================================================"
echo " L4PO  (${#L4PO_SPLITS[@]} splits)  seed=${SEED}"
echo "============================================================"

for SPLIT_STR in "${L4PO_SPLITS[@]}"; do
    read -ra HELD_OUT <<< "$SPLIT_STR"
    LABEL="$(printf '%s\n' "${HELD_OUT[@]}" | sort | tr '\n' '_' | sed 's/_$//')"
    L4PO_FT_DIR="${CKPT_ROOT}/l4po/${LABEL}"

    if has_ckpt "${L4PO_FT_DIR}"; then
        echo "[${LABEL}] L4PO checkpoint exists — skipping"
        continue
    fi

    echo ""
    echo "========== L4PO / ${LABEL} =========="

    LM_DIR="${CKPT_ROOT}/lm/${LABEL}"
    LM_CKPT="$(find_best_lm "$LM_DIR")"

    if [[ -z "$LM_CKPT" ]]; then
        echo "  [Stage 1] Training ByteLM (excl=${HELD_OUT[*]}, seed=${SEED}) ..."
        python -m src.training.train_lm \
            --tier1_dir         "$TIER1" \
            --tier2_dir         "$TIER2" \
            --ckpt_dir          "$CKPT_ROOT" \
            --exclude_protocols "${HELD_OUT[@]}" \
            --d_model           "$LM_D" \
            --n_heads           "$LM_H" \
            --d_ff              "$LM_FF" \
            --n_layers          "$LM_L" \
            --seed              "$SEED"
        LM_CKPT="$(find_best_lm "$LM_DIR")"
    else
        echo "  [Stage 1] ByteLM checkpoint exists — reusing ${LM_CKPT##*/}"
    fi

    echo "  [Stage 2+3] Training main + L4PO fine-tune (seed=${SEED}) ..."
    python -m src.training.train_l4po \
        --lm_ckpt          "$LM_CKPT" \
        --held_out         "${HELD_OUT[@]}" \
        --tier1_dir        "$TIER1" \
        --tier2_dir        "$TIER2" \
        --ckpt_dir         "${CKPT_ROOT}/l4po" \
        --results_dir      "${RESULTS_DIR}/l4po" \
        --main_epochs      "$MAIN_EPOCHS" \
        --max_steps        "$MAX_STEPS" \
        --n_msg            "$N_MSG" \
        --lr               "$LR" \
        --d_model          "$ENC_D" \
        --n_heads          "$ENC_H" \
        --d_ff             "$ENC_FF" \
        --n_layers         "$ENC_L" \
        --lm_d_model       "$LM_D" \
        --lm_n_heads       "$LM_H" \
        --lm_d_ff          "$LM_FF" \
        --lm_n_layers      "$LM_L" \
        --train_seed       "$SEED" \
        --skip_eval

    echo "  [${LABEL}] done."
done

echo ""
echo "========== EVALUATION  seed=${SEED} =========="
python -m src.evaluation.run_eval \
    --ckpt_root   "$CKPT_ROOT" \
    --tier2_dir   "$TIER2" \
    --results_dir "$RESULTS_DIR" \
    --threshold   "$THRESHOLD" \
    --max_msgs    "$MAX_MSGS"

if [[ -d results/reference ]]; then
    python -m src.evaluation.compare_results \
        --got "$RESULTS_DIR" \
        --ref results/reference
else
    echo "No results/reference/ — skip table comparison."
    echo "Eval CSVs are in ${RESULTS_DIR}/"
fi
