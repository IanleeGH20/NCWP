#!/usr/bin/env bash
# run_quora_v4.sh — v4 하이퍼파라미터로 Quora 재실행
# warmup_steps=200, cosine_tau=0.1 적용
set -euo pipefail

SCRIPT=/workspace/NCWP/run_beir_experiment.py
LOG_DIR=/workspace/NCWP/logs_v4
mkdir -p "$LOG_DIR"

run_exp() {
    local model=$1
    local dataset=$2
    local fit_on=$3
    local seed=$4
    local ncwp_only=${5:-""}
    local extra=""
    [[ -n "$ncwp_only" ]] && extra="--ncwp_only"
    local tag="${model}_${dataset}_${fit_on}_s${seed}"
    echo "[$(date '+%H:%M:%S')] START: $tag"
    python3 "$SCRIPT" \
        --model "$model" \
        --dataset "$dataset" \
        --fit_mode "$fit_on" \
        --seed "$seed" \
        --no_bm25 \
        $extra \
        >> "$LOG_DIR/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE:  $tag"
}

# E5-base / Quora / seed 42, 43, 44
run_exp e5-base quora corpus_sample 42
run_exp e5-base quora corpus_sample 43 ncwp_only
run_exp e5-base quora corpus_sample 44 ncwp_only

# BGE-base / Quora / seed 42
run_exp bge-base quora corpus_sample 42

echo "=== 모든 Quora v4 실험 완료 ==="
