#!/usr/bin/env bash
# run_qwen_scaling.sh
# qwen-4b / Quora / query_sample: N=500,1000,2000,3000,5000
# 각 N별로 최대 유효 dim = N-1 까지만 의미 있음
set -euo pipefail

SCRIPT=/workspace/NCWP/run_beir_experiment.py
LOG_DIR=/workspace/NCWP/logs_v4
mkdir -p "$LOG_DIR"

run_exp() {
    local N=$1
    echo "[$(date '+%H:%M:%S')] START: qwen-4b / Quora / N=$N"
    python3 "$SCRIPT" \
        --model qwen-4b \
        --dataset quora \
        --fit_mode query_sample \
        --fit_sample "$N" \
        --no_bm25 \
        >> "$LOG_DIR/qwen4b_quora_N${N}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE:  qwen-4b / Quora / N=$N"
}

echo "=== qwen-4b Quora Scaling 실험 시작 ==="

run_exp 500
run_exp 1000
run_exp 2000
run_exp 3000
run_exp 5000

echo "=== 모든 Scaling 실험 완료 ==="
