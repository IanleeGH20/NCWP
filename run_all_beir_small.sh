#!/bin/bash
# FiQA + SCIDOCS 경량 실험 (qwen-4b → llama-8b 순)

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/beir_small_logs"
mkdir -p "$LOG_DIR"

run_exp() {
    DATASET=$1; MODEL=$2
    LOG="$LOG_DIR/${DATASET}_${MODEL}.log"
    echo "[$(date '+%H:%M:%S')] START  $DATASET  $MODEL"
    python run_beir_small.py --dataset "$DATASET" --model "$MODEL" > "$LOG" 2>&1
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   $DATASET  $MODEL"
    else
        echo "[$(date '+%H:%M:%S')] ERROR  $DATASET  $MODEL  (exit=$CODE)"
        tail -5 "$LOG"
    fi
}

echo "===== FiQA + SCIDOCS 경량 실험 시작 ====="

run_exp fiqa    qwen-4b
run_exp fiqa    llama-8b
run_exp scidocs qwen-4b
run_exp scidocs llama-8b

echo "===== 전체 완료 ====="
