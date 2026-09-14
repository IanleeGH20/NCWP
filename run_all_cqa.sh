#!/bin/bash
# CQADupStack (english / gaming / physics) × (qwen-4b / llama-8b)

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/cqa_logs"
mkdir -p "$LOG_DIR"

run_exp() {
    DATASET=$1; MODEL=$2
    LOG="$LOG_DIR/${DATASET}_${MODEL}.log"
    echo "[$(date '+%H:%M:%S')] START  $DATASET  $MODEL"
    python run_beir_small.py --dataset "$DATASET" --model "$MODEL" > "$LOG" 2>&1
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   $DATASET  $MODEL"
        # 핵심 결과만 뽑아서 출력
        grep -E "NDCG@10 Summary|Method|NCWP|PCA-Whit|Base" "$LOG" | tail -8
    else
        echo "[$(date '+%H:%M:%S')] ERROR  $DATASET  $MODEL  (exit=$CODE)"
        tail -5 "$LOG"
    fi
    echo ""
}

echo "===== CQADupStack 실험 시작 (6개 조합) ====="

run_exp cqa-english  qwen-4b
run_exp cqa-english  llama-8b
run_exp cqa-gaming   qwen-4b
run_exp cqa-gaming   llama-8b
run_exp cqa-physics  qwen-4b
run_exp cqa-physics  llama-8b

echo "===== 전체 완료 ====="
