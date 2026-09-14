#!/bin/bash
# run_all_sts.sh  — Tier 1-1: MRPC + SICK STS retrieval 실험
# e5-base / bge-base 우선 (빠름), 이후 llama-8b / qwen-4b
# 소요: 약 1-2시간 (e5/bge는 캐시 없어 인코딩 필요)

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/sts_logs"
mkdir -p "$LOG_DIR"

run_sts() {
    local DATASET=$1
    local MODEL=$2
    echo ""
    echo "========================================"
    echo "  STS: $DATASET / $MODEL"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    python run_sts_experiment.py --dataset "$DATASET" --model "$MODEL" \
        2>&1 | tee "$LOG_DIR/${DATASET}_${MODEL}.log"
    echo "  DONE: $DATASET / $MODEL  ($(date '+%H:%M:%S'))"
}

# Encoder 모델 먼저 (빠름)
run_sts mrpc e5-base
run_sts mrpc bge-base
run_sts sick e5-base
run_sts sick bge-base

# LLM 모델 (느림, 선택적)
# run_sts mrpc llama-8b
# run_sts mrpc qwen-4b
# run_sts sick llama-8b
# run_sts sick qwen-4b

echo ""
echo "========================================"
echo "  STS 실험 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
