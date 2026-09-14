#!/bin/bash
# run_multiseed.sh  — 0-2: Multi-seed 실험
# Quora / E5-base / corpus_sample, seeds = 42(기존) / 43 / 44
# --ncwp_only: Base, PCA-White 등 deterministic 방법 스킵 (캐시 재사용)
# 소요: 약 30분 (seed당 15분, 2 seeds 추가)

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/quora_results/logs/multiseed"
mkdir -p "$LOG_DIR"

for SEED in 43 44; do
    echo ""
    echo "========================================"
    echo "  Multi-seed: e5-base / quora / corpus_sample / seed=$SEED"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    python run_beir_experiment.py \
        --dataset quora \
        --model e5-base \
        --fit_mode corpus_sample \
        --fit_sample 20000 \
        --seed "$SEED" \
        --ncwp_only \
        --no_bm25 \
        2>&1 | tee "$LOG_DIR/e5-base_corpus_sample_s${SEED}.log"
    echo "  DONE: seed=$SEED  ($(date '+%H:%M:%S'))"
done

echo ""
echo "========================================"
echo "  Multi-seed 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
