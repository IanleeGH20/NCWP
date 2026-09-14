#!/bin/bash
# run_all_nfcorpus_fiqa.sh  ── 실험 0-3
# BEIR subset 2개 추가: NFCorpus + FiQA
# 모델: llama-8b / qwen-4b  (기존과 동일)
# 추가로 e5-base / bge-base 도 함께 실행
#
# BM25 baseline 자동 포함
#
# 소요 시간 추정 (A100 1장 기준):
#   NFCorpus corpus 3.6K → 인코딩 매우 빠름 (< 5분/모델)
#   FiQA     corpus 57K  → 인코딩 ~10분/모델
#   llama-8b × 2 datasets × 2 modes : ~2시간
#   qwen-4b  × 2 datasets × 2 modes : ~2시간
#   e5-base  × 2 datasets × 2 modes : ~30분
#   bge-base × 2 datasets × 2 modes : ~30분
#   합계 ≈ 5시간 내외

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

run_exp() {
    local MODEL=$1
    local DATASET=$2
    local MODE=$3
    local EXTRA=${4:-""}
    echo ""
    echo "========================================"
    echo "  $DATASET / $MODEL / fit_mode=$MODE"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    local LOG_DIR="/workspace/NCWP/${DATASET}_results/logs"
    mkdir -p "$LOG_DIR"
    python run_beir_experiment.py \
        --dataset "$DATASET" \
        --model   "$MODEL" \
        --fit_mode "$MODE" \
        $EXTRA \
        2>&1 | tee "$LOG_DIR/${MODEL}_${MODE}.log"
    echo "  DONE: $MODEL / $DATASET / $MODE  ($(date '+%H:%M:%S'))"
}

echo "========================================"
echo "  NFCorpus + FiQA 실험 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ══════════════════════════════════════════════
#  NFCorpus  (corpus ~3.6K, dev split 있음)
# ══════════════════════════════════════════════
echo ""
echo "──── NFCorpus ────────────────────────────"

for MODEL in llama-8b qwen-4b; do
    run_exp "$MODEL" nfcorpus corpus_sample   # 방법 A (corpus 전체, 3.6K)
    run_exp "$MODEL" nfcorpus dev_queries     # 방법 C
done

for MODEL in e5-base bge-base; do
    run_exp "$MODEL" nfcorpus corpus_sample
    run_exp "$MODEL" nfcorpus dev_queries
done

# ══════════════════════════════════════════════
#  FiQA  (corpus ~57K, dev split 있음)
# ══════════════════════════════════════════════
echo ""
echo "──── FiQA ────────────────────────────────"

for MODEL in llama-8b qwen-4b; do
    run_exp "$MODEL" fiqa corpus_sample "--fit_sample 10000"   # 방법 A
    run_exp "$MODEL" fiqa dev_queries                          # 방법 C
done

for MODEL in e5-base bge-base; do
    run_exp "$MODEL" fiqa corpus_sample "--fit_sample 10000"
    run_exp "$MODEL" fiqa dev_queries
done

echo ""
echo "========================================"
echo "  NFCorpus + FiQA 모든 실험 완료!"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "결과 위치:"
echo "  /workspace/NCWP/nfcorpus_results/"
echo "  /workspace/NCWP/fiqa_results/"
