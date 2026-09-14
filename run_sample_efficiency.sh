#!/bin/bash
# run_sample_efficiency.sh
#
# NCWP Sample Efficiency 실험 (Quora only):
#   fit 데이터 수(N)를 3000 → 5000 → 10000 으로 늘리면서
#   검색 성능(NDCG@10)이 어떻게 변하는지 측정한다.
#
# 방법 D (query_sample):
#   test queries + dev queries 풀에서 N개를 랜덤 샘플해서 NCWP fit
#   → 전체 corpus(Quora: 52만)에 대해 검색 평가
#
# 결과 저장:
#   quora_results/fit_query_sample/llama-8b_N3000_results_main.csv  등

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/quora_results/logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local MODEL=$1
    local N=$2
    echo ""
    echo "========================================"
    echo "  START: quora / $MODEL / query_sample N=$N"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    python run_beir_experiment.py \
        --dataset quora \
        --model   "$MODEL" \
        --fit_mode query_sample \
        --fit_sample "$N" \
        2>&1 | tee "$LOG_DIR/${MODEL}_qsample_N${N}.log"
    echo "  DONE: quora / $MODEL / N=$N  ($(date '+%H:%M:%S'))"
}

# ── Quora: llama-8b ─────────────────────────────────────────
echo "=== [Quora] llama-8b: N=3000,5000,10000 ==="
for N in 3000 5000 10000; do
    run_exp llama-8b $N
done

# ── Quora: qwen-4b ──────────────────────────────────────────
echo "=== [Quora] qwen-4b: N=3000,5000,10000 ==="
for N in 3000 5000 10000; do
    run_exp qwen-4b $N
done

echo ""
echo "========================================"
echo "  모든 Sample Efficiency 실험 완료!"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  그래프 생성 중..."
echo "========================================"
python plot_sample_efficiency.py 2>&1 | tee "$LOG_DIR/plot_efficiency.log"
echo "  완료."
