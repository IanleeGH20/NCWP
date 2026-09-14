#!/bin/bash
# run_all_quora.sh
# Quora 실험 자동 실행 (방법 A + 방법 C, llama-8b / qwen-4b)
# run_beir_experiment.py 사용 (chunk 평가 최적화 버전)

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/quora_results/logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local MODEL=$1
    local MODE=$2
    local SAMPLE=${3:-20000}
    echo ""
    echo "========================================"
    echo "  START: quora / $MODEL / fit_mode=$MODE  (sample=$SAMPLE)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    python run_beir_experiment.py \
        --dataset quora \
        --model "$MODEL" \
        --fit_mode "$MODE" \
        --fit_sample "$SAMPLE" \
        2>&1 | tee "$LOG_DIR/${MODEL}_${MODE}.log"
    echo "  DONE: $MODEL / $MODE  ($(date '+%H:%M:%S'))"
}

# llama-8b: 방법 A (corpus 20K) + 방법 C (dev queries)
run_exp llama-8b corpus_sample 20000
run_exp llama-8b dev_queries

# qwen-4b: 방법 A + 방법 C
run_exp qwen-4b corpus_sample 20000
run_exp qwen-4b dev_queries

echo ""
echo "========================================"
echo "  방법 A/C 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "  다음: Sample Efficiency 실험 (방법 D) 시작..."
echo "========================================"

# ── 방법 D: Sample Efficiency (N=3000,5000,10000) ──
run_qsample() {
    local MODEL=$1
    local N=$2
    echo ""
    echo "  [방법D] quora / $MODEL / query_sample N=$N  ($(date '+%H:%M:%S'))"
    python run_beir_experiment.py \
        --dataset quora \
        --model "$MODEL" \
        --fit_mode query_sample \
        --fit_sample "$N" \
        2>&1 | tee "$LOG_DIR/${MODEL}_qsample_N${N}.log"
    echo "  DONE: $MODEL / N=$N"
}

for N in 3000 5000 10000; do
    run_qsample llama-8b $N
done

for N in 3000 5000 10000; do
    run_qsample qwen-4b $N
done

echo ""
echo "========================================"
echo "  모든 Quora 실험 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "  그래프 생성 중 (방법A/C + 방법D)..."
echo "========================================"
python plot_sample_efficiency.py 2>&1 | tee "$LOG_DIR/plot_efficiency.log"
echo "  완료."
