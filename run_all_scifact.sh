#!/bin/bash
# run_all_scifact.sh
# llama-8b mixed, qwen-4b corpus/mixed, qwen-8b corpus/mixed 순서대로 실행
# (llama-8b corpus는 별도로 이미 실행 중)

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/scifact_results/logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local MODEL=$1
    local MODE=$2
    echo ""
    echo "========================================"
    echo "  START: $MODEL / fit_mode=$MODE"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    python run_scifact_experiment.py --model "$MODEL" --fit_mode "$MODE" \
        2>&1 | tee "$LOG_DIR/${MODEL}_${MODE}.log"
    echo ""
    echo "  DONE : $MODEL / $MODE  ($(date '+%H:%M:%S'))"
}

# llama-8b mixed
run_exp llama-8b mixed

# qwen-4b corpus + mixed
run_exp qwen-4b corpus
run_exp qwen-4b mixed

# qwen-8b corpus + mixed
run_exp qwen-8b corpus
run_exp qwen-8b mixed

echo ""
echo "========================================"
echo "  모든 실험 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "  그래프 생성 중..."
echo "========================================"
python plot_best_ncwp_scifact.py 2>&1 | tee "$LOG_DIR/plot.log"
echo "  완료."
