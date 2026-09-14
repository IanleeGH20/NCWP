#!/bin/bash
# run_all_sensitivity.sh
# Quora Hyperparameter Sensitivity Analysis
# dim=[32,64] (2**5, 2**6), OAT sweep
# qwen-4b: 캐시 있음 → 즉시 시작
# llama-8b: 캐시 없음 → 인코딩 후 실행

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token}"
cd /workspace/NCWP

LOG_DIR="/workspace/NCWP/quora_results/sensitivity"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  Sensitivity Analysis 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# qwen-4b (캐시 있음, 빠름)
echo ""
echo "--- [1/2] qwen-4b ---"
python run_sensitivity.py --model qwen-4b \
    2>&1 | tee "$LOG_DIR/qwen-4b_sensitivity.log"

# llama-8b (캐시 없으면 인코딩 필요)
echo ""
echo "--- [2/2] llama-8b ---"
python run_sensitivity.py --model llama-8b \
    2>&1 | tee "$LOG_DIR/llama-8b_sensitivity.log"

echo ""
echo "========================================"
echo "  Sensitivity 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "  결과: /workspace/NCWP/quora_results/sensitivity/"
echo "========================================"
