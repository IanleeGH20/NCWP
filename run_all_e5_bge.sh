#!/bin/bash
# run_all_e5_bge.sh  ── 실험 0-1
# E5-base / BGE-base 에 NCWP 적용 → Quora + SCIDOCS 결과 확인
#
# 모델 크기: ~110M (BERT-base)  → GPU 1장으로 충분, 빠름
# BM25 baseline 자동 포함
#
# 소요 시간 추정 (A100 1장 기준):
#   E5-base  × Quora   corpus_sample : ~30분
#   E5-base  × Quora   dev_queries   : ~25분
#   BGE-base × Quora   corpus_sample : ~30분
#   BGE-base × Quora   dev_queries   : ~25분
#   E5-base  × SCIDOCS corpus_sample : ~5분
#   BGE-base × SCIDOCS corpus_sample : ~5분
#   합계 ≈ 2시간 내외

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
echo "  E5-base + BGE-base 실험 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── Quora ──────────────────────────────────────────────────
# dev_queries (방법 C): Quora는 dev split이 있어 분포가 동일 → 강력한 baseline
run_exp e5-base  quora corpus_sample "--fit_sample 20000"
run_exp e5-base  quora dev_queries

run_exp bge-base quora corpus_sample "--fit_sample 20000"
run_exp bge-base quora dev_queries

# ── SCIDOCS ─────────────────────────────────────────────────
# SCIDOCS: dev split 없음 → corpus_sample만 지원
# BM25 기본 포함 (corpus 25K, 속도 무방)
run_exp e5-base  scidocs corpus_sample
run_exp bge-base scidocs corpus_sample

echo ""
echo "========================================"
echo "  모든 E5/BGE 실험 완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "결과 위치:"
echo "  /workspace/NCWP/quora_results/fit_corpus_sample/  (e5-base, bge-base 파일)"
echo "  /workspace/NCWP/quora_results/fit_dev_queries/"
echo "  /workspace/NCWP/scidocs_results/fit_corpus_sample/"
