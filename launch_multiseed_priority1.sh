#!/usr/bin/env bash
# Priority 1: 본문 모든 NCWP cell × 3 seeds
# Qwen-4B STS (r=10,40,80,160,640,1280) + Quora (r=40,160,320,640,1280)
# r=20, r=320(STS), r=80(Quora)은 이미 측정됨

set -u
LOG_DIR=/workspace/NCWP/logs_v4
mkdir -p "$LOG_DIR"
cd /workspace/NCWP

# GPU별 queue 처리 함수
run_queue() {
    local GPU=$1
    shift
    for spec in "$@"; do
        IFS=':' read -r ds model dim seed <<< "$spec"
        local tag="ms_${ds}_${model}_r${dim}_s${seed}"
        echo "[GPU $GPU] $tag start"
        CUDA_VISIBLE_DEVICES=$GPU python3 multiseed_runner.py \
            --dataset "$ds" --model "$model" --dim "$dim" --seed "$seed" \
            > "$LOG_DIR/${tag}.log" 2>&1
        echo "[GPU $GPU] $tag done"
    done
}

# ───────── STS queues (3 GPU × 6 jobs) ─────────
# STS r=10, 40, 80, 160, 640, 1280 × 3 seeds = 18 jobs

# GPU 0: r=10, 40 (6 jobs)
run_queue 0 \
    "sts:qwen-4b:10:42" "sts:qwen-4b:10:43" "sts:qwen-4b:10:44" \
    "sts:qwen-4b:40:42" "sts:qwen-4b:40:43" "sts:qwen-4b:40:44" &

# GPU 1: r=80, 160 (6 jobs)
run_queue 1 \
    "sts:qwen-4b:80:42" "sts:qwen-4b:80:43" "sts:qwen-4b:80:44" \
    "sts:qwen-4b:160:42" "sts:qwen-4b:160:43" "sts:qwen-4b:160:44" &

# GPU 2: r=640, 1280 (6 jobs)
run_queue 2 \
    "sts:qwen-4b:640:42" "sts:qwen-4b:640:43" "sts:qwen-4b:640:44" \
    "sts:qwen-4b:1280:42" "sts:qwen-4b:1280:43" "sts:qwen-4b:1280:44" &

# ───────── Quora queues (5 GPU × 3 jobs) ─────────
# Quora r=40, 160, 320, 640, 1280 × 3 seeds = 15 jobs

# GPU 3: r=40 × 3 seeds
run_queue 3 \
    "quora:qwen-4b:40:42" "quora:qwen-4b:40:43" "quora:qwen-4b:40:44" &

# GPU 4: r=160 × 3 seeds
run_queue 4 \
    "quora:qwen-4b:160:42" "quora:qwen-4b:160:43" "quora:qwen-4b:160:44" &

# GPU 5: r=320 × 3 seeds
run_queue 5 \
    "quora:qwen-4b:320:42" "quora:qwen-4b:320:43" "quora:qwen-4b:320:44" &

# GPU 6: r=640 × 3 seeds
run_queue 6 \
    "quora:qwen-4b:640:42" "quora:qwen-4b:640:43" "quora:qwen-4b:640:44" &

# GPU 7: r=1280 × 3 seeds
run_queue 7 \
    "quora:qwen-4b:1280:42" "quora:qwen-4b:1280:43" "quora:qwen-4b:1280:44" &

wait
echo "=== ALL JOBS COMPLETE ==="
