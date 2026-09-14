#!/usr/bin/env bash
# Qwen-4B Quora N=2000 multi-seed (paper Table 2와 통일)
set -u
LOG_DIR=/workspace/NCWP/logs_v4
mkdir -p "$LOG_DIR"
cd /workspace/NCWP

run_queue() {
    local GPU=$1; shift
    for spec in "$@"; do
        IFS=':' read -r dim seed <<< "$spec"
        local tag="ms_quora_qwen-4b_r${dim}_s${seed}_N2000"
        echo "[GPU $GPU] $tag start"
        CUDA_VISIBLE_DEVICES=$GPU python3 multiseed_runner.py \
            --dataset quora --model qwen-4b --dim "$dim" --seed "$seed" --N 2000 \
            > "$LOG_DIR/${tag}.log" 2>&1
        echo "[GPU $GPU] $tag done"
    done
}

# 18 jobs / 6 GPU = 3 jobs each
run_queue 0 "40:42" "40:43" "40:44" &
run_queue 1 "80:42" "80:43" "80:44" &
run_queue 2 "160:42" "160:43" "160:44" &
run_queue 3 "320:42" "320:43" "320:44" &
run_queue 4 "640:42" "640:43" "640:44" &
run_queue 5 "1280:42" "1280:43" "1280:44" &
wait
echo "=== ALL DONE ==="
