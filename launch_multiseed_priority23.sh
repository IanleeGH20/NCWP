#!/usr/bin/env bash
# Priority 2 + 3: Appendix + Ablation NCWP multi-seed
set -u
LOG_DIR=/workspace/NCWP/logs_v4
mkdir -p "$LOG_DIR"
cd /workspace/NCWP

run_queue() {
    local GPU=$1; shift
    for spec in "$@"; do
        IFS=':' read -r ds model dim seed variant <<< "$spec"
        variant="${variant:-qr_norefine}"
        local tag="ms_${ds}_${model}_r${dim}_s${seed}"
        [[ "$variant" == "zca_contrastive" ]] && tag="${tag}_zcaCL"
        echo "[GPU $GPU] $tag start"
        CUDA_VISIBLE_DEVICES=$GPU python3 multiseed_runner.py \
            --dataset "$ds" --model "$model" --dim "$dim" --seed "$seed" \
            --variant "$variant" \
            > "$LOG_DIR/${tag}.log" 2>&1
        echo "[GPU $GPU] $tag done"
    done
}

# ──────── STS queues ────────
# C: Qwen-4B STS r=5 × 3 seeds = 3 jobs
# D: Llama-8B STS (r=16,32,64,128,256,1024,2048) × 3 seeds = 21 jobs (r=512 제외)
# E: Qwen-8B STS r=14~1792 × 3 seeds = 24 jobs
# H1: ZCA+CL STS r=20,80,320 × 3 seeds = 9 jobs
# H2: NCWP full STS r=80 × 3 seeds = 3 jobs

# GPU 0: Qwen-4B C + H1 + H2 (15 jobs, 모두 짧음)
run_queue 0 \
    "sts:qwen-4b:5:42" "sts:qwen-4b:5:43" "sts:qwen-4b:5:44" \
    "sts:qwen-4b:20:42:zca_contrastive" "sts:qwen-4b:20:43:zca_contrastive" "sts:qwen-4b:20:44:zca_contrastive" \
    "sts:qwen-4b:80:42:zca_contrastive" "sts:qwen-4b:80:43:zca_contrastive" "sts:qwen-4b:80:44:zca_contrastive" \
    "sts:qwen-4b:320:42:zca_contrastive" "sts:qwen-4b:320:43:zca_contrastive" "sts:qwen-4b:320:44:zca_contrastive" \
    "sts:qwen-4b:80:42" "sts:qwen-4b:80:43" "sts:qwen-4b:80:44" &

# GPU 1: Llama-8B STS 절반 (r=16, 32, 64) × 3 = 9 jobs
run_queue 1 \
    "sts:llama-8b:16:42" "sts:llama-8b:16:43" "sts:llama-8b:16:44" \
    "sts:llama-8b:32:42" "sts:llama-8b:32:43" "sts:llama-8b:32:44" \
    "sts:llama-8b:64:42" "sts:llama-8b:64:43" "sts:llama-8b:64:44" &

# GPU 2: Llama-8B STS 나머지 (r=128, 256, 1024, 2048) × 3 = 12 jobs
run_queue 2 \
    "sts:llama-8b:128:42" "sts:llama-8b:128:43" "sts:llama-8b:128:44" \
    "sts:llama-8b:256:42" "sts:llama-8b:256:43" "sts:llama-8b:256:44" \
    "sts:llama-8b:1024:42" "sts:llama-8b:1024:43" "sts:llama-8b:1024:44" \
    "sts:llama-8b:2048:42" "sts:llama-8b:2048:43" "sts:llama-8b:2048:44" &

# ──────── Qwen-8B STS (24 jobs) ────────
# GPU 3: Qwen-8B STS 절반 (r=14, 28, 56, 112) × 3 = 12 jobs
run_queue 3 \
    "sts:qwen-8b:14:42" "sts:qwen-8b:14:43" "sts:qwen-8b:14:44" \
    "sts:qwen-8b:28:42" "sts:qwen-8b:28:43" "sts:qwen-8b:28:44" \
    "sts:qwen-8b:56:42" "sts:qwen-8b:56:43" "sts:qwen-8b:56:44" \
    "sts:qwen-8b:112:42" "sts:qwen-8b:112:43" "sts:qwen-8b:112:44" &

# GPU 4: Qwen-8B STS 나머지 (r=224, 448, 896, 1792) × 3 = 12 jobs
run_queue 4 \
    "sts:qwen-8b:224:42" "sts:qwen-8b:224:43" "sts:qwen-8b:224:44" \
    "sts:qwen-8b:448:42" "sts:qwen-8b:448:43" "sts:qwen-8b:448:44" \
    "sts:qwen-8b:896:42" "sts:qwen-8b:896:43" "sts:qwen-8b:896:44" \
    "sts:qwen-8b:1792:42" "sts:qwen-8b:1792:43" "sts:qwen-8b:1792:44" &

# ──────── Llama-8B Quora (18 jobs) ────────
# GPU 5: Llama-8B Quora r=64, 128, 256 × 3 = 9 jobs (각 ~7분)
run_queue 5 \
    "quora:llama-8b:64:42" "quora:llama-8b:64:43" "quora:llama-8b:64:44" \
    "quora:llama-8b:128:42" "quora:llama-8b:128:43" "quora:llama-8b:128:44" \
    "quora:llama-8b:256:42" "quora:llama-8b:256:43" "quora:llama-8b:256:44" &

# GPU 6: Llama-8B Quora r=512, 1024, 2048 × 3 = 9 jobs
run_queue 6 \
    "quora:llama-8b:512:42" "quora:llama-8b:512:43" "quora:llama-8b:512:44" \
    "quora:llama-8b:1024:42" "quora:llama-8b:1024:43" "quora:llama-8b:1024:44" \
    "quora:llama-8b:2048:42" "quora:llama-8b:2048:43" "quora:llama-8b:2048:44" &

# ──────── Qwen-8B Quora (18 jobs) — split between GPU 7 and overflow ────────
# GPU 7: Qwen-8B Quora 전체 18 jobs (가장 오래 걸림 ~120분)
run_queue 7 \
    "quora:qwen-8b:56:42" "quora:qwen-8b:56:43" "quora:qwen-8b:56:44" \
    "quora:qwen-8b:112:42" "quora:qwen-8b:112:43" "quora:qwen-8b:112:44" \
    "quora:qwen-8b:224:42" "quora:qwen-8b:224:43" "quora:qwen-8b:224:44" \
    "quora:qwen-8b:448:42" "quora:qwen-8b:448:43" "quora:qwen-8b:448:44" \
    "quora:qwen-8b:896:42" "quora:qwen-8b:896:43" "quora:qwen-8b:896:44" \
    "quora:qwen-8b:1792:42" "quora:qwen-8b:1792:43" "quora:qwen-8b:1792:44" &

wait
echo "=== ALL PRIORITY 2+3 JOBS COMPLETE ==="
