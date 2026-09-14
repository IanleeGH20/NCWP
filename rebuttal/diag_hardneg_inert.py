#!/usr/bin/env python3
"""Is the declared K=256 hard-negative selection actually active?

`ncwp_ref` / `run_ablation.py` / `run_beir_experiment.py` all declare
topk_negatives=256, but their batch is Bp = max(64, min(1024, N//64)) pairs, so a
similarity row has only 2*Bp columns. `_contrastive_loss` falls back to the plain
full-row logsumexp whenever topk >= row_width - 1, which would make K=256 a no-op.

This prints the effective row width for the real fit sizes and checks whether
variant="full" (K=256) and variant="no_hardneg" (K=None) give identical results.
If they match bit-for-bit, K=256 never selected anything.
"""
import math

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    load_sts_benchmark,
    load_sts_embeddings,
    spearman_sts,
)
from rebuttal.ncwp_ref import fit_project


def infer_batch_pairs(N):
    """Bp as computed in ncwp_ref.train_ncwp_variant / run_beir_experiment."""
    return max(64, min(1024, N // 64))


def main():
    print("=== 유효 negative row 폭 (ncwp_ref / run_ablation / run_beir 계열) ===")
    print(f"  {'프로토콜':28s} {'N(fit)':>7} {'Bp':>5} {'row 폭(2*Bp)':>12} "
          f"{'K=256 발동?':>12}")
    cases = [("STS-B fit (valid)", 2910), ("Quora fit N=2000 (qwen-4b)", 2000),
             ("Quora fit N=1000 (8B급)", 1000)]
    for name, N in cases:
        Bp = infer_batch_pairs(N)
        width = 2 * Bp
        active = "예" if 256 < width - 1 else "아니오 (무효)"
        print(f"  {name:28s} {N:>7} {Bp:>5} {width:>12} {active:>12}")

    print("\n  참고: ncwp_ref는 batch_pairs를 max(64, min(1024, N//64))로 계산하므로")
    print("        위 세 경우 모두 Bp=64 → row 폭 128 → topk(256) >= 127 이라서")
    print("        `denom = logsumexp(전체 row)` 분기로 빠진다 = hard-negative 선택 없음.")

    print("\n=== 실증: variant='full'(K=256) vs 'no_hardneg'(K=None) ===")
    sts = load_sts_benchmark()
    keys_cache = {}
    for model in ["qwen-4b"]:
        X_fit, X_eval, _ = load_sts_embeddings(model, sts)
        keys = list(X_eval)
        keys_cache[model] = keys
        X_eval_mat = np.stack([X_eval[s] for s in keys]).astype(np.float32)
        for r in [80]:
            out = {}
            for variant in ["full", "no_hardneg"]:
                Z, _ = fit_project(X_fit, X_eval_mat, r, seed=42, variant=variant)
                out[variant] = (Z, spearman_sts({keys[i]: Z[i] for i in range(len(keys))},
                                                sts["eval_pairs"], sts["eval_scores"]))
            zf, sf = out["full"]
            zn, sn = out["no_hardneg"]
            same = np.array_equal(zf, zn)
            maxdiff = float(np.abs(zf - zn).max())
            print(f"  {model} r={r}: full={sf:.4f}  no_hardneg={sn:.4f}  "
                  f"Δ={sn - sf:+.4f}")
            print(f"    투영 행렬 bit-identical? {'예' if same else '아니오'}  "
                  f"(max |Δ| = {maxdiff:.2e})")
            print(f"    → {'K=256은 한 번도 발동하지 않았다 (무효 파라미터)' if maxdiff < 1e-6 else 'K=256이 실제로 작동했다'}")


if __name__ == "__main__":
    main()
