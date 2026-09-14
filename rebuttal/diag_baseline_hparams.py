#!/usr/bin/env python3
"""Pin down the unspecified baseline hyperparameters Review 1 (c) asked for.

Reviewer: "Baseline hyperparameters are unspecified (the Soft-whitening exponent
alpha, LPP neighborhood size); please state them."

These live only in code, and the code disagrees with itself: the main runners
implement Soft-whitening as an *eigenvalue shrinkage* with alpha=0.1, while the
rebuttal runners implement it as an *eigenvalue exponent* with alpha=0.25, and the
paper prose describes the exponent form. This reproduces both formulas on the main
STS protocol and compares them against the published Soft-White row, so the
camera-ready can state the setting that actually produced the tables.

  main formula      z = (x-mu) V_k / sqrt((1-a)*lambda_k + a*mean(lambda_k)),  a=0.1
  rebuttal formula  z = (x-mu) V_k diag(lambda_k^-a),                          a=0.25

Published Soft-White (Qwen-4B, STS-B train+test 7,128): see PAPER_SOFT_W below.
"""
import numpy as np
import torch

from rebuttal.common import (
    MODEL_CONFIGS,
    load_sts_benchmark,
    load_sts_embeddings,
    spearman_sts,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ncwp_extended_results/qwen-4b_results_main.csv, method="Soft-White"
PAPER_SOFT_W = {1280: 56.323, 640: 54.773, 320: 52.711, 160: 49.998,
                80: 45.489, 40: 39.868, 20: 32.802, 10: 23.335, 5: 16.419}
DIMS = [20, 80, 320, 1280]


def eig_shrunk(X_fit, shrink=0.08):
    """Shared front end: shrunk covariance eigendecomposition, descending."""
    xt = torch.from_numpy(X_fit).to(DEVICE).double()
    n, d = xt.shape
    mu = xt.mean(0)
    xc = xt - mu
    cov = (xc.T @ xc) / max(n - 1, 1)
    cs = (1 - shrink) * cov + shrink * (torch.trace(cov) / d) * torch.eye(
        d, device=DEVICE, dtype=xt.dtype)
    ev, evec = torch.linalg.eigh(cs)
    return (mu.cpu().numpy().astype(np.float32),
            torch.flip(ev, [0]).cpu().numpy().astype(np.float64),
            torch.flip(evec, [1]).cpu().numpy().astype(np.float32))


def soft_main(mu, ev, evec, X_eval, k, alpha=0.1):
    """run_sts_experiment.py L414 / run_beir_experiment.py proj_soft_white."""
    Z = (X_eval - mu) @ evec[:, :k]
    s_k = ev[:k]
    denom = np.sqrt((1.0 - alpha) * s_k + alpha * s_k.mean() + 1e-8)
    return Z / denom


def soft_rebuttal(mu, ev, evec, X_eval, k, alpha=0.25):
    """rebuttal/run_sts_testonly.py + run_quora_corpus_sample.py soft_white_matrix."""
    W = evec[:, :k] @ np.diag(1.0 / np.power(np.clip(ev[:k], 1e-6, None), alpha))
    return (X_eval - mu) @ W


def l2n(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def main():
    sts = load_sts_benchmark()  # train+test 7,128 = main-table protocol
    model = "qwen-4b"
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval)
    Xe = np.stack([X_eval[s] for s in keys]).astype(np.float32)
    mu, ev, evec = eig_shrunk(X_fit)

    def score(Z):
        Zn = l2n(Z)
        return spearman_sts({keys[i]: Zn[i] for i in range(len(keys))},
                            sts["eval_pairs"], sts["eval_scores"])

    print(f"=== Soft-whitening: 어느 수식이 논문 값을 만들었나 ({model}, STS-B 7,128) ===")
    print(f"  {'r':>5} {'논문 게재':>10} | {'main식 a=0.1':>13} {'Δ':>7} | "
          f"{'rebuttal식 a=0.25':>18} {'Δ':>7} | {'지수식 a=0.1':>13} {'Δ':>7}")
    for k in DIMS:
        pub = PAPER_SOFT_W[k]
        v_main = score(soft_main(mu, ev, evec, Xe, k, alpha=0.1))
        v_reb = score(soft_rebuttal(mu, ev, evec, Xe, k, alpha=0.25))
        v_exp10 = score(soft_rebuttal(mu, ev, evec, Xe, k, alpha=0.1))
        print(f"  {k:>5} {pub:>10.3f} | {v_main:>13.3f} {v_main - pub:>+7.3f} | "
              f"{v_reb:>18.3f} {v_reb - pub:>+7.3f} | {v_exp10:>13.3f} "
              f"{v_exp10 - pub:>+7.3f}")

    print("\n=== LPP 설정 (main = run_beir_experiment.fit_lpp_projector) ===")
    print("  n_neighbors    = 10          (기본 인자; 목표 차원 k와 별개)")
    print("  affinity       = binary 0/1  (Wg[i,j]=1.0 후 Wg = max(Wg, Wg.T) 대칭화)")
    print("                               → heat-kernel/가우시안 가중 아님")
    print("  kNN 공간       = 평균중심화 후 L2 정규화한 코사인 유사도")
    print("  일반화 고유문제 = (XᵗDX + 1e-3·I)⁻¹ XᵗLX 의 최소 고유값 k개")
    print("  reg            = 1e-3")
    print("  rebuttal 구현(run_quora_corpus_sample.lpp_matrix): k=10, binary, 대칭화 → 동일")
    print("                  단 reg 항 없음(scipy.linalg.eigh(XᵗLX, XᵗDX) 직접 풀이)")

    print("\n=== ABTT m ===")
    print("  m = 5  (rebuttal/run_abtt.py --abtt_m 기본값; 결과 CSV notes에 'm=5' 기록)")
    print("  ABTT는 논문 원본 실험에 없고 rebuttal에서 신설 → main 표와 무관, 충돌 없음")
    print("  구현: 상위 m개 주성분 제거 후 재정규화(transform_abtt_full),")
    print("        ABTT_PCA_r은 제거 후 PCA top-r (fit_abtt_pca_matrix)")


if __name__ == "__main__":
    main()
