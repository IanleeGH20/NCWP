#!/usr/bin/env python3
"""Missing STS-B test-only baselines: LPP, Random, ABTT (+ main-formula Soft-W).

`table_stsb_testonly_1379.csv` only carries Base / PCA / Soft / ZCA / NCWP, so the
paper's stated protocol (fit = 2,910 validation sentences, eval = 1,379 test pairs)
has no LPP, Random-projection or ABTT row. This fills them in.

Implementations are taken from the MAIN runners, not the reproduction helpers, so the
rows are comparable to the published tables:
  LPP     run_beir_experiment.fit_lpp_projector  (n_neighbors=10, binary symmetric
          affinity, ridge 1e-3, smallest-eigenvalue eigenvectors)
  Random  run_beir_experiment.fit_random_projector (QR of a Gaussian, seed 42)
  ABTT    ncwp.common fit_abtt_components / fit_abtt_pca_matrix (m=5) — ABTT is
          a reproduction-only baseline, so there is no main-runner counterpart
  Soft-W  run_sts_experiment.py L414 form: eigenvalue *shrinkage* alpha=0.1.
          Reported as `Soft_whitening_main` alongside the existing
          `Soft_whitening` row, which used the exponent form (alpha=0.25) and
          therefore does not match the published Soft-White numbers (see the
          round-2 doc, section 2-e).

All methods here are closed-form / single-valued; only Random carries a seed.
Cached embeddings, no model loading. Output: results/table_baselines_testonly1379.
"""
import argparse

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    fit_abtt_components,
    fit_abtt_pca_matrix,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    save_results,
    spearman_sts,
    transform_abtt_full,
    transform_abtt_pca,
)

ABTT_M = 5
LPP_NEIGHBORS = 10
LPP_REG = 1e-3
SOFT_ALPHA_MAIN = 0.1
RANDOM_SEED = 42


def fit_lpp_projector(X, k, n_neighbors=LPP_NEIGHBORS, reg=LPP_REG):
    """Verbatim from run_beir_experiment.py (the main-runner LPP)."""
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=min(n_neighbors, N - 1) - 1, axis=1)[:, :n_neighbors]
    Wg = np.zeros((N, N), dtype=np.float32)
    Wg[np.repeat(np.arange(N), idx.shape[1]), idx.reshape(-1)] = 1.0
    Wg = np.maximum(Wg, Wg.T)
    d = Wg.sum(1)
    Dg = np.diag(d)
    Lg = Dg - Wg
    XtDX = Xc.T @ Dg @ Xc + reg * np.eye(D, dtype=np.float32)
    XtLX = Xc.T @ Lg @ Xc
    try:
        A = np.linalg.solve(XtDX, XtLX)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(XtDX) @ XtLX
    w, V = np.linalg.eigh(A)
    return (V[:, np.argsort(w)][:, :min(k, D)].astype(np.float32),
            mu.squeeze(0).astype(np.float32))


def fit_random_projector(dim_in, k, seed=RANDOM_SEED):
    """Verbatim from run_beir_experiment.py."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(dim_in, k)).astype(np.float32))
    return Q[:, :k].astype(np.float32)


def soft_white_main(X_fit, k, alpha=SOFT_ALPHA_MAIN, shrink=0.08):
    """run_sts_experiment.py L412-417: eigenvalue shrinkage toward the mean, not an
    exponent. Returns (mu, W) with the 1/sqrt scaling folded into W."""
    Xd = X_fit.astype(np.float64)
    n, d = Xd.shape
    mu = Xd.mean(0)
    Xc = Xd - mu
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    cov_s = (1 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    ev, evec = np.linalg.eigh(cov_s)
    ev = ev[::-1]
    evec = evec[:, ::-1]
    kk = min(k, d)
    s_k = ev[:kk]
    denom = np.sqrt((1.0 - alpha) * s_k + alpha * s_k.mean() + 1e-8)
    return mu.astype(np.float32), (evec[:, :kk] / denom).astype(np.float32)


def l2n_map(keys, Z):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return {keys[i]: Zn[i] for i in range(len(keys))}


def run_model(model, dims):
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    sts = load_sts_benchmark(eval_split="test")
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval)
    Xe = np.stack([X_eval[s] for s in keys]).astype(np.float32)
    pairs, scores = sts["eval_pairs"], sts["eval_scores"]
    rows = []

    def add(method, r, val, seed=None, notes=""):
        rows.append(metadata_row(
            dataset="stsb", split="test", model_nickname=model,
            hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool", method=method,
            r=r, seed=seed, fit_size=sts["fit_size"], eval_size=sts["eval_size"],
            metric_name="spearman_x100", metric_value=round(float(val), 4), notes=notes))

    def score(Z):
        return spearman_sts(l2n_map(keys, Z), pairs, scores)

    add("Base_mean_pool", d_full, score(Xe), notes="full dim; sanity check vs existing table")
    print(f"[{model}] Base D={d_full} -> {rows[-1]['metric_value']:.2f}", flush=True)

    mu_a, comps = fit_abtt_components(X_fit, ABTT_M)
    add("ABTT_full", d_full, score(transform_abtt_full(Xe, mu_a, comps)), notes=f"m={ABTT_M}")
    print(f"[{model}] ABTT_full -> {rows[-1]['metric_value']:.2f}", flush=True)

    for r in dims:
        w_lpp, mu_lpp = fit_lpp_projector(X_fit, r)
        add("LPP", r, score((Xe - mu_lpp) @ w_lpp),
            notes=f"n_neighbors={LPP_NEIGHBORS} binary affinity ridge={LPP_REG}")

        w_rand = fit_random_projector(d_full, r)
        mu_rand = X_fit.mean(0).astype(np.float32)
        add("Random", r, score((Xe - mu_rand) @ w_rand), seed=RANDOM_SEED, notes="QR of Gaussian")

        mu_ap, w_ap = fit_abtt_pca_matrix(X_fit, mu_a, comps, r)
        add("ABTT_PCA_r", r, score(transform_abtt_pca(Xe, mu_a, comps, mu_ap, w_ap)),
            notes=f"m={ABTT_M}")

        mu_s, w_s = soft_white_main(X_fit, r)
        add("Soft_whitening_main", r, score((Xe - mu_s) @ w_s),
            notes=f"eigenvalue shrinkage alpha={SOFT_ALPHA_MAIN} (main-runner formula)")

        print(f"[{model}] r={r:>5} LPP={rows[-4]['metric_value']:>6.2f} "
              f"Random={rows[-3]['metric_value']:>6.2f} "
              f"ABTT_PCA={rows[-2]['metric_value']:>6.2f} "
              f"Soft_main={rows[-1]['metric_value']:>6.2f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                   choices=list(MODEL_CONFIGS))
    p.add_argument("--dims", type=int, nargs="+", default=None,
                   help="default: per-model cfg['dims'] (matches table_stsb_testonly_1379)")
    p.add_argument("--basename", default="table_baselines_testonly1379")
    args = p.parse_args()

    all_rows = []
    for model in args.models:
        dims = args.dims if args.dims is not None else MODEL_CONFIGS[model]["dims"]
        all_rows.extend(run_model(model, dims))
        save_results(all_rows, args.basename)
    csv_path, jsonl_path = save_results(all_rows, args.basename)
    print(f"\nSaved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
