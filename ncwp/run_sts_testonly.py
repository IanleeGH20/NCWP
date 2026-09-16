#!/usr/bin/env python3
"""STSBenchmark test-only (1,379 pairs) re-evaluation for all backbones.

Reproduces the paper's stated protocol (fit=valid 2,910 unique sentences,
eval=test 1,379 pairs) for Base / PCA-whitening / Soft-White / ZCA-only / NCWP
across the full per-backbone dim sweep, 3 seeds for NCWP.

Cached embeddings (sts_ablation_cache/{model}_main_embs.npy) are reused, so no
LLM re-encoding is needed. Test sentences are a subset of the cached fit+eval
sentence set.
"""
import argparse

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    NCWP_HPARAMS,
    apply_zca,
    compute_zca,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    pca_whitening_matrix,
    save_results,
    set_seed,
    spearman_sts,
    sts_emb_map_from_matrix,
    train_ncwp,
)


def soft_white_matrix(X_fit, dim, shrink=0.08, alpha=0.25):
    import torch

    xt = torch.from_numpy(X_fit).to("cuda").double()
    n, d = xt.shape
    mu = xt.mean(0)
    xc = xt - mu
    cov = (xc.T @ xc) / max(n - 1, 1)
    tr = torch.trace(cov)
    cs = (1 - shrink) * cov + shrink * (tr / d) * torch.eye(d, device="cuda", dtype=xt.dtype)
    ev, evec = torch.linalg.eigh(cs)
    ev = torch.flip(ev, [0])
    evec = torch.flip(evec, [1])
    k = min(dim, d)
    w = evec[:, :k] @ torch.diag(1.0 / torch.pow(torch.clamp(ev[:k], min=1e-6), alpha))
    return mu.cpu().numpy().astype(np.float32), w.cpu().numpy().astype(np.float32)


def run_model(model, seeds):
    cfg = MODEL_CONFIGS[model]
    hf = cfg["hf_model_id"]
    dims = cfg["dims"]
    d_full = cfg["hidden_dim"]

    sts = load_sts_benchmark(eval_split="test")
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    pairs, scores = sts["eval_pairs"], sts["eval_scores"]
    fit_size, eval_size = sts["fit_size"], sts["eval_size"]

    def add(rows, method, r, val, std=None, notes=""):
        rows.append(
            metadata_row(
                dataset="stsb", split="test", model_nickname=model, hf_model_id=hf,
                embedding_type="mean_pool", method=method, r=r, seed="42-44" if std is not None else 42,
                fit_size=fit_size, eval_size=eval_size, metric_name="spearman_x100",
                metric_value=round(float(val), 4),
                std_if_available=(round(float(std), 4) if std is not None else None),
                notes=notes,
            )
        )

    rows = []

    # Base (full dim)
    base_map = {s: x / (np.linalg.norm(x) + 1e-12) for s, x in X_eval.items()}
    base = spearman_sts(base_map, pairs, scores)
    add(rows, "Base_mean_pool", d_full, base, notes="full dim")
    print(f"[{model}] Base(D={d_full}) = {base:.2f}  (fit={fit_size}, eval={eval_size} pairs)")

    # Whitening prep
    mu_zca, s_zca = compute_zca(X_fit, shrink=NCWP_HPARAMS["shrink"])
    Xw_fit = apply_zca(X_fit, mu_zca, s_zca)
    Xw_eval = {s: apply_zca(X_eval[s].reshape(1, -1), mu_zca, s_zca)[0] for s in X_eval}
    mu_w = Xw_fit.mean(0)
    xwc = Xw_fit - mu_w
    _, _, vt = np.linalg.svd(xwc, full_matrices=False)

    for r in dims:
        # PCA-whitening
        mu_pca, w_pca = pca_whitening_matrix(X_fit, r, shrink=NCWP_HPARAMS["shrink"])
        pca_map = sts_emb_map_from_matrix({s: (X_eval[s] - mu_pca) @ w_pca for s in X_eval})
        add(rows, "PCA_whitening", r, spearman_sts(pca_map, pairs, scores))

        # Soft-White
        mu_sw, w_sw = soft_white_matrix(X_fit, r, shrink=NCWP_HPARAMS["shrink"])
        sw_map = sts_emb_map_from_matrix({s: (X_eval[s] - mu_sw) @ w_sw for s in X_eval})
        add(rows, "Soft_whitening", r, spearman_sts(sw_map, pairs, scores))

        # ZCA-only (ZCA whiten + PCA top-r)
        w_z = vt[:r].T.astype(np.float32)
        zca_map = sts_emb_map_from_matrix({s: (Xw_eval[s] - mu_w) @ w_z for s in X_eval})
        add(rows, "ZCA_only", r, spearman_sts(zca_map, pairs, scores))

        # NCWP (final protocol, 3 seeds)
        seed_scores = []
        for seed in seeds:
            set_seed(seed)
            w_ncwp = train_ncwp(Xw_fit, r, seed, positive_mode="knn", **NCWP_HPARAMS)
            mu_out = Xw_fit.mean(0) @ w_ncwp
            ncwp_map = sts_emb_map_from_matrix({s: Xw_eval[s] @ w_ncwp - mu_out for s in X_eval})
            seed_scores.append(spearman_sts(ncwp_map, pairs, scores))
        add(rows, "NCWP", r, float(np.mean(seed_scores)), std=float(np.std(seed_scores)),
            notes=f"seeds={list(seeds)}")
        print(
            f"[{model}] r={r:>5}  PCA={rows[-4]['metric_value']:.2f}  "
            f"ZCA={rows[-2]['metric_value']:.2f}  NCWP={np.mean(seed_scores):.2f}±{np.std(seed_scores):.2f}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    all_rows = []
    for model in args.models:
        all_rows.extend(run_model(model, args.seeds))

    csv_path, jsonl_path = save_results(all_rows, "table_stsb_testonly_1379")
    print(f"\nSaved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
