#!/usr/bin/env python3
"""BEIR-Quora main, corpus_sample fitting (fit_sample=1000).

3 backbones x {Base, PCA-whitening, Soft-whitening, ZCA-only, Random, LPP, NCWP}.
NCWP uses the faithful paper trainer (ncwp_ref, BEIR protocol k=10/tau=0), 3 seeds.
Closed-form baselines fit on the SAME 1000 corpus items (sampled with seed 42);
only the NCWP training seed varies, isolating NCWP seed variance.

Reuses cached full corpus/query embeddings (quora_results/embedding_cache), so no
LLM re-encoding is needed — the 1000-item fit set is sampled from corpus_base.npy.
Eval: full 522K corpus, all test queries. Metric: nDCG@10 (+ recall@10).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from ncwp.common import (
    MODEL_CONFIGS,
    QUORA_CACHE,
    apply_zca,
    compute_zca,
    load_quora_cached,
    metadata_row,
    pca_whitening_matrix,
    save_results,
    set_seed,
)
from ncwp.ncwp_ref import proj_ncwp, train_ncwp_variant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIT_SEED = 42


def soft_white_matrix(X_fit, dim, shrink=0.08, alpha=0.25):
    xt = torch.from_numpy(X_fit).to(DEVICE).double()
    n, d = xt.shape
    mu = xt.mean(0)
    xc = xt - mu
    cov = (xc.T @ xc) / max(n - 1, 1)
    tr = torch.trace(cov)
    cs = (1 - shrink) * cov + shrink * (tr / d) * torch.eye(d, device=DEVICE, dtype=xt.dtype)
    ev, evec = torch.linalg.eigh(cs)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(dim, d)
    w = evec[:, :k] @ torch.diag(1.0 / torch.pow(torch.clamp(ev[:k], min=1e-6), alpha))
    return mu.cpu().numpy().astype(np.float32), w.cpu().numpy().astype(np.float32)


def random_proj_matrix(d, dim, seed=FIT_SEED):
    rng = np.random.RandomState(seed)
    r = rng.randn(d, dim).astype(np.float32) / np.sqrt(d)
    q, _ = np.linalg.qr(r)
    return q[:, :dim].astype(np.float32)


def lpp_matrix(X_fit, dim, k=10):
    from scipy.linalg import eigh as scipy_eigh
    n, d = X_fit.shape
    xn = X_fit / (np.linalg.norm(X_fit, axis=1, keepdims=True) + 1e-12)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    knn = np.argsort(-sim, axis=1)[:, :k]
    w_adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in knn[i]:
            w_adj[i, j] = 1.0; w_adj[j, i] = 1.0
    d_diag = w_adj.sum(1)
    L = np.diag(d_diag) - w_adj
    xtx = X_fit.T @ (d_diag[:, None] * X_fit)
    xtlx = X_fit.T @ L @ X_fit
    try:
        _, eigvecs = scipy_eigh(xtlx, xtx)
    except Exception:
        _, eigvecs = np.linalg.eigh(np.linalg.pinv(xtx) @ xtlx)
    return eigvecs[:, :dim].astype(np.float32)


def gpu_project_linear(X_np, W_np, mu_np=None, chunk=20000):
    W = torch.from_numpy(W_np).to(DEVICE)
    mu = torch.from_numpy(mu_np).to(DEVICE) if mu_np is not None else None
    parts = []
    for i in range(0, len(X_np), chunk):
        xb = torch.from_numpy(X_np[i:i + chunk]).to(DEVICE)
        if mu is not None:
            xb = xb - mu
        z = F.normalize(xb @ W, dim=1)
        parts.append(z.half())
    return torch.cat(parts, 0)


def gpu_project_zca_pca(X_np, mu_zca, s_zca, mu_w, w_pca, chunk=20000):
    mu_z = torch.from_numpy(mu_zca).to(DEVICE) if isinstance(mu_zca, np.ndarray) else mu_zca
    parts = []
    W = torch.from_numpy(w_pca).to(DEVICE)
    muw = torch.from_numpy(mu_w).to(DEVICE)
    for i in range(0, len(X_np), chunk):
        xb = torch.from_numpy(X_np[i:i + chunk]).to(DEVICE)
        xw = F.normalize((xb - mu_z) @ s_zca, dim=1)
        z = F.normalize((xw - muw) @ W, dim=1)
        parts.append(z.half())
    return torch.cat(parts, 0)


def gpu_ndcg_recall(corpus_t, query_t, corpus_ids, test_qids, qrels, k=10, chunk=500):
    ndcg, recall = [], []
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi + chunk)
        with torch.no_grad():
            sim = (query_t[qi:qe] @ corpus_t.T).float()
            _, top_idx = torch.topk(sim, k=k, dim=1)
            top = top_idx.cpu().numpy()
        for ci, q2 in enumerate(range(qi, qe)):
            qid = test_qids[q2]
            if qid not in qrels:
                continue
            rel = qrels[qid]
            ranked = [corpus_ids[i] for i in top[ci]]
            hits = [1.0 if d in rel else 0.0 for d in ranked]
            dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), k)))
            ndcg.append(dcg / idcg if idcg > 0 else 0.0)
            recall.append(sum(hits) / min(len(rel), k) if rel else 0.0)
    return float(np.mean(ndcg) * 100), float(np.mean(recall) * 100)


def run_model(model, seeds, fit_sample=None):
    cfg = MODEL_CONFIGS[model]
    hf = cfg["hf_model_id"]
    dims = cfg["dims"]
    d_full = cfg["hidden_dim"]
    fit_n = fit_sample or cfg["quora_fit_size"]

    q = load_quora_cached(model)  # loads full corpus/query + qrels
    corpus, queries = q["corpus_np"], q["query_np"]
    corpus_ids, test_qids, qrels = q["corpus_ids"], q["test_qids"], q["qrels"]
    eval_size = q["eval_size"]

    rng = np.random.RandomState(FIT_SEED)
    fit_idx = rng.choice(len(corpus), fit_n, replace=False)
    X_fit = corpus[fit_idx].astype(np.float32)
    print(f"[{model}] corpus={corpus.shape} fit(corpus_sample N={fit_n})={X_fit.shape} queries={queries.shape}")

    rows = []

    def add(method, r, ndcg, recall, seed="42", std=None, notes=""):
        rows.append(metadata_row(
            dataset="quora", split="test", model_nickname=model, hf_model_id=hf,
            embedding_type="mean_pool", method=method, r=r, seed=seed,
            fit_mode="corpus_sample", fit_size=fit_n, eval_size=eval_size,
            metric_name="ndcg@10", metric_value=round(ndcg, 4),
            recall_at_10=round(recall, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    # Base (full dim)
    c_base = gpu_project_linear(corpus, np.eye(d_full, dtype=np.float32))
    q_base = gpu_project_linear(queries, np.eye(d_full, dtype=np.float32))
    n, rc = gpu_ndcg_recall(c_base, q_base, corpus_ids, test_qids, qrels)
    add("Base_mean_pool", d_full, n, rc, notes="full dim")
    print(f"[{model}] Base D={d_full} nDCG@10={n:.2f}")
    del c_base, q_base
    torch.cuda.empty_cache()

    # ZCA prep
    mu_zca_np, s_zca = compute_zca(X_fit)
    Xw_fit = apply_zca(X_fit, mu_zca_np, s_zca)
    mu_w = Xw_fit.mean(0).astype(np.float32)
    _, _, vt = np.linalg.svd(Xw_fit - mu_w, full_matrices=False)

    for r in dims:
        # PCA-whitening
        mu_pca, w_pca = pca_whitening_matrix(X_fit, r)
        cn = gpu_project_linear(corpus, w_pca, mu_pca)
        qn = gpu_project_linear(queries, w_pca, mu_pca)
        n, rc = gpu_ndcg_recall(cn, qn, corpus_ids, test_qids, qrels)
        add("PCA_whitening", r, n, rc)
        del cn, qn; torch.cuda.empty_cache()

        # Soft-whitening
        mu_sw, w_sw = soft_white_matrix(X_fit, r)
        cn = gpu_project_linear(corpus, w_sw, mu_sw)
        qn = gpu_project_linear(queries, w_sw, mu_sw)
        n_sw, rc_sw = gpu_ndcg_recall(cn, qn, corpus_ids, test_qids, qrels)
        add("Soft_whitening", r, n_sw, rc_sw)
        del cn, qn; torch.cuda.empty_cache()

        # ZCA-only (ZCA + PCA top-r on whitened)
        w_z = vt[:r].T.astype(np.float32)
        cn = gpu_project_zca_pca(corpus, mu_zca_np, s_zca, mu_w, w_z)
        qn = gpu_project_zca_pca(queries, mu_zca_np, s_zca, mu_w, w_z)
        n_z, rc_z = gpu_ndcg_recall(cn, qn, corpus_ids, test_qids, qrels)
        add("ZCA_only", r, n_z, rc_z)
        del cn, qn; torch.cuda.empty_cache()

        # Random projection
        w_rand = random_proj_matrix(d_full, r)
        mu_rand = X_fit.mean(0).astype(np.float32)
        cn = gpu_project_linear(corpus, w_rand, mu_rand)
        qn = gpu_project_linear(queries, w_rand, mu_rand)
        n_r, rc_r = gpu_ndcg_recall(cn, qn, corpus_ids, test_qids, qrels)
        add("Random", r, n_r, rc_r)
        del cn, qn; torch.cuda.empty_cache()

        # LPP
        w_lpp = lpp_matrix(X_fit, r)
        mu_lpp = X_fit.mean(0).astype(np.float32)
        cn = gpu_project_linear(corpus, w_lpp, mu_lpp)
        qn = gpu_project_linear(queries, w_lpp, mu_lpp)
        n_l, rc_l = gpu_ndcg_recall(cn, qn, corpus_ids, test_qids, qrels)
        add("LPP", r, n_l, rc_l)
        del cn, qn; torch.cuda.empty_cache()

        # NCWP (faithful trainer, 3 seeds, same fit set)
        ndcgs, recalls = [], []
        for seed in seeds:
            set_seed(seed)
            SW, mu_in, mu_out, std_out = train_ncwp_variant(X_fit, r, seed=seed, variant="full")
            SW_t = torch.from_numpy(SW).to(DEVICE)
            mu_in_t = torch.from_numpy(mu_in).to(DEVICE)
            mu_out_t = torch.from_numpy(mu_out).to(DEVICE)
            std_out_t = torch.from_numpy(std_out).to(DEVICE)

            def proj(X_np, chunk=20000):
                parts = []
                for i in range(0, len(X_np), chunk):
                    xb = torch.from_numpy(X_np[i:i + chunk]).to(DEVICE)
                    z = ((xb - mu_in_t) @ SW_t - mu_out_t) / std_out_t
                    parts.append(F.normalize(z, dim=1).half())
                return torch.cat(parts, 0)

            cn = proj(corpus); qn = proj(queries)
            n_c, rc_c = gpu_ndcg_recall(cn, qn, corpus_ids, test_qids, qrels)
            ndcgs.append(n_c); recalls.append(rc_c)
            del cn, qn; torch.cuda.empty_cache()
        add("NCWP", r, float(np.mean(ndcgs)), float(np.mean(recalls)),
            seed=str(list(seeds)), std=float(np.std(ndcgs)), notes="ncwp_ref BEIR protocol k=10 tau=0")
        # 6 rows were appended for this r: PCA, Soft, ZCA, Random, LPP, NCWP
        print(f"[{model}] r={r:>5} PCA={rows[-6]['metric_value']:.2f} ZCA={n_z:.2f} "
              f"NCWP={np.mean(ndcgs):.2f}±{np.std(ndcgs):.2f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--fit_sample", type=int, default=None,
                        help="Override corpus fit size (default: per-model quora_fit_size)")
    parser.add_argument("--basename", default="table_quora_corpus_sample",
                        help="Output basename (fit_size recorded per row)")
    args = parser.parse_args()

    all_rows = []
    for model in args.models:
        if args.dims is not None:
            MODEL_CONFIGS[model]["dims"] = args.dims
        fit_n = args.fit_sample or MODEL_CONFIGS[model]["quora_fit_size"]
        all_rows.extend(run_model(model, args.seeds, fit_sample=fit_n))
        save_results(all_rows, args.basename)  # checkpoint each model
    csv_path, jsonl_path = save_results(all_rows, args.basename)
    print(f"\nSaved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
