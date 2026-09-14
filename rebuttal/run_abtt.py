#!/usr/bin/env python3
"""Spec §6 (Review W4): All-But-The-Top baseline vs PCA / ZCA / NCWP (STS-B + Quora).

Methods: Base_mean_pool, ABTT_full, PCA_whitening, ABTT_PCA_r, ZCA_shrink,
NCWP_mean_pool. ABTT/PCA/ZCA are closed-form (rebuttal.common); NCWP uses the
faithful paper trainer (rebuttal.ncwp_ref). Comparison of published NCWP against
closed-form anisotropy correction -> seeds 42,43,44. Cached embeddings.
Output: rebuttal_outputs/table_abtt_baseline.
"""
import argparse

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    apply_zca,
    compute_zca,
    evaluate_retrieval_ndcg10,
    fit_abtt_components,
    fit_abtt_pca_matrix,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    pca_whitening_matrix,
    project_linear,
    save_results,
    spearman_sts,
    sts_emb_map_from_matrix,
    transform_abtt_full,
    transform_abtt_pca,
)
from rebuttal.ncwp_ref import fit_project, proj_ncwp, train_ncwp_variant

REBUTTAL_DIMS = {
    "qwen-4b": [40, 80, 160, 320],
    "qwen-8b": [56, 112, 224, 448],
    "llama-8b": [32, 64, 128, 256],
}


def run_sts(model, dims, seeds, m):
    rows = []
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    sts = load_sts_benchmark()
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval)
    X_eval_mat = np.stack([X_eval[s] for s in keys]).astype(np.float32)

    def add(method, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset="stsb", split="train+test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method=method, r=r, seed=str(seeds),
            fit_size=sts["fit_size"], eval_size=sts["eval_size"], metric_name="spearman_x100",
            metric_value=round(val, 4), std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    base_map = {s: X_eval[s] / (np.linalg.norm(X_eval[s]) + 1e-12) for s in keys}
    add("Base_mean_pool", d_full, spearman_sts(base_map, sts["eval_pairs"], sts["eval_scores"]), notes="full dim")

    mu_abtt, comps = fit_abtt_components(X_fit, m)
    abtt_map = {s: transform_abtt_full(X_eval[s].reshape(1, -1), mu_abtt, comps)[0] for s in keys}
    add("ABTT_full", d_full, spearman_sts(abtt_map, sts["eval_pairs"], sts["eval_scores"]), notes=f"m={m}")

    for r in dims:
        mu_pca, w_pca = pca_whitening_matrix(X_fit, r)
        pca_map = sts_emb_map_from_matrix({s: (X_eval[s] - mu_pca) @ w_pca for s in keys}, None, None)
        add("PCA_whitening", r, spearman_sts(pca_map, sts["eval_pairs"], sts["eval_scores"]))

        mu_ap, w_ap = fit_abtt_pca_matrix(X_fit, mu_abtt, comps, r)
        ap_map = {s: transform_abtt_pca(X_eval[s].reshape(1, -1), mu_abtt, comps, mu_ap, w_ap)[0] for s in keys}
        add("ABTT_PCA_r", r, spearman_sts(ap_map, sts["eval_pairs"], sts["eval_scores"]), notes=f"m={m}")

        Xw_fit = apply_zca(X_fit, *compute_zca(X_fit))
        mu_w = Xw_fit.mean(0)
        _, _, vt = np.linalg.svd(Xw_fit - mu_w, full_matrices=False)
        w_z = vt[:r].T.astype(np.float32)
        mu_zca, s_zca = compute_zca(X_fit)
        zca_map = sts_emb_map_from_matrix(
            {s: (apply_zca(X_eval[s].reshape(1, -1), mu_zca, s_zca)[0] - mu_w) @ w_z for s in keys}, None, None)
        add("ZCA_shrink", r, spearman_sts(zca_map, sts["eval_pairs"], sts["eval_scores"]))

        ss = []
        for seed in seeds:
            Z, _ = fit_project(X_fit, X_eval_mat, r, seed=seed, variant="full")
            ncwp_map = {keys[i]: Z[i] for i in range(len(keys))}
            ss.append(spearman_sts(ncwp_map, sts["eval_pairs"], sts["eval_scores"]))
        add("NCWP_mean_pool", r, float(np.mean(ss)), float(np.std(ss)), notes="ncwp_ref")
        print(f"[STS/{model}] r={r} NCWP={np.mean(ss):.2f}±{np.std(ss):.2f}")
    return rows


def run_quora(model, dims, seeds, m):
    rows = []
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    q = load_quora_cached(model)
    corpus, queries, X_fit = q["corpus_np"], q["query_np"], q["X_fit"]

    def ev(c, qe):
        return evaluate_retrieval_ndcg10(c, qe, q["corpus_ids"], q["test_qids"], q["qrels"])

    def add(method, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset="quora", split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method=method, r=r, seed=str(seeds),
            fit_size=q["fit_size"], eval_size=q["eval_size"], metric_name="ndcg@10",
            metric_value=round(val, 4), std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    add("Base_mean_pool", d_full, ev(corpus, queries), notes="full dim")
    mu_abtt, comps = fit_abtt_components(X_fit, m)
    add("ABTT_full", d_full, ev(transform_abtt_full(corpus, mu_abtt, comps),
                                transform_abtt_full(queries, mu_abtt, comps)), notes=f"m={m}")

    for r in dims:
        mu_pca, w_pca = pca_whitening_matrix(X_fit, r)
        add("PCA_whitening", r, ev(project_linear(corpus, w_pca, mu_pca), project_linear(queries, w_pca, mu_pca)))

        mu_ap, w_ap = fit_abtt_pca_matrix(X_fit, mu_abtt, comps, r)
        add("ABTT_PCA_r", r, ev(transform_abtt_pca(corpus, mu_abtt, comps, mu_ap, w_ap),
                                transform_abtt_pca(queries, mu_abtt, comps, mu_ap, w_ap)), notes=f"m={m}")

        ss = []
        for seed in seeds:
            SW, mu_in, mu_out, std_out = train_ncwp_variant(X_fit.astype(np.float32), r, variant="full", seed=seed)
            ss.append(ev(proj_ncwp(corpus.astype(np.float32), SW, mu_in, mu_out, std_out),
                         proj_ncwp(queries.astype(np.float32), SW, mu_in, mu_out, std_out)))
        add("NCWP_mean_pool", r, float(np.mean(ss)), float(np.std(ss)), notes="ncwp_ref")
        print(f"[Quora/{model}] r={r} NCWP={np.mean(ss):.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="both")
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--abtt_m", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else REBUTTAL_DIMS[model]
        if args.dataset in ("sts", "both"):
            rows.extend(run_sts(model, dims, args.seeds, args.abtt_m))
        if args.dataset in ("quora", "both"):
            rows.extend(run_quora(model, dims, args.seeds, args.abtt_m))
    csv_path, jsonl_path = save_results(rows, "table_abtt_baseline")
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
