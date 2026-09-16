#!/usr/bin/env python3
"""Spec §3 (Review W1): additional symmetric retrieval — CQADupStack.

Second duplicate/paraphrase retrieval testbed beyond Quora, to show the
exceeds-Base / NCWP-vs-baseline pattern generalizes. Reuses the paper's cached
CQADupStack embeddings (cqa-{english,gaming,physics}_results/embedding_cache/) so
this is a CACHED-track job (no model loading) for qwen-4b / llama-8b. Methods:
Base_mean_pool, PCA_whitening, ZCA_shrink_full, ABTT_full, ABTT_PCA_r,
NCWP_mean_pool. NCWP via ncwp_ref. Per-subforum + average rows. Seeds 42-44.
Output: results/table_cqadupstack.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from ncwp.common import (
    MODEL_CONFIGS,
    apply_zca,
    compute_zca,
    evaluate_retrieval_ndcg10,
    fit_abtt_components,
    fit_abtt_pca_matrix,
    metadata_row,
    pca_whitening_matrix,
    project_linear,
    save_results,
    transform_abtt_full,
    transform_abtt_pca,
)
from ncwp.ncwp_ref import proj_ncwp, train_ncwp_variant

SUBFORUMS = {
    "english": "mteb/cqadupstack-english",
    "gaming": "mteb/cqadupstack-gaming",
    "physics": "mteb/cqadupstack-physics",
}
TARGET_DIMS = {
    "qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448], "llama-8b": [32, 64, 128, 256],
}
FIT_N = 1000
ABTT_M = 5


def cache_dir(sub, model):
    return f"/workspace/NCWP/cqa-{sub}_results/embedding_cache/{model}"


def load_cqa(sub, model):
    cd = cache_dir(sub, model)
    corpus = np.load(f"{cd}/corpus_base.npy").astype(np.float32)
    query = np.load(f"{cd}/query_base.npy").astype(np.float32)
    X_fit = np.load(f"{cd}/fit_corpus_N{FIT_N}.npy").astype(np.float32)
    mteb_name = SUBFORUMS[sub]
    corp_ds = load_dataset(mteb_name, "corpus", split="corpus")
    corpus_ids = [str(r["_id"]) for r in corp_ds]
    q_ds = load_dataset(mteb_name, "queries", split="queries")
    qid2text = {str(r["_id"]): r["text"] for r in q_ds}
    qrel_ds = load_dataset(mteb_name, split="test")
    qrels = defaultdict(dict)
    for r in qrel_ds:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    test_qids = sorted([q for q in qrels if q in qid2text])
    return corpus, query, X_fit, corpus_ids, test_qids, dict(qrels)


def run_subforum(model, sub, dims, seeds):
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    if not os.path.exists(f"{cache_dir(sub, model)}/corpus_base.npy"):
        print(f"[skip] no cache for {sub}/{model}")
        return []
    corpus, query, X_fit, corpus_ids, test_qids, qrels = load_cqa(sub, model)

    def ev(c, qe):
        return evaluate_retrieval_ndcg10(c, qe, corpus_ids, test_qids, qrels)

    def norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    rows = []

    def add(method, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset=f"cqa-{sub}", split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method=method, r=r, seed=str(seeds), fit_size=FIT_N,
            eval_size=len(test_qids), metric_name="ndcg@10", metric_value=round(val, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    add("Base_mean_pool", d_full, ev(norm(corpus), norm(query)), notes="full dim")
    mu_a, comps = fit_abtt_components(X_fit, ABTT_M)
    add("ABTT_full", d_full, ev(transform_abtt_full(corpus, mu_a, comps),
                                transform_abtt_full(query, mu_a, comps)), notes=f"m={ABTT_M}")
    add("ZCA_shrink_full", d_full, ev(apply_zca(corpus, *compute_zca(X_fit)),
                                      apply_zca(query, *compute_zca(X_fit))),
        notes="full-dim ZCA-only" + (" rank_deficient_n<D" if FIT_N < d_full else ""))

    for r in dims:
        mu_p, w_p = pca_whitening_matrix(X_fit, r)
        add("PCA_whitening", r, ev(project_linear(corpus, w_p, mu_p), project_linear(query, w_p, mu_p)))
        mu_ap, w_ap = fit_abtt_pca_matrix(X_fit, mu_a, comps, r)
        add("ABTT_PCA_r", r, ev(transform_abtt_pca(corpus, mu_a, comps, mu_ap, w_ap),
                                transform_abtt_pca(query, mu_a, comps, mu_ap, w_ap)), notes=f"m={ABTT_M}")
        ss = []
        for seed in seeds:
            SW, mu_in, mu_out, std_out = train_ncwp_variant(X_fit, r, variant="full", seed=seed)
            ss.append(ev(proj_ncwp(corpus, SW, mu_in, mu_out, std_out),
                         proj_ncwp(query, SW, mu_in, mu_out, std_out)))
        add("NCWP_mean_pool", r, float(np.mean(ss)), float(np.std(ss)), notes="ncwp_ref")
        print(f"[cqa-{sub}/{model}] r={r} Base={rows[0]['metric_value']:.2f} NCWP={np.mean(ss):.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b", "llama-8b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--subforums", nargs="+", default=list(SUBFORUMS), choices=list(SUBFORUMS))
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    all_rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else TARGET_DIMS[model]
        per = []
        for sub in args.subforums:
            per.extend(run_subforum(model, sub, dims, args.seeds))
        # subforum-average per (method, r)
        agg = defaultdict(list)
        for row in per:
            agg[(row["method"], row.get("r"))].append(row["metric_value"])
        cfg = MODEL_CONFIGS[model]
        for (method, r), vals in agg.items():
            if len(vals) == len(args.subforums):
                all_rows.append(metadata_row(
                    dataset="cqa-avg", split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
                    embedding_type="mean_pool", method=method, r=r, seed=str(args.seeds),
                    metric_name="ndcg@10_avg", metric_value=round(float(np.mean(vals)), 4),
                    notes=f"avg over {len(vals)} subforums"))
        all_rows.extend(per)

    csv_path, jsonl_path = save_results(all_rows, "table_cqadupstack")
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
