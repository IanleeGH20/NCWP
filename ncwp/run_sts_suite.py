#!/usr/bin/env python3
"""Spec §2 (all reviewers, esp. R3 W1): broader STS suite (STS12-16 + SICK-R).

Shows the exceeds-Base / dimensional-crossover findings hold beyond STSBenchmark.
Per dataset, encode its unique sentences (unsupervised, labels never used for
fitting), fit each post-hoc transform transductively, and report Spearman on its
pairs. Methods: Base_mean_pool, PCA_whitening, ZCA_shrink, ABTT_full,
NCWP_mean_pool, PromptEOL, NCWP_on_PromptEOL. NCWP via ncwp_ref. Also emits an
STS-avg row per (method, r). New datasets -> seeds 0,1,2. MODEL-LOAD track.
Output: results/table_sts_suite.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr

from ncwp.common import (
    MODEL_CONFIGS,
    PromptEmbedder,
    apply_zca,
    compute_zca,
    fit_abtt_components,
    metadata_row,
    pca_whitening_matrix,
    save_results,
    transform_abtt_full,
)
from ncwp.ncwp_ref import fit_project

from ncwp.common import PROMPT_CACHE as CACHE_DIR
DATASETS = {
    "STS12": "mteb/sts12-sts", "STS13": "mteb/sts13-sts", "STS14": "mteb/sts14-sts",
    "STS15": "mteb/sts15-sts", "STS16": "mteb/sts16-sts", "SICK-R": "mteb/sickr-sts",
}
TARGET_DIMS = {
    "qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448], "llama-8b": [32, 64, 128, 256],
}
ABTT_M = 5


def load_sts_dataset(name):
    ds = load_dataset(DATASETS[name], split="test")
    pairs = [(r["sentence1"], r["sentence2"]) for r in ds]
    scores = [float(r["score"]) for r in ds]
    sents = sorted({s for p in pairs for s in p})
    return pairs, scores, sents


def encode_cached(embedder, model, dataset, source_tag, sents):
    os.makedirs(CACHE_DIR, exist_ok=True)
    npy = os.path.join(CACHE_DIR, f"{model}__{dataset}__{source_tag}.npy")
    sj = os.path.join(CACHE_DIR, f"{model}__{dataset}__{source_tag}.sents.json")
    if os.path.exists(npy) and os.path.exists(sj) and json.load(open(sj)) == sents:
        return np.load(npy).astype(np.float32)
    arr = embedder.encode(sents).astype(np.float32)
    np.save(npy, arr); json.dump(sents, open(sj, "w"))
    return arr


def spearman(Z, sents, pairs, scores):
    idx = {s: i for i, s in enumerate(sents)}
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    pred = [float(Zn[idx[a]] @ Zn[idx[b]]) for a, b in pairs]
    return float(spearmanr(scores, pred).correlation * 100)


def eval_dataset(model, name, dims, seeds, embedder):
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    pairs, scores, sents = load_sts_dataset(name)

    embedder.template_name = "Plain"; embedder.pool = "mean"
    Xb = encode_cached(embedder, model, name, "Plain_mean", sents)
    embedder.template_name = "PromptEOL_A"; embedder.pool = "last"
    Xp = encode_cached(embedder, model, name, "PromptEOL_A_last", sents)

    rows = []

    def add(method, r, val, std=None, emb_src="mean_pool", notes=""):
        rows.append(metadata_row(
            dataset=name, split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type=emb_src, method=method, r=r, seed=str(seeds), fit_size=len(sents),
            eval_size=len(pairs), metric_name="spearman_x100", metric_value=round(val, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    add("Base_mean_pool", d_full, spearman(Xb, sents, pairs, scores))
    add("PromptEOL", d_full, spearman(Xp, sents, pairs, scores), emb_src="PromptEOL", notes="template=A pool=last")
    mu_a, comps = fit_abtt_components(Xb, ABTT_M)
    add("ABTT_full", d_full, spearman(transform_abtt_full(Xb, mu_a, comps), sents, pairs, scores), notes=f"m={ABTT_M}")
    # ZCA-only is a full-dim method (paper Table 3 semantics). Truncated ZCA is
    # unstable when the transductive fit is rank-deficient (n<D), so report full-dim.
    rank_note = "rank_deficient_n<D" if len(sents) < d_full else ""
    add("ZCA_shrink_full", d_full, spearman(apply_zca(Xb, *compute_zca(Xb)), sents, pairs, scores),
        notes=f"full-dim ZCA-only {rank_note}".strip())

    for r in dims:
        mu_p, w_p = pca_whitening_matrix(Xb, r)
        pca_val = spearman((Xb - mu_p) @ w_p, sents, pairs, scores)
        add("PCA_whitening", r, pca_val)
        ss = [spearman(fit_project(Xb, Xb, r, seed=s, variant="full")[0], sents, pairs, scores) for s in seeds]
        add("NCWP_mean_pool", r, float(np.mean(ss)), float(np.std(ss)), notes="ncwp_ref")
        ssp = [spearman(fit_project(Xp, Xp, r, seed=s, variant="full")[0], sents, pairs, scores) for s in seeds]
        add("NCWP_on_PromptEOL", r, float(np.mean(ssp)), float(np.std(ssp)), emb_src="PromptEOL", notes="ncwp_ref")
        print(f"[{name}/{model}] r={r} PCA={pca_val:.2f} NCWP={np.mean(ss):.2f} "
              f"NCWP_on_PromptEOL={np.mean(ssp):.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    all_rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else TARGET_DIMS[model]
        embedder = PromptEmbedder(model, template_name="Plain", pool="mean")
        per_model = []
        for name in args.datasets:
            per_model.extend(eval_dataset(model, name, dims, args.seeds, embedder))
        # STS-avg rows per (method, r) across datasets
        agg = defaultdict(list)
        for row in per_model:
            agg[(row["method"], row.get("r"), row["embedding_type"])].append(row["metric_value"])
        cfg = MODEL_CONFIGS[model]
        for (method, r, emb_src), vals in agg.items():
            if len(vals) == len(args.datasets):
                all_rows.append(metadata_row(
                    dataset="STS-avg", split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
                    embedding_type=emb_src, method=method, r=r, seed=str(args.seeds),
                    metric_name="spearman_x100_avg", metric_value=round(float(np.mean(vals)), 4),
                    notes=f"avg over {len(vals)} datasets"))
        all_rows.extend(per_model)
        del embedder

    csv_path, jsonl_path = save_results(all_rows, "table_sts_suite")
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
