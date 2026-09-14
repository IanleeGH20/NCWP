#!/usr/bin/env python3
"""v2 spec §2 (Review 3 W2): pooling-variant baselines + pooling+whitening + NCWP.

R3 asks whether NCWP starting from mean-pooled states is fair vs stronger frozen
recipes, specifically last-token pooling. Compare on STS-B:
  Base_mean_pool, Base_last_token_pool,
  LastToken_PCA_whitening, LastToken_ZCA_shrink, NCWP_on_last_token,
  (+ NCWP_mean_pool reference).
Shows whether NCWP is a mean-pool-only fix or a general post-hoc layer over any
frozen extraction. Encodings cached. NCWP via ncwp_ref. New -> seeds 0,1,2.
MODEL-LOAD track. Output: rebuttal_outputs_v2/table_pooling_variants.
"""
import argparse
import json
import os

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    PromptEmbedder,
    apply_zca,
    compute_zca,
    load_sts_benchmark,
    metadata_row,
    pca_whitening_matrix,
    spearman_sts,
    subset_eval_from_cache,
)
from rebuttal.ncwp_ref import fit_project

CACHE_DIR = "/workspace/NCWP/rebuttal_outputs_v2/prompt_cache"
OUT_ROOT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"
REBUTTAL_DIMS = {
    "qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448], "llama-8b": [32, 64, 128, 256],
}
POOLINGS = [("mean", "Base_mean_pool"), ("last", "Base_last_token_pool")]
EVAL_SPLIT = "train+test"


def save_v2(rows, basename):
    import pandas as pd
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT_V2, f"{basename}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(os.path.join(OUT_ROOT_V2, f"{basename}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return csv_path


def encode_cached(embedder, model, tag, split, sents):
    os.makedirs(CACHE_DIR, exist_ok=True)
    npy = os.path.join(CACHE_DIR, f"{model}__stsb__{tag}__{split}.npy")
    sj = os.path.join(CACHE_DIR, f"{model}__stsb__{tag}__{split}.sents.json")
    if os.path.exists(npy) and os.path.exists(sj):
        saved = json.load(open(sj))
        if saved == sents:
            return np.load(npy).astype(np.float32)
        if set(sents).issubset(set(saved)):
            return subset_eval_from_cache(npy, sj, sents)
    arr = embedder.encode(sents).astype(np.float32)
    np.save(npy, arr); json.dump(sents, open(sj, "w"))
    return arr


def run(model, dims, seeds):
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    sts = load_sts_benchmark(eval_split=EVAL_SPLIT)
    keys = sts["eval_sents"]
    rows = []

    def add(method, pooling, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset="stsb", split=EVAL_SPLIT, model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type=f"{pooling}_pool", method=method, pooling=pooling, r=r, seed=str(seeds),
            fit_size=sts["fit_size"], eval_size=sts["eval_size"], metric_name="spearman_x100",
            metric_value=round(val, 4), std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    embedder = PromptEmbedder(model, template_name="Plain", pool="mean")
    for pool, base_method in POOLINGS:
        embedder.pool = pool
        Xf = encode_cached(embedder, model, pool, "fit", sts["fit_sents"])
        Xe = encode_cached(embedder, model, pool, "eval", keys)
        Xe_map = {keys[i]: Xe[i] for i in range(len(keys))}

        base_map = {s: Xe_map[s] / (np.linalg.norm(Xe_map[s]) + 1e-12) for s in keys}
        add(base_method, pool, d_full, spearman_sts(base_map, sts["eval_pairs"], sts["eval_scores"]), notes="full dim")
        print(f"[{model}] {base_method} D={d_full} -> {rows[-1]['metric_value']:.2f}")

        prefix = "LastToken" if pool == "last" else "Mean"
        for r in dims:
            mu_p, w_p = pca_whitening_matrix(Xf, r)
            pca_map = {s: (v := (Xe_map[s] - mu_p) @ w_p) / (np.linalg.norm(v) + 1e-12) for s in keys}
            add(f"{prefix}_PCA_whitening", pool, r, spearman_sts(pca_map, sts["eval_pairs"], sts["eval_scores"]))

            mu_z, s_z = compute_zca(Xf)
            Xwf = apply_zca(Xf, mu_z, s_z); mu_w = Xwf.mean(0)
            _, _, vt = np.linalg.svd(Xwf - mu_w, full_matrices=False)
            zmap = {s: (v := (apply_zca(Xe_map[s].reshape(1, -1), mu_z, s_z)[0] - mu_w) @ vt[:r].T) / (np.linalg.norm(v) + 1e-12) for s in keys}
            add(f"{prefix}_ZCA_shrink", pool, r, spearman_sts(zmap, sts["eval_pairs"], sts["eval_scores"]))

            method = "NCWP_on_last_token" if pool == "last" else "NCWP_mean_pool"
            ss = []
            for seed in seeds:
                Z, _ = fit_project(Xf, Xe, r, seed=seed, variant="full")
                ss.append(spearman_sts({keys[i]: Z[i] for i in range(len(keys))}, sts["eval_pairs"], sts["eval_scores"]))
            add(method, pool, r, float(np.mean(ss)), float(np.std(ss)), notes="ncwp_ref")
            print(f"[{model}] {pool} r={r} NCWP -> {np.mean(ss):.2f}±{np.std(ss):.2f}")
    del embedder
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval_split", choices=["train+test", "test"], default="train+test")
    args = parser.parse_args()
    global EVAL_SPLIT
    EVAL_SPLIT = args.eval_split
    suffix = "" if EVAL_SPLIT == "train+test" else "_testonly1379"
    rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else REBUTTAL_DIMS[model]
        rows.extend(run(model, dims, args.seeds))
    print(f"Saved: {save_v2(rows, 'table_pooling_variants' + suffix)}")


if __name__ == "__main__":
    main()
