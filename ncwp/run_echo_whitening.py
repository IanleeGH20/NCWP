#!/usr/bin/env python3
"""Echo + whitening-only baselines on STS-B (reviewer R3 W2).

Separates the whitening-only baseline (Echo+ZCA / Echo+PCA) from the method
(NCWP-on-Echo, already in table_prompt_baseline_stsb). Reuses cached Echo
embeddings in results/prompt_cache (no re-encoding).

Reproduces cited numbers (Qwen-4B, Echo_mean, eval=train+test 7128):
  Echo+ZCA full D = 62.8,  Echo+PCA r=80 = 52.9,  NCWP-on-Echo r=80 = 60.3
"""
import argparse
import json
import os

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    apply_zca,
    compute_zca,
    load_sts_benchmark,
    metadata_row,
    pca_whitening_matrix,
    save_results,
    set_seed,
    spearman_sts,
    subset_eval_from_cache,
)
from ncwp.ncwp_ref import fit_project

CACHE_DIR = "/workspace/NCWP/results/prompt_cache"
TARGET_DIMS = {
    "qwen-4b": [40, 80, 160, 320],
    "qwen-8b": [56, 112, 224, 448],
    "llama-8b": [32, 64, 128, 256],
}


def load_echo(model, source_tag, split, wanted_sents):
    npy = os.path.join(CACHE_DIR, f"{model}__{source_tag}__{split}.npy")
    sj = os.path.join(CACHE_DIR, f"{model}__{source_tag}__{split}.sents.json")
    saved = json.load(open(sj))
    if saved == wanted_sents:
        return np.load(npy).astype(np.float32)
    return subset_eval_from_cache(npy, sj, wanted_sents)


def run_model(model, dims, seeds, eval_split, source_tag="Echo_mean"):
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    sts = load_sts_benchmark(eval_split=eval_split)
    keys = sts["eval_sents"]
    pairs, scores = sts["eval_pairs"], sts["eval_scores"]

    X_fit = load_echo(model, source_tag, "fit", sts["fit_sents"])
    X_eval = load_echo(model, source_tag, "eval", keys)

    rows = []

    def add(method, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset="stsb", split=eval_split, model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type=source_tag, method=method, r=r, seed=str(seeds) if std is not None else 0,
            fit_size=sts["fit_size"], eval_size=sts["eval_size"], metric_name="spearman_x100",
            metric_value=round(val, 4), std_if_available=(round(std, 4) if std is not None else None),
            notes=notes))

    # Echo (raw, full D)
    emap = {keys[i]: X_eval[i] / (np.linalg.norm(X_eval[i]) + 1e-12) for i in range(len(keys))}
    add("Echo", d_full, spearman_sts(emap, pairs, scores), notes="raw echo full dim")

    # Echo + ZCA (full D, whitening only)
    mu_z, s_z = compute_zca(X_fit)
    Xw_eval = apply_zca(X_eval, mu_z, s_z)
    zmap = {keys[i]: Xw_eval[i] for i in range(len(keys))}  # already L2-normalized
    add("Echo_ZCA_full", d_full, spearman_sts(zmap, pairs, scores), notes="ZCA-shrink full D")
    print(f"[{model}] Echo_ZCA_full D={d_full} -> {rows[-1]['metric_value']:.2f}")

    # Echo + PCA-whitening and NCWP-on-Echo per r
    Xw_fit = apply_zca(X_fit, mu_z, s_z)
    mu_w = Xw_fit.mean(0)
    _, _, vt = np.linalg.svd(Xw_fit - mu_w, full_matrices=False)
    for r in dims:
        mu_p, w_p = pca_whitening_matrix(X_fit, r)
        pmap = {keys[i]: (v := (X_eval[i] - mu_p) @ w_p) / (np.linalg.norm(v) + 1e-12) for i in range(len(keys))}
        add("Echo_PCA_whitening", r, spearman_sts(pmap, pairs, scores))

        w_z = vt[:r].T.astype(np.float32)
        zrmap = {keys[i]: (v := (Xw_eval[i] - mu_w) @ w_z) / (np.linalg.norm(v) + 1e-12) for i in range(len(keys))}
        add("Echo_ZCA_only", r, spearman_sts(zrmap, pairs, scores))

        ss = []
        for seed in seeds:
            set_seed(seed)
            Z, _ = fit_project(X_fit, X_eval, r, seed=seed, variant="full")
            ss.append(spearman_sts({keys[i]: Z[i] for i in range(len(keys))}, pairs, scores))
        add("NCWP_on_Echo", r, float(np.mean(ss)), float(np.std(ss)), notes="ncwp_ref")
        print(f"[{model}] r={r} Echo_PCA={rows[-3]['metric_value']:.2f} "
              f"NCWP_on_Echo={np.mean(ss):.2f}±{np.std(ss):.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval_split", choices=["train+test", "test"], default="train+test")
    parser.add_argument("--source", default="Echo_mean", choices=["Echo_mean", "Echo_last"])
    args = parser.parse_args()

    all_rows = []
    for model in args.models:
        all_rows.extend(run_model(model, TARGET_DIMS[model], args.seeds, args.eval_split, args.source))
    suffix = "" if args.eval_split == "train+test" else "_testonly1379"
    csv_path, jsonl_path = save_results(all_rows, f"table_echo_whitening{suffix}")
    print(f"\nSaved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
