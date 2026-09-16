#!/usr/bin/env python3
"""v2 spec §7 (Review 2 W1): regularizer ablation for Eq (6)/(7).

Are BOTH the covariance and orthogonality regularizers needed? Train with the
FAITHFUL paper trainer (ncwp.ncwp_ref, == run_ablation.py), varying ONLY the
regularizer weights (and optionally QR / hard-neg). Report task score + training
diagnostics so the reviewer can see each term's role.

Re-examination of the published NCWP method -> seeds 42,43,44 (matches the
paper's ablation seed). Cached embeddings, no model loading. Outputs to
results/.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ncwp.common import (
    MODEL_CONFIGS,
    evaluate_retrieval_ndcg10,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    spearman_sts,
)
from ncwp.ncwp_ref import proj_ncwp, train_ncwp_variant

OUT_ROOT_V2 = "/workspace/NCWP/results"

# minimal set answers Review 2's Eq(6)/(7) question; --full adds QR isolation
MIN_VARIANTS = ["full", "no_cov", "no_orth", "no_both"]
FULL_VARIANTS = ["full", "no_cov", "no_orth", "no_both", "no_qr", "no_hardneg", "no_refinement"]


def save_v2(rows, basename):
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT_V2, f"{basename}.csv")
    jsonl_path = os.path.join(OUT_ROOT_V2, f"{basename}.jsonl")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def geom_diagnostics(Z):
    """Z: (n, r) L2-normalized rows -> (anisotropy_mean, offdiag_cov, avg_feature_std)."""
    n, r = Z.shape
    mu = Z.mean(0)
    mu_dir = mu / (np.linalg.norm(mu) + 1e-12)
    aniso = float(np.mean(Z @ mu_dir))
    Zc = Z - mu
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    offdiag = float(np.mean(np.abs(cov - np.diag(np.diag(cov)))))
    fstd = float(Z.std(0).mean())
    return aniso, offdiag, fstd


def run_sts(model, dims, seeds, variants):
    rows = []
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark()
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    eval_keys = list(X_eval)
    X_eval_mat = np.stack([X_eval[s] for s in eval_keys]).astype(np.float32)

    for variant in variants:
        for r in dims:
            scores, conds, offs, stds, anisos = [], [], [], [], []
            for seed in seeds:
                SW, mu_in, mu_out, std_out = train_ncwp_variant(
                    X_fit.astype(np.float32), r, variant=variant, seed=seed)
                Z = proj_ncwp(X_eval_mat, SW, mu_in, mu_out, std_out)
                emb_map = {eval_keys[i]: Z[i] for i in range(len(eval_keys))}
                scores.append(spearman_sts(emb_map, sts["eval_pairs"], sts["eval_scores"]))
                aniso, off, fstd = geom_diagnostics(Z)
                conds.append(float(np.linalg.cond(SW)))
                offs.append(off); stds.append(fstd); anisos.append(aniso)
            rows.append(metadata_row(
                dataset="stsb", split="train+test", model_nickname=model,
                hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool", method=f"NCWP_{variant}",
                r=r, seed=str(seeds), fit_size=sts["fit_size"], eval_size=sts["eval_size"],
                metric_name="spearman_x100", metric_value=round(float(np.mean(scores)), 4),
                std_if_available=round(float(np.std(scores)), 4),
                cond_number=round(float(np.mean(conds)), 4),
                offdiag_cov=round(float(np.mean(offs)), 6),
                avg_feature_std=round(float(np.mean(stds)), 6),
                anisotropy_mean=round(float(np.mean(anisos)), 6),
                notes=f"variant={variant} trainer=ncwp_ref",
            ))
            print(f"[STS/{model}] {variant} r={r} -> {np.mean(scores):.2f}±{np.std(scores):.2f} "
                  f"cond={np.mean(conds):.1f} offdiag={np.mean(offs):.4f} fstd={np.mean(stds):.4f} aniso={np.mean(anisos):.4f}")
    return rows


def run_quora(model, dims, seeds, variants):
    rows = []
    cfg = MODEL_CONFIGS[model]
    q = load_quora_cached(model)

    def eval_embs(c, qe):
        return evaluate_retrieval_ndcg10(c, qe, q["corpus_ids"], q["test_qids"], q["qrels"])

    for variant in variants:
        for r in dims:
            scores, conds = [], []
            for seed in seeds:
                SW, mu_in, mu_out, std_out = train_ncwp_variant(
                    q["X_fit"].astype(np.float32), r, variant=variant, seed=seed)
                c = proj_ncwp(q["corpus_np"].astype(np.float32), SW, mu_in, mu_out, std_out)
                qe = proj_ncwp(q["query_np"].astype(np.float32), SW, mu_in, mu_out, std_out)
                scores.append(eval_embs(c, qe))
                conds.append(float(np.linalg.cond(SW)))
            rows.append(metadata_row(
                dataset="quora", split="test", model_nickname=model,
                hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool", method=f"NCWP_{variant}",
                r=r, seed=str(seeds), fit_size=q["fit_size"], eval_size=q["eval_size"],
                metric_name="ndcg@10", metric_value=round(float(np.mean(scores)), 4),
                std_if_available=round(float(np.std(scores)), 4),
                cond_number=round(float(np.mean(conds)), 4),
                notes=f"variant={variant} trainer=ncwp_ref",
            ))
            print(f"[Quora/{model}] {variant} r={r} -> {np.mean(scores):.2f}±{np.std(scores):.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="sts")
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=[40, 80, 160])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--full", action="store_true", help="include no_qr/no_hardneg/no_refinement variants")
    parser.add_argument("--basename", default="table_regularizer_ablation")
    args = parser.parse_args()

    variants = FULL_VARIANTS if args.full else MIN_VARIANTS
    rows = []
    for model in args.model:
        if args.dataset in ("sts", "both"):
            rows.extend(run_sts(model, args.dims, args.seeds, variants))
        if args.dataset in ("quora", "both"):
            rows.extend(run_quora(model, args.dims, args.seeds, variants))
    csv_path, jsonl_path = save_v2(rows, args.basename)
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
