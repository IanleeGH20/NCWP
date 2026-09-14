#!/usr/bin/env python3
"""Spec §4 (Review W3a): positive-generation ablation.

Same NCWP training (faithful ncwp_ref: identical loss/schedule/QR/regularizers),
changing ONLY how positives are constructed:
  - NCWP_knn_positive    : kNN-mined positives (paper NCWP)      -> variant 'full'
  - NCWP_noise_positive  : Gaussian-noise views (SimCSE-noise)   -> variant 'noise'
  - NCWP_random_positive : random positives (sanity control)     -> variant 'random_pairs'
  - NCWP_dropout_positive: SimCSE dropout — NOT APPLICABLE for frozen decoder-only
    LLM embeddings (inference dropout ~0 -> degenerate identical views); reported
    as not_applicable per spec §4.2, with noise positive as the augmentation control.

Re-examination of the published method's positive source -> seeds 42,43,44.
Cached embeddings. Output: rebuttal_outputs/table_positive_generation_ablation.
"""
import argparse

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    evaluate_retrieval_ndcg10,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    save_results,
    spearman_sts,
)
from rebuttal.ncwp_ref import fit_project, proj_ncwp, train_ncwp_variant

MODES = [
    ("NCWP_knn_positive", "full"),
    ("NCWP_noise_positive", "noise"),
    ("NCWP_random_positive", "random_pairs"),
]


def run_sts(model, dims, seeds, noise_sigma, whiten=True):
    rows = []
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark()
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval)
    X_eval_mat = np.stack([X_eval[s] for s in keys]).astype(np.float32)

    def add(method, r, val, std, notes):
        rows.append(metadata_row(
            dataset="stsb", split="train+test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method=method, r=r, seed=str(seeds),
            fit_size=sts["fit_size"], eval_size=sts["eval_size"], metric_name="spearman_x100",
            metric_value=(round(val, 4) if val is not None else None),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    for method, variant in MODES:
        for r in dims:
            ss = []
            for seed in seeds:
                Z, _ = fit_project(X_fit, X_eval_mat, r, seed=seed, variant=variant,
                                   noise_sigma=noise_sigma, whiten=whiten)
                emb_map = {keys[i]: Z[i] for i in range(len(keys))}
                ss.append(spearman_sts(emb_map, sts["eval_pairs"], sts["eval_scores"]))
            add(method, r, float(np.mean(ss)), float(np.std(ss)),
                f"variant={variant} whiten={whiten}" + (f" sigma={noise_sigma}" if variant == "noise" else ""))
            print(f"[STS/{model}] {method} r={r} -> {np.mean(ss):.2f}±{np.std(ss):.2f}")
    for r in dims:
        add("NCWP_dropout_positive", r, None, None, "not_applicable: frozen decoder-only inference dropout~0")
    return rows


def run_quora(model, dims, seeds, noise_sigma, whiten=True):
    rows = []
    cfg = MODEL_CONFIGS[model]
    q = load_quora_cached(model)

    def eval_embs(c, qe):
        return evaluate_retrieval_ndcg10(c, qe, q["corpus_ids"], q["test_qids"], q["qrels"])

    def add(method, r, val, std, notes):
        rows.append(metadata_row(
            dataset="quora", split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method=method, r=r, seed=str(seeds),
            fit_size=q["fit_size"], eval_size=q["eval_size"], metric_name="ndcg@10",
            metric_value=(round(val, 4) if val is not None else None),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    for method, variant in MODES:
        for r in dims:
            ss = []
            for seed in seeds:
                SW, mu_in, mu_out, std_out = train_ncwp_variant(
                    q["X_fit"].astype(np.float32), r, variant=variant, seed=seed,
                    noise_sigma=noise_sigma, whiten=whiten)
                c = proj_ncwp(q["corpus_np"].astype(np.float32), SW, mu_in, mu_out, std_out)
                qe = proj_ncwp(q["query_np"].astype(np.float32), SW, mu_in, mu_out, std_out)
                ss.append(eval_embs(c, qe))
            add(method, r, float(np.mean(ss)), float(np.std(ss)),
                f"variant={variant} whiten={whiten}" + (f" sigma={noise_sigma}" if variant == "noise" else ""))
            print(f"[Quora/{model}] {method} r={r} -> {np.mean(ss):.2f}±{np.std(ss):.2f}")
    for r in dims:
        add("NCWP_dropout_positive", r, None, None, "not_applicable: frozen decoder-only inference dropout~0")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="sts")
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=[40, 80, 160])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--basename", default="table_positive_generation_ablation")
    parser.add_argument("--raw", action="store_true",
                        help="ablate WITHOUT ZCA whitening (tests if kNN positives matter in raw space)")
    args = parser.parse_args()

    whiten = not args.raw
    rows = []
    for model in args.model:
        if args.dataset in ("sts", "both"):
            rows.extend(run_sts(model, args.dims, args.seeds, args.noise_sigma, whiten))
        if args.dataset in ("quora", "both"):
            rows.extend(run_quora(model, args.dims, args.seeds, args.noise_sigma, whiten))
    csv_path, jsonl_path = save_results(rows, args.basename)
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
