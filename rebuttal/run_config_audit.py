#!/usr/bin/env python3
"""Final-NCWP-configuration audit (K / QR interval / Eq(6)-(7) regularizers).

Why this exists: the paper appendix documents K=32 hard negatives, N_qr=200, and a
4096-entry memory bank, but the code that produced every table
(`run_ablation.py` == `rebuttal/ncwp_ref.py`) uses K=256, N_qr=10, one kNN
refinement round, and NO memory bank. The rebuttal additionally promised to drop
the Eq (6)/(7) regularizers from the default objective. None of the result CSVs
record these fields, so we cannot tell from the tables which setting produced
them.

This runner re-measures the main STS-B protocol under three explicit configs and
writes every hyperparameter into the output row, so the camera-ready tables can
state the setting they were produced with:

  paper_appendix   K=32,  N_qr=200, regularizers on   (what §A claims)
  code_actual      K=256, N_qr=10,  regularizers on   (what produced the tables)
  camera_ready     K=256, N_qr=10,  regularizers off  (what the rebuttal promises)

If `code_actual` reproduces the published main numbers and `camera_ready` stays
within seed noise, the main results stand and only the appendix text needs
fixing. Otherwise the main results must be re-run.

STS-B honours `--eval_split` (train+test 7,128 or the paper's test-only 1,379).
Quora fits on a corpus subsample (evaluation queries excluded), with the same
per-model fit N as main — i.e. both datasets run the protocol the camera-ready
will actually report, so the method description and the numbers line up.

Cached embeddings only (no model loading). Re-verification of a published result
-> seeds 42,43,44. Output: rebuttal_outputs_v2/table_ncwp_config_audit.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from rebuttal.common import (
    MODEL_CONFIGS,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    spearman_sts,
)
from rebuttal.ncwp_ref import proj_ncwp, train_ncwp_variant
from rebuttal.run_quora_corpus_sample import FIT_SEED, gpu_ndcg_recall

OUT_ROOT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"
REBUTTAL_DIMS = {
    "qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448], "llama-8b": [32, 64, 128, 256],
}

# Every field the result CSVs were missing. `memory_bank` is not a knob in
# ncwp_ref (the paper trainer has none) — recorded so the mismatch is explicit.
CONFIGS = {
    "paper_appendix": dict(topk_negatives=32, retraction_interval=200,
                           lambda_cov=0.05, lambda_orth=0.02),
    "code_actual": dict(topk_negatives=256, retraction_interval=10,
                        lambda_cov=0.05, lambda_orth=0.02),
    "camera_ready": dict(topk_negatives=256, retraction_interval=10,
                         lambda_cov=0.0, lambda_orth=0.0),
}


def save_v2(rows, basename):
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT_V2, f"{basename}.csv")
    jsonl_path = os.path.join(OUT_ROOT_V2, f"{basename}.jsonl")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def config_fields(name, hp):
    return dict(
        config_name=name,
        k_neighbors=10, cosine_tau=0.0, temperature=0.12, shrink=0.08,
        hard_neg_K=hp["topk_negatives"], qr_interval=hp["retraction_interval"],
        lambda_cov=hp["lambda_cov"], lambda_orth=hp["lambda_orth"],
        memory_bank_size=0, refine_knn_rounds=1,
    )


def run_sts(model, dims, seeds, configs, eval_split):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark(eval_split=eval_split)
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval)
    X_eval_mat = np.stack([X_eval[s] for s in keys]).astype(np.float32)
    rows = []

    for name in configs:
        hp = CONFIGS[name]
        for r in dims:
            ss = []
            for seed in seeds:
                SW, mu_in, mu_out, std_out = train_ncwp_variant(
                    X_fit.astype(np.float32), r, variant="full", seed=seed, **hp)
                Z = proj_ncwp(X_eval_mat.astype(np.float32), SW, mu_in, mu_out, std_out)
                ss.append(spearman_sts({keys[i]: Z[i] for i in range(len(keys))},
                                       sts["eval_pairs"], sts["eval_scores"]))
            rows.append(metadata_row(
                dataset="stsb", split=eval_split, model_nickname=model,
                hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool",
                method="NCWP_mean_pool", r=r, seed=str(list(seeds)),
                fit_size=sts["fit_size"], eval_size=sts["eval_size"],
                metric_name="spearman_x100", metric_value=round(float(np.mean(ss)), 4),
                std_if_available=round(float(np.std(ss)), 4),
                per_seed=[round(float(v), 4) for v in ss],
                notes="ncwp_ref config audit", **config_fields(name, hp)))
            print(f"[STS/{model}] {name:14s} r={r:>4} -> {np.mean(ss):.2f}±{np.std(ss):.2f}",
                  flush=True)
    return rows


def gpu_proj_ncwp(X_np, SW, mu_in, mu_out, std_out, chunk=20000):
    """NCWP projection of a large matrix on GPU, returned as fp16 for top-k."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SW_t = torch.from_numpy(SW).to(dev)
    mu_in_t = torch.from_numpy(mu_in).to(dev)
    mu_out_t = torch.from_numpy(mu_out).to(dev)
    std_out_t = torch.from_numpy(std_out).to(dev)
    parts = []
    for i in range(0, len(X_np), chunk):
        xb = torch.from_numpy(X_np[i:i + chunk]).to(dev)
        z = ((xb - mu_in_t) @ SW_t - mu_out_t) / std_out_t
        parts.append(F.normalize(z, dim=1).half())
    return torch.cat(parts, 0)


def run_quora(model, dims, seeds, configs):
    """Quora under corpus_sample fitting — the protocol the camera-ready will report
    (evaluation queries excluded from fitting), with the same per-model fit N as main."""
    cfg = MODEL_CONFIGS[model]
    q = load_quora_cached(model)  # full corpus/query embeddings + qrels
    corpus, queries = q["corpus_np"], q["query_np"]
    fit_n = cfg["quora_fit_size"]
    rng = np.random.RandomState(FIT_SEED)
    X_fit = corpus[rng.choice(len(corpus), fit_n, replace=False)].astype(np.float32)
    print(f"[Quora/{model}] corpus={corpus.shape} fit(corpus_sample N={fit_n})={X_fit.shape}",
          flush=True)
    rows = []

    for name in configs:
        hp = CONFIGS[name]
        for r in dims:
            ss = []
            for seed in seeds:
                SW, mu_in, mu_out, std_out = train_ncwp_variant(
                    X_fit, r, variant="full", seed=seed, **hp)
                c = gpu_proj_ncwp(corpus.astype(np.float32), SW, mu_in, mu_out, std_out)
                qq = gpu_proj_ncwp(queries.astype(np.float32), SW, mu_in, mu_out, std_out)
                val, _ = gpu_ndcg_recall(c, qq, q["corpus_ids"], q["test_qids"], q["qrels"])
                ss.append(val)
                del c, qq
                torch.cuda.empty_cache()
            rows.append(metadata_row(
                dataset="quora", split="test", model_nickname=model,
                hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool",
                method="NCWP_mean_pool", r=r, seed=str(list(seeds)),
                fit_mode="corpus_sample", fit_size=fit_n, eval_size=q["eval_size"],
                metric_name="ndcg@10", metric_value=round(float(np.mean(ss)), 4),
                std_if_available=round(float(np.std(ss)), 4),
                per_seed=[round(float(v), 4) for v in ss],
                notes="ncwp_ref config audit; corpus_sample fitting",
                **config_fields(name, hp)))
            print(f"[Quora/{model}] {name:14s} r={r:>4} -> {np.mean(ss):.2f}±{np.std(ss):.2f}",
                  flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="sts")
    parser.add_argument("--model", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--eval_split", choices=["train+test", "test"], default="train+test")
    parser.add_argument("--basename", default="table_ncwp_config_audit")
    args = parser.parse_args()

    rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else REBUTTAL_DIMS[model]
        if args.dataset in ("sts", "both"):
            rows.extend(run_sts(model, dims, args.seeds, args.configs, args.eval_split))
        if args.dataset in ("quora", "both"):
            rows.extend(run_quora(model, dims, args.seeds, args.configs))
        save_v2(rows, args.basename)  # checkpoint per model
    csv_path, jsonl_path = save_v2(rows, args.basename)
    print(f"\nSaved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
