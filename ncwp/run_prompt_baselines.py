#!/usr/bin/env python3
"""Spec §1: PromptEOL / Echo training-free baselines + NCWP-on-prompt (STS-B, Quora).

Answers reviewer W2: is NCWP on mean-pooled states better than prompt-based
extraction (PromptEOL / Echo) alone, and does NCWP stacked on prompt embeddings
preserve compression gains?

Embedding sources:
  - mean_pool : paper Base (cached in sts_ablation_cache) -> Base_mean_pool, NCWP_mean_pool
  - PromptEOL : template A/B, last-token pooling            -> PromptEOL, NCWP_on_PromptEOL
  - Echo      : "{s} {s}", mean & last pooling              -> Echo, NCWP_on_Echo

NCWP uses the FAITHFUL paper trainer (ncwp.ncwp_ref), which whitens (ZCA)
internally; only the raw embedding source changes across variants. Prompt-source
embeddings are cached to prompt_cache/ so trainer swaps / reruns skip re-encoding.

New experiment -> default seeds 0,1,2.
"""
import argparse
import json
import os

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    PromptEmbedder,
    evaluate_retrieval_ndcg10,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    save_results,
    spearman_sts,
    subset_eval_from_cache,
)
from ncwp.ncwp_ref import fit_project, proj_ncwp, train_ncwp_variant

from ncwp.common import PROMPT_CACHE as CACHE_DIR

# When eval_split="test", reuse the cached 7128-eval arrays and subset to test
# sentences instead of re-encoding with the model.
EVAL_SPLIT = "train+test"

TARGET_DIMS = {
    "qwen-4b": [40, 80, 160, 320],
    "qwen-8b": [56, 112, 224, 448],
    "llama-8b": [32, 64, 128, 256],
}

# (method_name, template_name, pool)
PROMPT_SOURCES = [
    ("PromptEOL_A", "PromptEOL_A", "last"),
    ("PromptEOL_B", "PromptEOL_B", "last"),
    ("Echo_mean", "Echo", "mean"),
    ("Echo_last", "Echo", "last"),
]


def encode_or_load(embedder, model, source_tag, split, sents):
    """Encode `sents` with `embedder`, caching to prompt_cache/ keyed by sentence order.

    For eval on test-only, reuse the cached 7128-eval array by subsetting (no
    re-encode) as long as the cache exists.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    npy = os.path.join(CACHE_DIR, f"{model}__{source_tag}__{split}.npy")
    sj = os.path.join(CACHE_DIR, f"{model}__{source_tag}__{split}.sents.json")
    if os.path.exists(npy) and os.path.exists(sj):
        saved = json.load(open(sj))
        if saved == sents:
            return np.load(npy).astype(np.float32)
        # test-only subset of the cached full-eval array
        if set(sents).issubset(set(saved)):
            return subset_eval_from_cache(npy, sj, sents)
    arr = embedder.encode(sents).astype(np.float32)
    np.save(npy, arr)
    json.dump(sents, open(sj, "w"))
    return arr


def ncwp_sts(X_fit, X_eval_mat, eval_keys, sts, dims, seeds):
    """{r: (mean, std)} for NCWP (faithful trainer) fit on X_fit, eval on matrix."""
    out = {}
    for r in dims:
        ss = []
        for seed in seeds:
            Z, _ = fit_project(X_fit, X_eval_mat, r, seed=seed, variant="full")
            emb_map = {eval_keys[i]: Z[i] for i in range(len(eval_keys))}
            ss.append(spearman_sts(emb_map, sts["eval_pairs"], sts["eval_scores"]))
        out[r] = (float(np.mean(ss)), float(np.std(ss)))
    return out


def run_sts(model, dims, seeds):
    rows = []
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    sts = load_sts_benchmark(eval_split=EVAL_SPLIT)
    eval_keys = sts["eval_sents"]

    def add(method, r, val, std=None, emb_src="", notes=""):
        rows.append(metadata_row(
            dataset="stsb", split=EVAL_SPLIT, model_nickname=model,
            hf_model_id=cfg["hf_model_id"], embedding_type=emb_src, method=method,
            r=r, seed=str(seeds), fit_size=sts["fit_size"], eval_size=sts["eval_size"],
            metric_name="spearman_x100", metric_value=round(val, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes,
        ))

    # ── mean-pool source (paper Base / NCWP), raw cached embeddings ──
    X_fit_mp, X_eval_mp, _ = load_sts_embeddings(model, sts)
    X_eval_mp_mat = np.stack([X_eval_mp[s] for s in eval_keys]).astype(np.float32)
    base_map = {s: X_eval_mp[s] / (np.linalg.norm(X_eval_mp[s]) + 1e-12) for s in eval_keys}
    base = spearman_sts(base_map, sts["eval_pairs"], sts["eval_scores"])
    add("Base_mean_pool", d_full, base, emb_src="mean_pool", notes="paper Base (cached)")
    print(f"[STS/{model}] Base_mean_pool D={d_full} -> {base:.2f}")

    for r, (m, s) in ncwp_sts(X_fit_mp, X_eval_mp_mat, eval_keys, sts, dims, seeds).items():
        add("NCWP_mean_pool", r, m, s, emb_src="mean_pool", notes="ncwp_ref; new-seed reverify vs Table 1")
        print(f"[STS/{model}] NCWP_mean_pool r={r} -> {m:.2f}±{s:.2f}")

    # ── prompt sources (encode once w/ cache, reuse model across templates) ──
    embedder = PromptEmbedder(model, template_name="PromptEOL_A", pool="last")
    for method, tmpl, pool in PROMPT_SOURCES:
        embedder.template_name = tmpl
        embedder.pool = pool
        X_fit_p = encode_or_load(embedder, model, method, "fit", sts["fit_sents"])
        X_eval_p = encode_or_load(embedder, model, method, "eval", eval_keys)

        pmap = {eval_keys[i]: X_eval_p[i] / (np.linalg.norm(X_eval_p[i]) + 1e-12) for i in range(len(eval_keys))}
        pfull = spearman_sts(pmap, sts["eval_pairs"], sts["eval_scores"])
        add(method, d_full, pfull, emb_src=method, notes=f"template={tmpl} pool={pool}")
        print(f"[STS/{model}] {method} (full D={d_full}) -> {pfull:.2f}")

        stacked = "NCWP_on_PromptEOL" if method.startswith("PromptEOL") else "NCWP_on_Echo"
        for r, (m, s) in ncwp_sts(X_fit_p, X_eval_p, eval_keys, sts, dims, seeds).items():
            add(stacked, r, m, s, emb_src=method, notes=f"ncwp_ref; template={tmpl} pool={pool}")
            print(f"[STS/{model}] {stacked}[{method}] r={r} -> {m:.2f}±{s:.2f}")

    return rows


def run_quora(model, dims, seeds):
    rows = []
    cfg = MODEL_CONFIGS[model]
    d_full = cfg["hidden_dim"]
    q = load_quora_cached(model)

    def eval_embs(c, qe):
        return evaluate_retrieval_ndcg10(c, qe, q["corpus_ids"], q["test_qids"], q["qrels"])

    def add(method, r, val, std=None, emb_src="", notes=""):
        rows.append(metadata_row(
            dataset="quora", split="test", model_nickname=model,
            hf_model_id=cfg["hf_model_id"], embedding_type=emb_src, method=method,
            r=r, seed=str(seeds), fit_size=q["fit_size"], eval_size=q["eval_size"],
            metric_name="ndcg@10", metric_value=round(val, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes,
        ))

    base = eval_embs(q["corpus_np"], q["query_np"])
    add("Base_mean_pool", d_full, base, emb_src="mean_pool", notes="paper Base (cached)")
    print(f"[Quora/{model}] Base_mean_pool D={d_full} -> {base:.2f}")

    for r in dims:
        ss = []
        for seed in seeds:
            SW, mu_in, mu_out, std_out = train_ncwp_variant(q["X_fit"].astype(np.float32), r, seed=seed, variant="full")
            c = proj_ncwp(q["corpus_np"].astype(np.float32), SW, mu_in, mu_out, std_out)
            qe = proj_ncwp(q["query_np"].astype(np.float32), SW, mu_in, mu_out, std_out)
            ss.append(eval_embs(c, qe))
        add("NCWP_mean_pool", r, float(np.mean(ss)), float(np.std(ss)), emb_src="mean_pool", notes="ncwp_ref")
        print(f"[Quora/{model}] NCWP_mean_pool r={r} -> {np.mean(ss):.2f}")

    print(f"[Quora/{model}] prompt-source variants skipped (heavy 522K encode; add later)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="sts")
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval_split", choices=["train+test", "test"], default="train+test")
    args = parser.parse_args()

    global EVAL_SPLIT
    EVAL_SPLIT = args.eval_split
    suffix = "" if EVAL_SPLIT == "train+test" else "_testonly1379"

    if args.dataset in ("sts", "both"):
        rows = []
        for model in args.model:
            dims = args.dims if args.dims is not None else TARGET_DIMS[model]
            rows.extend(run_sts(model, dims, args.seeds))
        csv_path, jsonl_path = save_results(rows, f"table_prompt_baseline_stsb{suffix}")
        print(f"Saved: {csv_path}\n       {jsonl_path}")
    if args.dataset in ("quora", "both"):
        rows = []
        for model in args.model:
            dims = args.dims if args.dims is not None else TARGET_DIMS[model]
            rows.extend(run_quora(model, dims, args.seeds))
        csv_path, jsonl_path = save_results(rows, "table_prompt_baseline_quora")
        print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
