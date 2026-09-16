#!/usr/bin/env python3
"""Search for a regime where kNN-mined positives genuinely beat random/noise.

Hypothesis: neighbor mining matters when the fit corpus is LARGE and DIVERSE (so
random pairs are truly unrelated) and the positive is SPECIFIC (small k), possibly
with stabilizers OFF (so the positive is the only structure). We sweep those
levers and compare kNN(k) vs random vs noise on Quora retrieval, fitting on a
subsample of the full 522K Quora corpus (diverse) rather than the small 2K query
sample.

Scientific guardrail: we report the kNN−random gap with 3 seeds; a regime "counts"
only if kNN robustly (> seed noise) beats random, and it must make mechanistic
sense. Output: results/table_positive_search.csv
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from ncwp.common import MODEL_CONFIGS, evaluate_retrieval_ndcg10
from ncwp.ncwp_ref import proj_ncwp, train_ncwp_variant

OUT_ROOT_V2 = "/workspace/NCWP/results"
QCACHE = "/workspace/NCWP/quora_results/embedding_cache"


def save_v2(rows, basename):
    import pandas as pd
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT_V2, f"{basename}.csv"), index=False)
    with open(os.path.join(OUT_ROOT_V2, f"{basename}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_quora_eval(model):
    cache = f"{QCACHE}/{model}"
    corpus = np.load(f"{cache}/corpus_base.npy").astype(np.float32)
    query = np.load(f"{cache}/test_queries_base.npy").astype(np.float32)
    ds_corp = load_dataset("mteb/quora", "corpus", split="corpus")
    ds_qs = load_dataset("mteb/quora", "queries", split="queries")
    ds_qrel = load_dataset("mteb/quora", split="test")
    corpus_ids = [str(r["_id"]) for r in ds_corp]
    qid2tx = {str(r["_id"]): r["text"] for r in ds_qs}
    qrels = defaultdict(set)
    for r in ds_qrel:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    test_qids = sorted(q for q in qrels if q in qid2tx)
    return corpus, query, corpus_ids, test_qids, dict(qrels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen-4b", choices=list(MODEL_CONFIGS))
    p.add_argument("--fit_n", type=int, default=20000, help="diverse fit subsample from full corpus")
    p.add_argument("--r", type=int, default=80)
    p.add_argument("--r_list", type=int, nargs="+", default=None, help="sweep multiple r (overrides --r)")
    p.add_argument("--k_list", type=int, nargs="+", default=[1, 5, 10, 40])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--raw", action="store_true", help="no ZCA whitening")
    p.add_argument("--minimal", action="store_true", help="stabilizers OFF (no QR/reg/hardneg/refine)")
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--basename", default="table_positive_search")
    args = parser = p.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    corpus, query, corpus_ids, test_qids, qrels = load_quora_eval(args.model)
    rng = np.random.RandomState(0)
    fit_idx = rng.choice(len(corpus), min(args.fit_n, len(corpus)), replace=False)
    X_fit = corpus[fit_idx].copy()
    whiten = not args.raw

    stab = {}
    if args.minimal:
        stab = dict(retraction_interval=0, lambda_cov=0.0, lambda_orth=0.0,
                    topk_negatives=None, refine_knn_rounds=0)

    def ev(seed, variant, k, r):
        SW, mu_in, mu_out, std_out = train_ncwp_variant(
            X_fit, r, variant=variant, seed=seed, whiten=whiten,
            k_neighbors=k, cosine_tau=args.tau, **stab)
        c = proj_ncwp(corpus, SW, mu_in, mu_out, std_out)
        q = proj_ncwp(query, SW, mu_in, mu_out, std_out)
        return evaluate_retrieval_ndcg10(c, q, corpus_ids, test_qids, qrels)

    rows = []
    r_list = args.r_list if args.r_list is not None else [args.r]
    for r in r_list:
        tag = f"fit={args.fit_n} r={r} whiten={whiten} minimal={args.minimal} tau={args.tau}"
        print(f"=== positive search: {args.model} {tag} ===")
        ref = {}
        for name, variant in [("random", "random_pairs"), ("noise", "noise")]:
            s = [ev(sd, variant, 10, r) for sd in args.seeds]
            ref[name] = (float(np.mean(s)), float(np.std(s)))
            rows.append(dict(model=args.model, config=tag, r=r, positive=name, k=None,
                             ndcg=round(ref[name][0], 4), std=round(ref[name][1], 4)))
            print(f"  {name:8s}: {ref[name][0]:.3f} ± {ref[name][1]:.3f}")
        for k in args.k_list:
            s = [ev(sd, "full", k, r) for sd in args.seeds]
            m, sd = float(np.mean(s)), float(np.std(s))
            gap = m - ref["random"][0]
            rows.append(dict(model=args.model, config=tag, r=r, positive="knn", k=k,
                             ndcg=round(m, 4), std=round(sd, 4), gap_vs_random=round(gap, 4)))
            flag = "  <-- kNN>random!" if gap > max(ref["random"][1], sd, 0.2) else ""
            print(f"  knn k={k:<3d}: {m:.3f} ± {sd:.3f}  (gap vs random {gap:+.3f}){flag}")

    save_v2(rows, args.basename)
    print(f"Saved: {OUT_ROOT_V2}/{args.basename}.csv")


if __name__ == "__main__":
    main()
