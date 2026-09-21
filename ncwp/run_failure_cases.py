#!/usr/bin/env python3
"""v2 spec §6 (Review 3 W3): mined-positive failure cases.

Collect pairs that are mined as positives in the ZCA-whitened space (k=10, tau=0)
but are lexically close yet semantically far:
  - STS-B : mined pair with gold score <= 2.0 and high token Jaccard.
  - Quora : mined corpus item NOT labeled duplicate, with high token Jaccard to the query.

Human-readable examples for the reproduction. Cached embeddings + ZCA (closed-form),
no NCWP training needed. Output: results/mined_positive_failure_cases.{csv,md}.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    STS_DIR,
    apply_zca,
    compute_knn,
    compute_zca,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
)

from ncwp.common import OUT_ROOT as OUT_ROOT_V2


def jaccard(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sts_gold_scores():
    train = json.load(open(f"{STS_DIR}/sts_train.json"))
    test = json.load(open(f"{STS_DIR}/sts_test.json"))
    g = {}
    for it in train + test:
        g[frozenset((it["sentence1"], it["sentence2"]))] = it["score"]
    return g


def run_sts(model, k, tau, gold_max, jac_min, topn):
    sts = load_sts_benchmark()
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    gold = sts_gold_scores()
    sents = list(X_eval)
    mu, s = compute_zca(X_fit)
    zca = np.stack([apply_zca(X_eval[x].reshape(1, -1), mu, s)[0] for x in sents]).astype(np.float32)
    ki, ks = compute_knn(zca, k=k)
    cases = []
    for i, a in enumerate(sents):
        for t, j in enumerate(ki[i]):
            if ks[i, t] < tau:
                continue
            b = sents[int(j)]
            sc = gold.get(frozenset((a, b)))
            if sc is None or sc > gold_max:
                continue
            jac = jaccard(a, b)
            if jac >= jac_min:
                cases.append(dict(dataset="stsb", model=model, anchor_text=a, neighbor_text=b,
                                  space="ZCA-whitened", rank=t + 1, cosine_similarity=round(float(ks[i, t]), 4),
                                  gold_label_or_score=sc, lexical_overlap=round(jac, 3),
                                  notes=f"gold<= {gold_max}, jaccard>= {jac_min}"))
    cases.sort(key=lambda c: (-c["lexical_overlap"], -c["cosine_similarity"]))
    print(f"[STS/{model}] {len(cases)} failure candidates (gold<={gold_max}, jaccard>={jac_min})")
    return cases[:topn], len(cases)


def run_quora(model, k, tau, jac_min, topn):
    from datasets import load_dataset
    q = load_quora_cached(model)
    corpus, query = q["corpus_np"], q["query_np"]
    ds_corp = load_dataset("mteb/quora", "corpus", split="corpus")
    ds_qs = load_dataset("mteb/quora", "queries", split="queries")
    cid2tx = {str(r["_id"]): r["text"] for r in ds_corp}
    qid2tx = {str(r["_id"]): r["text"] for r in ds_qs}
    corpus_ids, test_qids, qrels = q["corpus_ids"], q["test_qids"], q["qrels"]

    mu, s = compute_zca(q["X_fit"])
    c_z = apply_zca(corpus, mu, s).astype(np.float32)
    q_z = apply_zca(query, mu, s).astype(np.float32)
    cid_arr = np.array(corpus_ids)
    cases = []
    for qi in range(0, len(test_qids), 500):
        qe = min(len(test_qids), qi + 500)
        sim = q_z[qi:qe] @ c_z.T
        top = np.argsort(-sim, axis=1)[:, :k]
        for ci, qidx in enumerate(range(qi, qe)):
            qid = test_qids[qidx]
            gold = qrels.get(qid, set())
            qtext = qid2tx.get(qid, "")
            for rank, cix in enumerate(top[ci]):
                cid = str(cid_arr[cix])
                if cid in gold:
                    continue
                ctext = cid2tx.get(cid, "")
                jac = jaccard(qtext, ctext)
                if jac >= jac_min:
                    cases.append(dict(dataset="quora", model=model, anchor_text=qtext, neighbor_text=ctext,
                                      space="ZCA-whitened", rank=rank + 1,
                                      cosine_similarity=round(float(sim[ci, cix]), 4),
                                      gold_label_or_score="not_duplicate", lexical_overlap=round(jac, 3),
                                      notes=f"mined but not duplicate, jaccard>= {jac_min}"))
    cases.sort(key=lambda c: (-c["lexical_overlap"], -c["cosine_similarity"]))
    print(f"[Quora/{model}] {len(cases)} failure candidates (non-dup, jaccard>={jac_min})")
    return cases[:topn], len(cases)


def save(cases, counts):
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    cols = ["dataset", "model", "anchor_text", "neighbor_text", "space", "rank",
            "cosine_similarity", "gold_label_or_score", "lexical_overlap", "notes"]
    with open(os.path.join(OUT_ROOT_V2, "mined_positive_failure_cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(cases)
    with open(os.path.join(OUT_ROOT_V2, "mined_positive_failure_cases.md"), "w") as f:
        f.write("# Mined-positive failure cases (lexically close, semantically far)\n\n")
        f.write("ZCA-whitened space, k=10, tau=0. Counts of failure candidates:\n\n")
        for kk, vv in counts.items():
            f.write(f"- {kk}: {vv}\n")
        f.write("\n")
        for c in cases:
            f.write(f"**[{c['dataset']}/{c['model']}]** rank {c['rank']}, cos={c['cosine_similarity']}, "
                    f"gold={c['gold_label_or_score']}, jaccard={c['lexical_overlap']}\n")
            f.write(f"- anchor: {c['anchor_text']}\n- neighbor: {c['neighbor_text']}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="both")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--sts_gold_max", type=float, default=2.0)
    parser.add_argument("--jaccard_min", type=float, default=0.5)
    parser.add_argument("--topn", type=int, default=30)
    args = parser.parse_args()

    all_cases, counts = [], {}
    for model in args.model:
        if args.dataset in ("sts", "both"):
            cs, n = run_sts(model, args.k, args.tau, args.sts_gold_max, args.jaccard_min, args.topn)
            all_cases += cs; counts[f"stsb/{model}"] = n
        if args.dataset in ("quora", "both"):
            cs, n = run_quora(model, args.k, args.tau, args.jaccard_min, args.topn)
            all_cases += cs; counts[f"quora/{model}"] = n
    save(all_cases, counts)
    print(f"Saved {len(all_cases)} examples -> {OUT_ROOT_V2}/mined_positive_failure_cases.{{csv,md}}")


if __name__ == "__main__":
    main()
