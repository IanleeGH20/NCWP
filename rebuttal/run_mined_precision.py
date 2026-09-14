#!/usr/bin/env python3
"""Spec §5 (Review W3b): does whitened-space k-NN approximate relevance?

Reports precision@k / recall@k of mined nearest neighbors against gold labels, in
three spaces {raw mean-pool, ZCA-whitened, NCWP-projected}, at k=10, tau=0:
  - Quora : for each test query, top-k neighbors in the corpus vs gold duplicates (qrels).
  - STS-B : for each sentence, top-k neighbors vs gold high-similarity partners (score>=thr).

The reviewer specifically asked for the ZCA-whitened number (that is the space
NCWP mines positives in); raw + NCWP-projected are for comparison. NCWP uses the
faithful trainer (ncwp_ref, seed 0). Output: rebuttal_outputs/table_mined_positive_precision.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    STS_DIR,
    apply_zca,
    compute_knn,
    compute_zca,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    save_results,
)
from rebuttal.ncwp_ref import fit_project, proj_ncwp, train_ncwp_variant


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ── STS ───────────────────────────────────────────────────────
def sts_high_sim(threshold):
    train = json.load(open(f"{STS_DIR}/sts_train.json"))
    test = json.load(open(f"{STS_DIR}/sts_test.json"))
    partners = defaultdict(set)
    sents = set()
    for it in train + test:
        s1, s2, sc = it["sentence1"], it["sentence2"], it["score"]
        sents.add(s1); sents.add(s2)
        if sc >= threshold:
            partners[s1].add(s2); partners[s2].add(s1)
    return partners, sorted(sents)


def precision_recall(knn_idx, knn_sims, sent_list, partners, k, tau):
    precs, recs = [], []
    hit = 0
    for i, s in enumerate(sent_list):
        gold = partners.get(s, set())
        if not gold:
            continue
        mined = [sent_list[int(j)] for t, j in enumerate(knn_idx[i]) if knn_sims[i, t] >= tau][:k]
        if not mined:
            continue
        h = sum(1 for m in mined if m in gold)
        precs.append(h / len(mined))
        recs.append(h / min(len(gold), k))
        if h > 0:
            hit += 1
    return (float(np.mean(precs)) if precs else float("nan"),
            float(np.mean(recs)) if recs else float("nan"),
            hit / max(len(precs), 1), len(precs))


def run_sts(model, k, tau, thr, r_ncwp, seed):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark()
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    partners, sent_list = sts_high_sim(thr)
    sent_list = [s for s in sent_list if s in X_eval]  # keep sentences we have embeddings for
    raw = _norm(np.stack([X_eval[s] for s in sent_list]).astype(np.float32))

    mu_zca, s_zca = compute_zca(X_fit)
    zca = np.stack([apply_zca(X_eval[s].reshape(1, -1), mu_zca, s_zca)[0] for s in sent_list]).astype(np.float32)
    ncwp, _ = fit_project(X_fit.astype(np.float32), np.stack([X_eval[s] for s in sent_list]).astype(np.float32),
                          r_ncwp, seed=seed, variant="full")

    rows = []
    for space, mat in [("raw_mean_pool", raw), ("ZCA-whitened", zca), ("NCWP-projected", ncwp)]:
        ki, ks = compute_knn(mat.astype(np.float32), k=k)
        p, rec, hitrate, n = precision_recall(ki, ks, sent_list, partners, k, tau)
        rows.append(metadata_row(
            dataset="stsb", split="eval_all_sents", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method="mined_positive_precision", space=space, k=k, tau=tau,
            gold_threshold=f">={thr}", r=(r_ncwp if space == "NCWP-projected" else None), seed=seed,
            fit_size=sts["fit_size"], eval_size=len(sent_list), metric_name="precision@k",
            metric_value=round(p, 4), recall_at_k=round(rec, 4), anchor_hit_rate=round(hitrate, 4),
            notes=f"n_anchors={n}"))
        print(f"[STS/{model}] {space}: P@{k}={p:.3f} R@{k}={rec:.3f} hit={hitrate:.3f} n={n}")
    return rows


# ── Quora ─────────────────────────────────────────────────────
def quora_prec_recall(corpus, query, corpus_ids, test_qids, qrels, k):
    precs, recs = [], []
    hit = 0
    cid_arr = np.array(corpus_ids)
    for qi in range(0, len(test_qids), 500):
        qe = min(len(test_qids), qi + 500)
        sim = query[qi:qe] @ corpus.T
        top = np.argsort(-sim, axis=1)[:, :k]
        for ci, qidx in enumerate(range(qi, qe)):
            qid = test_qids[qidx]
            gold = qrels.get(qid, set())
            if not gold:
                continue
            mined = set(cid_arr[top[ci]].tolist())
            h = len(mined & gold)
            precs.append(h / k)
            recs.append(h / min(len(gold), k))
            if h > 0:
                hit += 1
    return (float(np.mean(precs)), float(np.mean(recs)), hit / max(len(precs), 1), len(precs))


def run_quora(model, k, r_ncwp, seed):
    cfg = MODEL_CONFIGS[model]
    q = load_quora_cached(model)
    corpus, query, X_fit = q["corpus_np"], q["query_np"], q["X_fit"]

    mu_zca, s_zca = compute_zca(X_fit)
    spaces = {
        "raw_mean_pool": (_norm(corpus), _norm(query)),
        "ZCA-whitened": (apply_zca(corpus, mu_zca, s_zca), apply_zca(query, mu_zca, s_zca)),
    }
    SW, mu_in, mu_out, std_out = train_ncwp_variant(X_fit.astype(np.float32), r_ncwp, variant="full", seed=seed)
    spaces["NCWP-projected"] = (proj_ncwp(corpus.astype(np.float32), SW, mu_in, mu_out, std_out),
                                proj_ncwp(query.astype(np.float32), SW, mu_in, mu_out, std_out))

    rows = []
    for space, (c, qe) in spaces.items():
        p, rec, hitrate, n = quora_prec_recall(c.astype(np.float32), qe.astype(np.float32),
                                                q["corpus_ids"], q["test_qids"], q["qrels"], k)
        rows.append(metadata_row(
            dataset="quora", split="test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type="mean_pool", method="mined_positive_precision", space=space, k=k, tau=0.0,
            r=(r_ncwp if space == "NCWP-projected" else None), seed=seed,
            fit_size=q["fit_size"], eval_size=n, metric_name="precision@k",
            metric_value=round(p, 4), recall_at_k=round(rec, 4), anchor_hit_rate=round(hitrate, 4),
            notes="query->corpus NN vs gold duplicates (qrels)"))
        print(f"[Quora/{model}] {space}: P@{k}={p:.3f} R@{k}={rec:.3f} hit={hitrate:.3f} n={n}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dataset", choices=["sts", "quora", "both"], default="both")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--sts_threshold", type=float, default=4.0)
    parser.add_argument("--r_ncwp", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for model in args.model:
        if args.dataset in ("sts", "both"):
            rows.extend(run_sts(model, args.k, args.tau, args.sts_threshold, args.r_ncwp, args.seed))
        if args.dataset in ("quora", "both"):
            rows.extend(run_quora(model, args.k, args.r_ncwp, args.seed))
    csv_path, jsonl_path = save_results(rows, "table_mined_positive_precision")
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
