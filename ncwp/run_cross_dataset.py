#!/usr/bin/env python3
"""v2 spec §4 (Review 2 + 3): cross-dataset / unseen-task generalization.

Does NCWP transfer, or must whitening/projection be refit per corpus (the
inference-cost concern R2 raised)? Three settings, all from cached embeddings:
  A. STS-to-STS  : fit on STS-B, test on STS12-16/SICK-R; vs in-domain refit vs Base.
  B. Cross-task  : STS-B->Quora and Quora->STS-B (stress test).
  C. Quora->CQADupStack : fit on Quora, test on CQADupStack subforums.
Reports transfer score, in-domain-refit score, drop, and still-above-Base.
STS target embeddings come from run_sts_suite's prompt_cache (Plain_mean). NCWP
via ncwp_ref. New -> seeds 0,1,2. Output: results/table_cross_dataset_transfer_{sts,retrieval}.
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
    evaluate_retrieval_ndcg10,
    load_quora_cached,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
    pca_whitening_matrix,
    project_linear,
)
from ncwp.ncwp_ref import proj_ncwp, train_ncwp_variant

PROMPT_CACHE = "/workspace/NCWP/results/prompt_cache"
OUT_ROOT_V2 = "/workspace/NCWP/results"
STS_TARGETS = {"STS12": "mteb/sts12-sts", "STS13": "mteb/sts13-sts", "STS14": "mteb/sts14-sts",
               "STS15": "mteb/sts15-sts", "STS16": "mteb/sts16-sts", "SICK-R": "mteb/sickr-sts"}
CQA = {"english": "mteb/cqadupstack-english", "gaming": "mteb/cqadupstack-gaming", "physics": "mteb/cqadupstack-physics"}


def save_v2(rows, basename):
    import pandas as pd
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT_V2, f"{basename}.csv"), index=False)
    with open(os.path.join(OUT_ROOT_V2, f"{basename}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sp(Z, sents, pairs, scores):
    idx = {s: i for i, s in enumerate(sents)}
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    pred = [float(Zn[idx[a]] @ Zn[idx[b]]) for a, b in pairs]
    return float(spearmanr(scores, pred).correlation * 100)


def load_target_sts(model, name):
    npy = f"{PROMPT_CACHE}/{model}__{name}__Plain_mean.npy"
    sj = f"{PROMPT_CACHE}/{model}__{name}__Plain_mean.sents.json"
    if not (os.path.exists(npy) and os.path.exists(sj)):
        return None
    X = np.load(npy).astype(np.float32)
    sents = json.load(open(sj))
    ds = load_dataset(STS_TARGETS[name], split="test")
    pairs = [(r["sentence1"], r["sentence2"]) for r in ds]
    scores = [float(r["score"]) for r in ds]
    return X, sents, pairs, scores


def sts_to_sts(model, dims, seeds, targets):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark()
    X_fit_stsb, _, _ = load_sts_embeddings(model, sts)
    rows = []
    for name in targets:
        loaded = load_target_sts(model, name)
        if loaded is None:
            print(f"[skip] no STS-suite cache for {name}/{model} (run run_sts_suite first)")
            continue
        Xt, sents, pairs, scores = loaded
        base = sp(Xt, sents, pairs, scores)

        def add(method, r, fit_src, val, drop=None):
            rows.append(metadata_row(
                train_dataset=fit_src, test_dataset=name, model_nickname=model, hf_model_id=cfg["hf_model_id"],
                method=method, r=r, seed=str(seeds), metric_name="spearman_x100", metric_value=round(val, 4),
                drop_vs_refit=(round(drop, 4) if drop is not None else None),
                still_above_base=bool(val > base), notes=f"base={base:.2f}"))

        add("Base", cfg["hidden_dim"], "None", base)
        for r in dims:
            # PCA transfer vs refit
            mu_s, w_s = pca_whitening_matrix(X_fit_stsb, r)
            pca_tr = sp((Xt - mu_s) @ w_s, sents, pairs, scores)
            mu_t, w_t = pca_whitening_matrix(Xt, r)
            pca_rf = sp((Xt - mu_t) @ w_t, sents, pairs, scores)
            add("PCA_refit", r, name, pca_rf, 0.0)
            add("PCA_transfer", r, "STS-B", pca_tr, pca_rf - pca_tr)
            # NCWP transfer vs refit
            tr = [sp(proj_ncwp(Xt, *train_ncwp_variant(X_fit_stsb, r, variant="full", seed=s)), sents, pairs, scores) for s in seeds]
            rf = [sp(proj_ncwp(Xt, *train_ncwp_variant(Xt, r, variant="full", seed=s)), sents, pairs, scores) for s in seeds]
            add("NCWP_refit", r, name, float(np.mean(rf)), 0.0)
            add("NCWP_transfer", r, "STS-B", float(np.mean(tr)), float(np.mean(rf) - np.mean(tr)))
            print(f"[STS-to-STS {name}/{model}] r={r} base={base:.2f} NCWP refit={np.mean(rf):.2f} transfer={np.mean(tr):.2f}")
    return rows


def cross_task_and_cqa(model, dims, seeds):
    cfg = MODEL_CONFIGS[model]
    rows = []
    sts = load_sts_benchmark()
    X_fit_stsb, X_eval_stsb, _ = load_sts_embeddings(model, sts)
    keys = list(X_eval_stsb)
    Xe_stsb = np.stack([X_eval_stsb[s] for s in keys]).astype(np.float32)
    q = load_quora_cached(model)

    def ev_q(c, qe):
        return evaluate_retrieval_ndcg10(c, qe, q["corpus_ids"], q["test_qids"], q["qrels"])

    def addret(train_ds, test_ds, method, r, val, drop=None, base=None):
        rows.append(metadata_row(
            train_dataset=train_ds, test_dataset=test_ds, model_nickname=model, hf_model_id=cfg["hf_model_id"],
            method=method, r=r, seed=str(seeds), metric_name="ndcg@10", metric_value=round(val, 4),
            drop_vs_refit=(round(drop, 4) if drop is not None else None),
            still_above_base=(None if base is None else bool(val > base)), notes=(f"base={base:.2f}" if base else "")))

    for r in dims:
        # B1. STS-B -> Quora (fit STS-B, test Quora retrieval)
        base_q = ev_q(q["corpus_np"] / (np.linalg.norm(q["corpus_np"], axis=1, keepdims=True) + 1e-12),
                      q["query_np"] / (np.linalg.norm(q["query_np"], axis=1, keepdims=True) + 1e-12))
        tr = [ev_q(proj_ncwp(q["corpus_np"], *train_ncwp_variant(X_fit_stsb, r, variant="full", seed=s)),
                   proj_ncwp(q["query_np"], *train_ncwp_variant(X_fit_stsb, r, variant="full", seed=s))) for s in seeds]
        rf = [ev_q(proj_ncwp(q["corpus_np"], *train_ncwp_variant(q["X_fit"], r, variant="full", seed=s)),
                   proj_ncwp(q["query_np"], *train_ncwp_variant(q["X_fit"], r, variant="full", seed=s))) for s in seeds]
        addret("Quora", "Quora", "NCWP_refit", r, float(np.mean(rf)), 0.0, base_q)
        addret("STS-B", "Quora", "NCWP_transfer", r, float(np.mean(tr)), float(np.mean(rf) - np.mean(tr)), base_q)
        print(f"[STS-B->Quora {model}] r={r} base={base_q:.2f} refit={np.mean(rf):.2f} transfer={np.mean(tr):.2f}")

        # B2. Quora -> STS-B (fit Quora, test STS-B intrinsic)
        base_s = sp(Xe_stsb, keys, sts["eval_pairs"], sts["eval_scores"])
        trs = [sp(proj_ncwp(Xe_stsb, *train_ncwp_variant(q["X_fit"], r, variant="full", seed=s)), keys, sts["eval_pairs"], sts["eval_scores"]) for s in seeds]
        rfs = [sp(proj_ncwp(Xe_stsb, *train_ncwp_variant(X_fit_stsb, r, variant="full", seed=s)), keys, sts["eval_pairs"], sts["eval_scores"]) for s in seeds]
        rows.append(metadata_row(train_dataset="STS-B", test_dataset="STS-B", model_nickname=model,
                    hf_model_id=cfg["hf_model_id"], method="NCWP_refit", r=r, seed=str(seeds),
                    metric_name="spearman_x100", metric_value=round(float(np.mean(rfs)), 4), drop_vs_refit=0.0,
                    still_above_base=bool(np.mean(rfs) > base_s), notes=f"base={base_s:.2f}"))
        rows.append(metadata_row(train_dataset="Quora", test_dataset="STS-B", model_nickname=model,
                    hf_model_id=cfg["hf_model_id"], method="NCWP_transfer", r=r, seed=str(seeds),
                    metric_name="spearman_x100", metric_value=round(float(np.mean(trs)), 4),
                    drop_vs_refit=round(float(np.mean(rfs) - np.mean(trs)), 4),
                    still_above_base=bool(np.mean(trs) > base_s), notes=f"base={base_s:.2f}"))
        print(f"[Quora->STS-B {model}] r={r} base={base_s:.2f} refit={np.mean(rfs):.2f} transfer={np.mean(trs):.2f}")

        # C. Quora -> CQADupStack
        for sub, mteb in CQA.items():
            cd = f"/workspace/NCWP/cqa-{sub}_results/embedding_cache/{model}"
            if not os.path.exists(f"{cd}/corpus_base.npy"):
                continue
            corpus = np.load(f"{cd}/corpus_base.npy").astype(np.float32)
            query = np.load(f"{cd}/query_base.npy").astype(np.float32)
            Xf_cqa = np.load(f"{cd}/fit_corpus_N1000.npy").astype(np.float32)
            corp_ds = load_dataset(mteb, "corpus", split="corpus")
            corpus_ids = [str(x["_id"]) for x in corp_ds]
            q_ds = load_dataset(mteb, "queries", split="queries")
            qid2t = {str(x["_id"]): x["text"] for x in q_ds}
            qr = defaultdict(dict)
            for x in load_dataset(mteb, split="test"):
                qr[str(x["query-id"])][str(x["corpus-id"])] = int(x["score"])
            tqids = sorted([x for x in qr if x in qid2t])

            def ev_c(c, qe):
                return evaluate_retrieval_ndcg10(c, qe, corpus_ids, tqids, dict(qr))

            base_c = ev_c(corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12),
                          query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12))
            trc = [ev_c(proj_ncwp(corpus, *train_ncwp_variant(q["X_fit"], r, variant="full", seed=s)),
                        proj_ncwp(query, *train_ncwp_variant(q["X_fit"], r, variant="full", seed=s))) for s in seeds]
            rfc = [ev_c(proj_ncwp(corpus, *train_ncwp_variant(Xf_cqa, r, variant="full", seed=s)),
                        proj_ncwp(query, *train_ncwp_variant(Xf_cqa, r, variant="full", seed=s))) for s in seeds]
            addret(f"cqa-{sub}", f"cqa-{sub}", "NCWP_refit", r, float(np.mean(rfc)), 0.0, base_c)
            addret("Quora", f"cqa-{sub}", "NCWP_transfer", r, float(np.mean(trc)), float(np.mean(rfc) - np.mean(trc)), base_c)
            print(f"[Quora->cqa-{sub}/{model}] r={r} base={base_c:.2f} refit={np.mean(rfc):.2f} transfer={np.mean(trc):.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=[80])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sts_targets", nargs="+", default=list(STS_TARGETS), choices=list(STS_TARGETS))
    parser.add_argument("--skip_sts", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    args = parser.parse_args()

    sts_rows, ret_rows = [], []
    for model in args.model:
        if not args.skip_sts:
            sts_rows.extend(sts_to_sts(model, args.dims, args.seeds, args.sts_targets))
        if not args.skip_retrieval:
            ret_rows.extend(cross_task_and_cqa(model, args.dims, args.seeds))
    if sts_rows:
        save_v2(sts_rows, "table_cross_dataset_transfer_sts")
    if ret_rows:
        save_v2(ret_rows, "table_cross_dataset_transfer_retrieval")
    print(f"Saved cross-dataset tables ({len(sts_rows)} sts rows, {len(ret_rows)} retrieval rows)")


if __name__ == "__main__":
    main()
