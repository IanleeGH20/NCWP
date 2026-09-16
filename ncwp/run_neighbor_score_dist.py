#!/usr/bin/env python3
"""v2 spec §5 (Review 3 W3): STS score distribution among mined neighbors.

Directly shows pseudo-positive quality: for each sentence, take top-k neighbors
in each space {raw mean-pool, ZCA-whitened, NCWP-projected} plus a random-pair
baseline, and look up the STS gold score for any mined pair that is annotated.
Report the distribution of those gold scores per space — a good space should mine
neighbors with high STS scores.

Cached-embedding experiment (no model loading). New analysis -> seed 0 for the
NCWP projector unless overridden. Outputs to results/.
"""
import argparse
import json
import os

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    apply_zca,
    compute_knn,
    compute_zca,
    load_sts_benchmark,
    load_sts_embeddings,
    metadata_row,
)
from ncwp.ncwp_ref import fit_project

OUT_ROOT_V2 = "/workspace/NCWP/results"


def save_v2(rows, basename):
    import pandas as pd
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT_V2, f"{basename}.csv")
    jsonl_path = os.path.join(OUT_ROOT_V2, f"{basename}.jsonl")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def gold_lookup(sts):
    """(sentence_a, sentence_b) -> gold score, symmetric."""
    g = {}
    for (s1, s2), sc in zip(sts["eval_pairs"], sts["eval_scores"]):
        g[(s1, s2)] = sc
        g[(s2, s1)] = sc
    return g


def collect_scores(emb_mat, sents, gold, k=10, tau=0.0):
    """kNN in the given (already-normalized) space; return gold scores of mined pairs."""
    knn_idx, knn_sims = compute_knn(emb_mat.astype(np.float32), k=k)
    scores = []
    for i in range(len(sents)):
        for t, j in enumerate(knn_idx[i]):
            if knn_sims[i, t] < tau:
                continue
            sc = gold.get((sents[i], sents[int(j)]))
            if sc is not None:
                scores.append(float(sc))
    return np.array(scores, dtype=np.float32)


def random_pair_scores(sents, gold, n_pairs, seed=0):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(sents)
    tries = 0
    while len(scores) < n_pairs and tries < n_pairs * 200:
        i, j = rng.randint(0, n), rng.randint(0, n)
        tries += 1
        if i == j:
            continue
        sc = gold.get((sents[i], sents[j]))
        if sc is not None:
            scores.append(float(sc))
    return np.array(scores, dtype=np.float32)


def dist_stats(scores):
    if len(scores) == 0:
        return dict(mean_score=float("nan"), median=float("nan"), p25=float("nan"),
                    p75=float("nan"), pct_ge_4_0=float("nan"), pct_ge_4_5=float("nan"),
                    matched_pairs=0)
    return dict(
        mean_score=round(float(np.mean(scores)), 4),
        median=round(float(np.median(scores)), 4),
        p25=round(float(np.percentile(scores, 25)), 4),
        p75=round(float(np.percentile(scores, 75)), 4),
        pct_ge_4_0=round(float(np.mean(scores >= 4.0) * 100), 4),
        pct_ge_4_5=round(float(np.mean(scores >= 4.5) * 100), 4),
        matched_pairs=int(len(scores)),
    )


def run(model, k, tau, r_ncwp, seed, dump_raw):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark()
    X_fit, X_eval, _ = load_sts_embeddings(model, sts)
    gold = gold_lookup(sts)
    sents = list(X_eval)
    raw_mat = np.stack([X_eval[s] for s in sents]).astype(np.float32)
    raw_mat = raw_mat / (np.linalg.norm(raw_mat, axis=1, keepdims=True) + 1e-12)

    mu_zca, s_zca = compute_zca(X_fit)
    zca_mat = np.stack([apply_zca(X_eval[s].reshape(1, -1), mu_zca, s_zca)[0] for s in sents]).astype(np.float32)

    # NCWP-projected space via the faithful trainer (whitens internally)
    raw_eval_mat = np.stack([X_eval[s] for s in sents]).astype(np.float32)
    ncwp_mat, _ = fit_project(X_fit.astype(np.float32), raw_eval_mat, r_ncwp, seed=seed, variant="full")

    spaces = {
        "Raw": collect_scores(raw_mat, sents, gold, k, tau),
        "ZCA-whitened": collect_scores(zca_mat, sents, gold, k, tau),
        "NCWP-projected": collect_scores(ncwp_mat, sents, gold, k, tau),
    }
    n_ref = max(len(v) for v in spaces.values()) or 1000
    spaces["Random"] = random_pair_scores(sents, gold, n_ref, seed)

    rows, raw_dump = [], {}
    for space, sc in spaces.items():
        st = dist_stats(sc)
        rows.append(metadata_row(
            dataset="stsb", split="train+test", model_nickname=model,
            hf_model_id=cfg["hf_model_id"], embedding_type="mean_pool", method="mined_neighbor_score_dist",
            space=space, k=(k if space != "Random" else None), tau=(tau if space != "Random" else None),
            r=(r_ncwp if space == "NCWP-projected" else None), seed=seed,
            fit_size=sts["fit_size"], eval_size=len(sents),
            metric_name="mean_gold_sts_score", metric_value=st["mean_score"],
            median=st["median"], p25=st["p25"], p75=st["p75"],
            pct_ge_4_0=st["pct_ge_4_0"], pct_ge_4_5=st["pct_ge_4_5"], matched_pairs=st["matched_pairs"],
            notes=f"k={k} tau={tau}" + (f" r_ncwp={r_ncwp}" if space == "NCWP-projected" else ""),
        ))
        raw_dump[space] = sc.tolist()
        print(f"[{model}] {space}: mean={st['mean_score']} median={st['median']} "
              f">=4.0={st['pct_ge_4_0']}% n={st['matched_pairs']}")

    if dump_raw:
        os.makedirs(OUT_ROOT_V2, exist_ok=True)
        with open(os.path.join(OUT_ROOT_V2, f"mined_neighbor_scores_raw_{model}.json"), "w") as f:
            json.dump(raw_dump, f)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--r_ncwp", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump_raw", action="store_true", help="dump raw score arrays for plotting")
    args = parser.parse_args()

    rows = []
    for model in args.model:
        rows.extend(run(model, args.k, args.tau, args.r_ncwp, args.seed, args.dump_raw))
    csv_path, jsonl_path = save_v2(rows, "table_sts_mined_neighbor_score_distribution")
    print(f"Saved: {csv_path}\n       {jsonl_path}")


if __name__ == "__main__":
    main()
