#!/usr/bin/env python3
"""v2 spec §3 (Review 3 W2): layer-selection baseline.

Is the last layer actually best for STS, and does NCWP's gain depend on the
layer? Extract hidden states at {last, second-last, middle, early-middle} x
{mean, last-token} pooling, report Base Spearman for all, and run the NCWP r-sweep
on (a) last-layer+mean (paper's reference) and (b) the best-Base layer/pooling
combo. STS-B. Encodings cached. NCWP via ncwp_ref. New -> seeds 0,1,2.
MODEL-LOAD track. Output: rebuttal_outputs_v2/table_layer_selection.
"""
import argparse
import json
import os

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    PromptEmbedder,
    load_sts_benchmark,
    metadata_row,
    spearman_sts,
)
from rebuttal.ncwp_ref import fit_project

CACHE_DIR = "/workspace/NCWP/rebuttal_outputs_v2/prompt_cache"
OUT_ROOT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"
REBUTTAL_DIMS = {
    "qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448], "llama-8b": [32, 64, 128, 256],
}


def save_v2(rows, basename):
    import pandas as pd
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT_V2, f"{basename}.csv"), index=False)
    with open(os.path.join(OUT_ROOT_V2, f"{basename}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return os.path.join(OUT_ROOT_V2, f"{basename}.csv")


def encode_cached(embedder, model, tag, split, sents):
    os.makedirs(CACHE_DIR, exist_ok=True)
    npy = os.path.join(CACHE_DIR, f"{model}__stsb__{tag}__{split}.npy")
    sj = os.path.join(CACHE_DIR, f"{model}__stsb__{tag}__{split}.sents.json")
    if os.path.exists(npy) and os.path.exists(sj) and json.load(open(sj)) == sents:
        return np.load(npy).astype(np.float32)
    arr = embedder.encode(sents).astype(np.float32)
    np.save(npy, arr); json.dump(sents, open(sj, "w"))
    return arr


def run(model, dims, seeds):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark()
    keys = sts["eval_sents"]
    emb = PromptEmbedder(model, template_name="Plain", pool="mean")
    L = emb.num_layers
    layers = {"last": L, "second_last": L - 1, "middle": L // 2, "early_middle": L // 3}
    rows = []

    def add(method, layer_name, layer_idx, pool, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset="stsb", split="train+test", model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type=f"{pool}_pool", method=method, layer_name=layer_name, layer_idx=layer_idx,
            pooling=pool, r=r, seed=str(seeds), fit_size=sts["fit_size"], eval_size=sts["eval_size"],
            metric_name="spearman_x100", metric_value=round(val, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    # Base for all layer x pooling; remember embeddings + best combo
    cache_fit, cache_eval, best = {}, {}, (None, -1e9)
    for lname, lidx in layers.items():
        emb.layer = lidx
        for pool in ("mean", "last"):
            emb.pool = pool
            Xf = encode_cached(emb, model, f"L{lidx}_{pool}", "fit", sts["fit_sents"])
            Xe = encode_cached(emb, model, f"L{lidx}_{pool}", "eval", keys)
            emap = {keys[i]: Xe[i] / (np.linalg.norm(Xe[i]) + 1e-12) for i in range(len(keys))}
            base = spearman_sts(emap, sts["eval_pairs"], sts["eval_scores"])
            add(f"LayerSelect_{lname}_{pool}_Base", lname, lidx, pool, cfg["hidden_dim"], base)
            print(f"[{model}] layer={lname}(L{lidx}) pool={pool} Base={base:.2f}")
            cache_fit[(lname, pool)] = Xf; cache_eval[(lname, pool)] = Xe
            if base > best[1]:
                best = ((lname, pool, lidx), base)

    # NCWP r-sweep on last-layer+mean (reference) and best-Base combo
    combos = {("last", "mean", layers["last"]): "reference"}
    (bl, bp, bi), _ = best
    combos[(bl, bp, bi)] = "best_base"
    for (lname, pool, lidx), tag in combos.items():
        Xf, Xe = cache_fit[(lname, pool)], cache_eval[(lname, pool)]
        for r in dims:
            ss = [spearman_sts({keys[i]: fit_project(Xf, Xe, r, seed=s, variant="full")[0][i]
                                for i in range(len(keys))}, sts["eval_pairs"], sts["eval_scores"]) for s in seeds]
            add(f"LayerSelect_{lname}_{pool}_NCWP", lname, lidx, pool, r,
                float(np.mean(ss)), float(np.std(ss)), notes=f"ncwp_ref ({tag})")
            print(f"[{model}] NCWP layer={lname} pool={pool} r={r} -> {np.mean(ss):.2f}")
    del emb
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b"], choices=list(MODEL_CONFIGS))
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    rows = []
    for model in args.model:
        dims = args.dims if args.dims is not None else REBUTTAL_DIMS[model]
        rows.extend(run(model, dims, args.seeds))
    basename = "table_layer_selection" if len(args.model) > 1 else f"table_layer_selection_{args.model[0]}"
    print(f"Saved: {save_v2(rows, basename)}")


if __name__ == "__main__":
    main()
