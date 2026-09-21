#!/usr/bin/env python3
"""Layer-selection table from CACHED layer embeddings (no model loading).

The encode step (PromptEmbedder with output_hidden_states) already cached every
{last, second_last, middle, early_middle} x {mean, last} embedding under
results/prompt_cache. This runner just:
  1. computes Base Spearman for all layer x pooling,
  2. runs the NCWP r-sweep on (a) last+mean (paper reference) and (b) best-Base
     combo (typically second-to-last + mean),
for all 3 backbones -> full table_layer_selection. Decoupling NCWP from the
resident 8B model avoids the CUDA-context stall seen when both share one process.
"""
import argparse
import json
import os

import numpy as np

from ncwp.common import (
    MODEL_CONFIGS,
    NUM_LAYERS,
    load_sts_benchmark,
    metadata_row,
    spearman_sts,
    subset_eval_from_cache,
)
from ncwp.ncwp_ref import fit_project

from ncwp.common import PROMPT_CACHE as CACHE_DIR
from ncwp.common import OUT_ROOT as OUT_ROOT_V2
TARGET_DIMS = {
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


def load_cached(model, lidx, pool, split, wanted_sents):
    npy = os.path.join(CACHE_DIR, f"{model}__stsb__L{lidx}_{pool}__{split}.npy")
    sj = os.path.join(CACHE_DIR, f"{model}__stsb__L{lidx}_{pool}__{split}.sents.json")
    if not (os.path.exists(npy) and os.path.exists(sj)):
        raise FileNotFoundError(
            f"Missing layer cache for {model} (layer {lidx}, {pool} pooling): {npy}\n"
            f"Build it first:  python -m ncwp.build_cache --models {model} --datasets layers"
        )
    saved = json.load(open(sj))
    arr = np.load(npy).astype(np.float32)
    if saved == wanted_sents:
        return arr
    return subset_eval_from_cache(npy, sj, wanted_sents)


def run(model, dims, seeds, eval_split="train+test"):
    cfg = MODEL_CONFIGS[model]
    sts = load_sts_benchmark(eval_split=eval_split)
    keys = sts["eval_sents"]
    L = NUM_LAYERS[model]
    layers = {"last": L, "second_last": L - 1, "middle": L // 2, "early_middle": L // 3}
    rows = []

    def add(method, lname, lidx, pool, r, val, std=None, notes=""):
        rows.append(metadata_row(
            dataset="stsb", split=eval_split, model_nickname=model, hf_model_id=cfg["hf_model_id"],
            embedding_type=f"{pool}_pool", method=method, layer_name=lname, layer_idx=lidx,
            pooling=pool, r=r, seed=str(seeds), fit_size=sts["fit_size"], eval_size=sts["eval_size"],
            metric_name="spearman_x100", metric_value=round(val, 4),
            std_if_available=(round(std, 4) if std is not None else None), notes=notes))

    cache_fit, cache_eval, best = {}, {}, (None, -1e9)
    for lname, lidx in layers.items():
        for pool in ("mean", "last"):
            Xf = load_cached(model, lidx, pool, "fit", sts["fit_sents"])
            Xe = load_cached(model, lidx, pool, "eval", keys)
            emap = {keys[i]: Xe[i] / (np.linalg.norm(Xe[i]) + 1e-12) for i in range(len(keys))}
            base = spearman_sts(emap, sts["eval_pairs"], sts["eval_scores"])
            add(f"LayerSelect_{lname}_{pool}_Base", lname, lidx, pool, cfg["hidden_dim"], base)
            print(f"[{model}] layer={lname}(L{lidx}) pool={pool} Base={base:.2f}", flush=True)
            cache_fit[(lname, pool)] = Xf
            cache_eval[(lname, pool)] = Xe
            if base > best[1]:
                best = ((lname, pool, lidx), base)

    combos = {("last", "mean", layers["last"]): "reference",
              ("second_last", "mean", layers["second_last"]): "second_last"}
    (bl, bp, bi), _ = best
    combos[(bl, bp, bi)] = "best_base" if (bl, bp) not in [("last", "mean"), ("second_last", "mean")] else combos.get((bl, bp, bi), "best_base")
    for (lname, pool, lidx), tag in combos.items():
        Xf, Xe = cache_fit[(lname, pool)], cache_eval[(lname, pool)]
        for r in dims:
            ss = []
            for s in seeds:
                Z, _ = fit_project(Xf, Xe, r, seed=s, variant="full")
                ss.append(spearman_sts({keys[i]: Z[i] for i in range(len(keys))},
                                       sts["eval_pairs"], sts["eval_scores"]))
            add(f"LayerSelect_{lname}_{pool}_NCWP", lname, lidx, pool, r,
                float(np.mean(ss)), float(np.std(ss)), notes=f"ncwp_ref ({tag})")
            print(f"[{model}] NCWP layer={lname} pool={pool} r={r} -> {np.mean(ss):.2f}±{np.std(ss):.2f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval_split", choices=["train+test", "test"], default="train+test")
    args = parser.parse_args()
    suffix = "" if args.eval_split == "train+test" else "_testonly1379"
    rows = []
    for model in args.model:
        rows.extend(run(model, TARGET_DIMS[model], args.seeds, args.eval_split))
    print(f"Saved: {save_v2(rows, f'table_layer_selection{suffix}')}")


if __name__ == "__main__":
    main()
