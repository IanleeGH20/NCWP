#!/usr/bin/env python3
"""Diagnose the Qwen-8B Base mismatch (paper 37.02 vs measured 37.96 / 39.24).

Table 3 of the paper states Base = 37.02 for Qwen-8B, but the rebuttal runs report
37.96 (train+test 7128 pairs) and 39.24 (test-only 1379 pairs). This prints Base
for every combination of eval split and embedding source so the camera-ready can
cite the number that matches its stated protocol.

Sources compared:
  main_cache   sts_ablation_cache/{model}_main_embs.npy  (used by all rebuttal runs)
  layer_cache  rebuttal_outputs_v2/prompt_cache L{last}_mean (layer-selection run)
"""
import json
import os

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    load_sts_benchmark,
    load_sts_embeddings,
    spearman_sts,
    subset_eval_from_cache,
)

CACHE_DIR = "/workspace/NCWP/rebuttal_outputs_v2/prompt_cache"
NUM_LAYERS = {"qwen-4b": 40, "qwen-8b": 28, "llama-8b": 32}


def base_score(X_eval_map, sts):
    m = {s: x / (np.linalg.norm(x) + 1e-12) for s, x in X_eval_map.items()}
    return spearman_sts(m, sts["eval_pairs"], sts["eval_scores"])


def main():
    for model in ["qwen-4b", "qwen-8b", "llama-8b"]:
        d = MODEL_CONFIGS[model]["hidden_dim"]
        lidx = NUM_LAYERS[model]
        print(f"\n=== {model} (D={d}) ===")
        for split in ["train+test", "test"]:
            sts = load_sts_benchmark(eval_split=split)
            n_pairs, n_sents = len(sts["eval_pairs"]), len(sts["eval_sents"])

            _, X_eval, _ = load_sts_embeddings(model, sts)
            s_main = base_score(X_eval, sts)

            npy = os.path.join(CACHE_DIR, f"{model}__stsb__L{lidx}_mean__eval.npy")
            sj = os.path.join(CACHE_DIR, f"{model}__stsb__L{lidx}_mean__eval.sents.json")
            if os.path.exists(npy):
                saved = json.load(open(sj))
                keys = sts["eval_sents"]
                arr = (np.load(npy).astype(np.float32) if saved == keys
                       else subset_eval_from_cache(npy, sj, keys))
                s_layer = f"{base_score({keys[i]: arr[i] for i in range(len(keys))}, sts):.2f}"
            else:
                s_layer = "n/a"

            print(f"  split={split:11s} pairs={n_pairs:5d} sents={n_sents:5d} "
                  f"main_cache={s_main:.2f}  layer_cache(L{lidx}_mean)={s_layer}")


if __name__ == "__main__":
    main()
