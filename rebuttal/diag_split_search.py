#!/usr/bin/env python3
"""Search which STS-B split combination reproduces the paper's Base numbers.

Table 3 states Qwen-8B Base = 37.02, but both embedding caches give 37.97
(train+test) and 39.27 (test-only). This enumerates every plausible eval-pair
subset over the cached embeddings to find which protocol, if any, yields the
published value — so the camera-ready either cites the right protocol or corrects
the number.
"""
import json
import itertools

import numpy as np

from rebuttal.common import (
    MODEL_CONFIGS,
    STS_DIR,
    load_sts_benchmark,
    load_sts_embeddings,
    spearman_sts,
)

PAPER_BASE = {"qwen-4b": 36.48, "qwen-8b": 37.02, "llama-8b": 46.35}


def main():
    raw = {name: json.load(open(f"{STS_DIR}/sts_{name}.json"))
           for name in ["train", "valid", "test"]}

    # One embedding load per model, using the widest sentence set available.
    sts_full = load_sts_benchmark(eval_split="train+test")

    for model in ["qwen-4b", "qwen-8b", "llama-8b"]:
        _, X_eval, _ = load_sts_embeddings(model, sts_full)
        emb = {s: x / (np.linalg.norm(x) + 1e-12) for s, x in X_eval.items()}
        print(f"\n=== {model} (paper Base = {PAPER_BASE[model]}) ===")

        combos = []
        for k in (1, 2, 3):
            combos += list(itertools.combinations(["train", "valid", "test"], k))
        for combo in combos:
            items = [it for name in combo for it in raw[name]]
            pairs = [(it["sentence1"], it["sentence2"]) for it in items]
            scores = [it["score"] for it in items]
            missing = sum(1 for p in pairs for s in p if s not in emb)
            if missing:
                print(f"  {'+'.join(combo):18s} pairs={len(pairs):5d} "
                      f"-> skipped ({missing} sentences not in cache)")
                continue
            val = spearman_sts(emb, pairs, scores)
            flag = "  <-- MATCHES PAPER" if abs(val - PAPER_BASE[model]) < 0.1 else ""
            print(f"  {'+'.join(combo):18s} pairs={len(pairs):5d} Base={val:.2f}{flag}")


if __name__ == "__main__":
    main()
