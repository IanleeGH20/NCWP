#!/usr/bin/env python3
"""Train an NCWP projector on your own unlabeled corpus.

Encodes a corpus with a frozen decoder-only LLM (mean-pooled Base), fits
ZCA-shrink whitening + the neighbor-contrastive projection, and saves the
projector (mu, whitening-composed projection, output stats) to a single .npz.
No labels are needed.

    python -m ncwp.train --model qwen-4b --corpus my_corpus.txt --dim 80 \
        --out projector_qwen4b_r80.npz

`--corpus` is a UTF-8 text file with one sentence/document per line. Requires
HF_TOKEN for gated backbones. Use `ncwp.apply` afterwards for inference.
"""
import argparse

import numpy as np

from ncwp.common import MODEL_CONFIGS, PromptEmbedder, set_seed
from ncwp.ncwp_ref import train_ncwp_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--corpus", required=True, help="UTF-8 text file, one item per line")
    parser.add_argument("--dim", type=int, required=True, help="target dimension r")
    parser.add_argument("--out", required=True, help="output projector .npz path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", default="mean", choices=["mean", "last"])
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    print(f"corpus: {len(texts)} items")

    emb = PromptEmbedder(args.model, template_name="Plain", pool=args.pool)
    X = emb.encode(texts).astype(np.float32)
    print(f"encoded Base: {X.shape}")

    set_seed(args.seed)
    SW, mu_in, mu_out, std_out = train_ncwp_variant(X, args.dim, variant="full", seed=args.seed)

    np.savez(
        args.out,
        SW=SW.astype(np.float32), mu_in=mu_in.astype(np.float32),
        mu_out=mu_out.astype(np.float32), std_out=std_out.astype(np.float32),
        model=args.model, dim=args.dim, pool=args.pool, seed=args.seed,
    )
    print(f"saved projector ({X.shape[1]} -> {args.dim}) to {args.out}")


if __name__ == "__main__":
    main()
