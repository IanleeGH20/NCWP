#!/usr/bin/env python3
"""Apply a trained NCWP projector to new text (inference).

Loads a projector saved by `ncwp.train`, encodes the input text with the same
frozen backbone (mean-pooled Base), applies the single linear projection + L2
normalization, and writes the compact embeddings to a .npy file.

    python -m ncwp.apply --projector projector_qwen4b_r80.npz \
        --input queries.txt --out queries_r80.npy

`--input` is a UTF-8 text file (one item per line). The output is an
(N x r) float32 array of L2-normalized NCWP embeddings, ready for cosine
retrieval. Requires HF_TOKEN for gated backbones.
"""
import argparse

import numpy as np

from ncwp.common import PromptEmbedder
from ncwp.ncwp_ref import proj_ncwp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector", required=True, help="projector .npz from ncwp.train")
    parser.add_argument("--input", required=True, help="UTF-8 text file, one item per line")
    parser.add_argument("--out", required=True, help="output .npy path (N x r)")
    args = parser.parse_args()

    d = np.load(args.projector, allow_pickle=True)
    model = str(d["model"])
    pool = str(d["pool"]) if "pool" in d else "mean"
    SW, mu_in, mu_out, std_out = d["SW"], d["mu_in"], d["mu_out"], d["std_out"]

    with open(args.input, encoding="utf-8") as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    print(f"input: {len(texts)} items | backbone: {model} | r={SW.shape[1]}")

    emb = PromptEmbedder(model, template_name="Plain", pool=pool)
    X = emb.encode(texts).astype(np.float32)
    Z = proj_ncwp(X, SW, mu_in, mu_out, std_out).astype(np.float32)

    np.save(args.out, Z)
    print(f"saved NCWP embeddings {Z.shape} to {args.out}")


if __name__ == "__main__":
    main()
