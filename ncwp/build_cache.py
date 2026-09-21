#!/usr/bin/env python3
"""Build the Base embedding caches the experiment runners read.

Each backbone's frozen hidden states are extracted once (last layer, mean-pooled
over valid tokens, L2-normalized, bf16, max_len 128 — the paper's "Base"), and
cached to disk so that the fitting / evaluation runners never re-encode. Run this
before the STS / Quora / CQADupStack experiments.

    # STS only (fast: ~15k sentences per backbone)
    python -m ncwp.build_cache --models qwen-4b --datasets sts

    # STS + BEIR-Quora (Quora corpus = 522,931 items; slow, needs disk + time)
    python -m ncwp.build_cache --models qwen-4b qwen-8b llama-8b --datasets sts quora

    # per-layer x per-pooling STS caches (needed by the layer-selection analysis)
    python -m ncwp.build_cache --models qwen-4b qwen-8b llama-8b --datasets layers

    # add the optional CQADupStack breadth caches
    python -m ncwp.build_cache --models qwen-4b llama-8b --datasets cqa

Requires HF_TOKEN in the environment for the gated Llama / Qwen checkpoints.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from ncwp.common import (
    CACHE_DIR,
    CQA_CACHE,
    DEVICE,
    MODEL_CONFIGS,
    NUM_LAYERS,
    PROMPT_CACHE,
    PromptEmbedder,
    QUORA_CACHE,
    STS_DIR,
    ensure_dir,
    load_sts_benchmark,
)

CQA_SUBFORUMS = {
    "english": "mteb/cqadupstack-english",
    "gaming": "mteb/cqadupstack-gaming",
    "physics": "mteb/cqadupstack-physics",
}
CQA_FIT_N = 1000


def base_encoder(model: str) -> PromptEmbedder:
    """Plain passthrough template + mean pooling == the paper's Base extraction."""
    return PromptEmbedder(model, template_name="Plain", pool="mean")


def build_sts(model: str, emb: PromptEmbedder) -> None:
    ensure_dir(CACHE_DIR)
    sents = set()
    for split in ("valid", "train", "test"):
        for it in json.load(open(os.path.join(STS_DIR, f"sts_{split}.json"))):
            sents.add(it["sentence1"])
            sents.add(it["sentence2"])
    sents = sorted(sents)
    arr = emb.encode(sents).astype(np.float32)
    np.save(os.path.join(CACHE_DIR, f"{model}_main_embs.npy"), arr)
    json.dump(sents, open(os.path.join(CACHE_DIR, f"{model}_main_sents.json"), "w"))
    print(f"[STS/{model}] {arr.shape} -> {CACHE_DIR}")


def _encode_to_npy(emb: PromptEmbedder, texts, path: str, dim: int, chunk: int = 20000) -> None:
    """Encode into a preallocated .npy on disk.

    Writing chunk-by-chunk keeps peak memory at O(chunk x dim) instead of
    materializing the whole corpus (522k x 2560 floats is ~5 GB per copy).
    """
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(texts), dim))
    for start in range(0, len(texts), chunk):
        block = texts[start:start + chunk]
        arr[start:start + len(block)] = emb.encode(block).astype(np.float32)
        arr.flush()
        print(f"    encoded {min(start + len(block), len(texts))}/{len(texts)}", flush=True)
    del arr


def build_quora(model: str, emb: PromptEmbedder) -> None:
    out = os.path.join(QUORA_CACHE, model)
    ensure_dir(out)
    dim = MODEL_CONFIGS[model]["hidden_dim"]
    corp = load_dataset("mteb/quora", "corpus", split="corpus")
    _encode_to_npy(emb, [r["text"] for r in corp], f"{out}/corpus_base.npy", dim)
    qs = load_dataset("mteb/quora", "queries", split="queries")
    qid2tx = {str(r["_id"]): r["text"] for r in qs}
    qrel = load_dataset("mteb/quora", split="test")
    qrels = defaultdict(set)
    for r in qrel:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    test_qids = sorted(q for q in qrels if q in qid2tx)
    np.save(f"{out}/test_queries_base.npy",
            emb.encode([qid2tx[q] for q in test_qids]).astype(np.float32))
    # fit_query_sample: deterministic query subsample (seed 42) used by appendix runners
    all_qids = sorted(qid2tx)
    n = MODEL_CONFIGS[model]["quora_fit_size"]
    idx = np.random.RandomState(42).choice(len(all_qids), min(n, len(all_qids)), replace=False)
    np.save(f"{out}/fit_query_sample_N{n}.npy",
            emb.encode([qid2tx[all_qids[i]] for i in idx]).astype(np.float32))
    print(f"[Quora/{model}] corpus+queries+fitN{n} -> {out}")


@torch.no_grad()
def _encode_multi_layer(emb: PromptEmbedder, sents, layer_idxs, pools, batch_size: int = 64):
    """Encode `sents` once per batch and read out every (layer, pooling) pair.

    Matches the Base recipe (raw sentence, max_len 128, bf16, L2-normalized) but
    reads intermediate `hidden_states` instead of only the last layer.
    """
    out = {(l, p): [] for l in layer_idxs for p in pools}
    dev = getattr(emb.model, "device", DEVICE)
    for i in range(0, len(sents), batch_size):
        enc = emb.tok(list(sents[i:i + batch_size]), padding=True, truncation=True,
                      max_length=emb.max_length, return_tensors="pt").to(dev)
        hs = emb.model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"]
        for l in layer_idxs:
            h = hs[l]
            for p in pools:
                if p == "last":
                    x = h[torch.arange(h.size(0), device=h.device), mask.sum(1) - 1]
                else:
                    m = mask.unsqueeze(-1).float()
                    x = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
                out[(l, p)].append(F.normalize(x, p=2, dim=1).cpu().float().numpy())
    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in out.items()}


def build_layers(model: str, emb: PromptEmbedder) -> None:
    """Per-layer x per-pooling STS caches for the layer-selection analysis.

    Cached on the train+test sentence superset, so the official-test subset is
    derived without re-encoding.
    """
    ensure_dir(PROMPT_CACHE)
    n_layers = NUM_LAYERS[model]
    layer_idxs = sorted({n_layers, n_layers - 1, n_layers // 2, n_layers // 3})
    pools = ("mean", "last")
    sts = load_sts_benchmark(eval_split="train+test")
    for split, sents in (("fit", sts["fit_sents"]), ("eval", sts["eval_sents"])):
        arrs = _encode_multi_layer(emb, sents, layer_idxs, pools)
        for (l, p), arr in arrs.items():
            stem = os.path.join(PROMPT_CACHE, f"{model}__stsb__L{l}_{p}__{split}")
            np.save(f"{stem}.npy", arr)
            json.dump(sents, open(f"{stem}.sents.json", "w"))
        print(f"[layers/{model}] {split}: {len(sents)} sents x layers {layer_idxs} x {pools}")
    print(f"[layers/{model}] -> {PROMPT_CACHE}")


def build_cqa(model: str, emb: PromptEmbedder) -> None:
    dim = MODEL_CONFIGS[model]["hidden_dim"]
    for sub, name in CQA_SUBFORUMS.items():
        out = os.path.join(CQA_CACHE, sub, model)
        ensure_dir(out)
        corp = load_dataset(name, "corpus", split="corpus")
        # BEIR convention: a CQADupStack document is its title plus its body.
        docs = [f"{r.get('title', '')} {r['text']}".strip() for r in corp]
        _encode_to_npy(emb, docs, f"{out}/corpus_base.npy", dim)
        # Queries must be stored in the same order the runners index them:
        # the sorted test query ids that have both a qrel and a text.
        qs = load_dataset(name, "queries", split="queries")
        qid2text = {str(r["_id"]): r["text"] for r in qs}
        qrels = defaultdict(dict)
        for r in load_dataset(name, split="test"):
            qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
        test_qids = sorted(q for q in qrels if q in qid2text)
        np.save(f"{out}/query_base.npy",
                emb.encode([qid2text[q] for q in test_qids]).astype(np.float32))
        corpus = np.load(f"{out}/corpus_base.npy", mmap_mode="r")
        idx = np.random.RandomState(42).choice(len(corpus), min(CQA_FIT_N, len(corpus)), replace=False)
        np.save(f"{out}/fit_corpus_N{CQA_FIT_N}.npy", np.asarray(corpus[idx], dtype=np.float32))
        print(f"[CQA/{sub}/{model}] {corpus.shape} -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-4b", "qwen-8b", "llama-8b"],
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--datasets", nargs="+", default=["sts", "quora"],
                        choices=["sts", "quora", "cqa", "layers"])
    args = parser.parse_args()

    builders = {"sts": build_sts, "quora": build_quora, "cqa": build_cqa, "layers": build_layers}
    for model in args.models:
        emb = None
        for ds in args.datasets:
            if emb is None:
                emb = base_encoder(model)  # load the backbone once per model
            builders[ds](model, emb)
        del emb


if __name__ == "__main__":
    main()
