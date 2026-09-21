#!/usr/bin/env python3
"""Anisotropy diagnostics (paper Table 3 + Appendix per-r tables).

Self-contained: encodes the 2,910 STSBenchmark validation sentences (mean-pooled,
L2-normalized Base), measures anisotropy_mean and self_sim for the raw Base and
for NCWP at every target dimension. NCWP projections are trained on the fly with
the reference trainer (no external weight files). Run from the repo root:

    python run_t2_anisotropy.py --models qwen-4b qwen-8b llama-8b
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from ncwp.common import CACHE_DIR, STS_DIR, OUT_ROOT, ensure_dir
from ncwp.ncwp_ref import train_ncwp_variant, proj_ncwp

STS_VALID = os.path.join(STS_DIR, "sts_valid.json")
OUT_CSV = os.path.join(OUT_ROOT, "t2_anisotropy_results.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MODELS = {
    "qwen-4b":  {"hf": "Qwen/Qwen1.5-4B",              "base_dim": 2560,
                 "dims": [5, 10, 20, 40, 80, 160, 320, 640, 1280]},
    "qwen-8b":  {"hf": "Qwen/Qwen2-7B",                "base_dim": 3584,
                 "dims": [7, 14, 28, 56, 112, 224, 448, 896, 1792]},
    "llama-8b": {"hf": "meta-llama/Meta-Llama-3.1-8B", "base_dim": 4096,
                 "dims": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
}

np.random.seed(SEED)


def encode_sentences(sentences, model_key):
    cfg = MODELS[model_key]
    print(f"  loading model: {cfg['hf']}")
    tok = AutoTokenizer.from_pretrained(cfg['hf'], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(cfg['hf'], trust_remote_code=True, torch_dtype=dtype).to(DEVICE)
    model.eval()
    parts = []
    for i in range(0, len(sentences), 32):
        batch = sentences[i:i + 32]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        parts.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(parts)


def compute_anisotropy(embs_np):
    embs = embs_np / (np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-12)
    mu_hat = embs.mean(0)
    mu_norm = mu_hat / (np.linalg.norm(mu_hat) + 1e-12)
    aniso_mean = float(np.mean(embs @ mu_norm))
    rng = np.random.RandomState(SEED)
    n = len(embs)
    i_idx = rng.randint(0, n, 5000)
    j_idx = rng.randint(0, n, 5000)
    m = i_idx != j_idx
    i_idx, j_idx = i_idx[m], j_idx[m]
    self_sim = float(np.mean(np.einsum('id,id->i', embs[i_idx], embs[j_idx])))
    return aniso_mean, self_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    with open(STS_VALID) as f:
        valid_data = json.load(f)
    fit_sents = list({s for item in valid_data for s in (item['sentence1'], item['sentence2'])})
    print(f"STSBenchmark validation unique sentences: {len(fit_sents)}")

    ensure_dir(CACHE_DIR)
    ensure_dir(OUT_ROOT)
    results = []

    for model_key in args.models:
        cfg = MODELS[model_key]
        print(f"\n{'=' * 55}\nModel: {model_key} (base_dim={cfg['base_dim']})\n{'=' * 55}")
        cache_emb = os.path.join(CACHE_DIR, f"{model_key}_valid_embs.npy")
        cache_sent = os.path.join(CACHE_DIR, f"{model_key}_valid_sents.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_sent) and json.load(open(cache_sent)) == fit_sents:
            embs_full = np.load(cache_emb).astype(np.float32)
        else:
            embs_full = encode_sentences(fit_sents, model_key)
            np.save(cache_emb, embs_full)
            json.dump(fit_sents, open(cache_sent, "w"))
        print(f"  embs_full shape: {embs_full.shape}")

        aniso_b, self_b = compute_anisotropy(embs_full)
        print(f"  [Base dim={cfg['base_dim']}] anisotropy_mean={aniso_b:.4f}  self_sim={self_b:.4f}")
        results.append({"model": model_key, "method": "Base", "dim": cfg['base_dim'],
                        "anisotropy_mean": round(aniso_b, 4), "self_sim": round(self_b, 4)})

        for r in cfg['dims']:
            SW, mu_in, mu_out, std_out = train_ncwp_variant(embs_full, r, variant="full", seed=SEED)
            embs_proj = proj_ncwp(embs_full, SW, mu_in, mu_out, std_out)
            aniso_p, self_p = compute_anisotropy(embs_proj)
            print(f"  [NCWP r={r:>4}] anisotropy_mean={aniso_p:.4f}  self_sim={self_p:.4f}")
            results.append({"model": model_key, "method": "NCWP", "dim": r,
                            "anisotropy_mean": round(aniso_p, 4), "self_sim": round(self_p, 4)})

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
