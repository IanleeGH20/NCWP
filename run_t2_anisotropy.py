"""
T2: LLM Backbone Anisotropy 측정
─────────────────────────────────────────────────────────────
3개 LLM backbone (Llama-8B, Qwen-4B, Qwen-8B)에서
STSBenchmark validation (2,910 sentences) 기준 anisotropy 측정.

Diagnostic:
  - anisotropy_mean = mean_i cos(x_i, μ_hat)     where μ_hat = mean(embs)
  - self_sim       = mean_{i≠j} cos(x_i, x_j)    (5000 random pairs)

Method:
  - Base: raw embedding → L2 norm
  - NCWP: 학습된 W 적용 → L2 norm

NCWP weight: /workspace/RAG/code/Make_embedding/nanoGPT/ncwp_extended_results/{model}_ncwp_dim{r}.npz
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

CACHE_DIR    = "/workspace/NCWP/sts_ablation_cache"
WEIGHT_DIR   = "/workspace/RAG/code/Make_embedding/nanoGPT/ncwp_extended_results"
STS_VALID    = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark/sts_valid.json"
OUT_CSV      = "/workspace/NCWP/t2_anisotropy_results.csv"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

MODELS = {
    "qwen-4b":  {"hf": "Qwen/Qwen1.5-4B",                 "base_dim": 2560,
                 "dims": [5, 10, 20, 40, 80, 160, 320, 640, 1280]},
    "qwen-8b":  {"hf": "Qwen/Qwen2-7B",                   "base_dim": 3584,
                 "dims": [7, 14, 28, 56, 112, 224, 448, 896, 1792]},
    "llama-8b": {"hf": "meta-llama/Meta-Llama-3.1-8B",    "base_dim": 4096,
                 "dims": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
}

np.random.seed(SEED)

# ── validation 문장 로드 ─────────────────────────────────────
print("STSBenchmark validation 로드...")
with open(STS_VALID) as f:
    valid_data = json.load(f)
fit_sents = list({s for item in valid_data
                  for s in (item['sentence1'], item['sentence2'])})
print(f"  valid unique sentences: {len(fit_sents)}")


def encode_sentences(sentences, model_key):
    """모델 로드 + 임베딩"""
    cfg = MODELS[model_key]
    print(f"  모델 로드: {cfg['hf']}")
    tok = AutoTokenizer.from_pretrained(cfg['hf'], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True,
          "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    n_gpu = torch.cuda.device_count()
    model = (torch.nn.DataParallel(AutoModel.from_pretrained(cfg['hf'], **kw)).cuda()
             if n_gpu > 1 else
             AutoModel.from_pretrained(cfg['hf'], **kw).to(DEVICE))
    model.eval()
    parts = []
    for i in range(0, len(sentences), 32):
        batch = sentences[i:i+32]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc) if n_gpu <= 1 else model.module(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        parts.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        if (i // 32) % 20 == 0:
            print(f"    {i+len(batch)}/{len(sentences)}", end="\r")
    print()
    del model
    torch.cuda.empty_cache()
    return np.vstack(parts)


def compute_anisotropy(embs_np):
    """L2-normalized embeddings에 대한 두 diagnostic 계산"""
    # 1. L2-normalize
    embs = embs_np / (np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-12)
    
    # 2. anisotropy_mean: mean cosine to mu_hat
    mu_hat = embs.mean(0)
    mu_norm = mu_hat / (np.linalg.norm(mu_hat) + 1e-12)
    aniso_mean = float(np.mean(embs @ mu_norm))
    
    # 3. self_sim: 5000 random pair (i≠j)
    rng = np.random.RandomState(SEED)
    N = len(embs)
    i_idx = rng.randint(0, N, 5000)
    j_idx = rng.randint(0, N, 5000)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]; j_idx = j_idx[mask]
    sims = np.einsum('id,id->i', embs[i_idx], embs[j_idx])
    self_sim = float(np.mean(sims))
    
    return aniso_mean, self_sim


def apply_ncwp(X, W, mu_in, mu_out, std_out):
    """NCWP transform"""
    # ncwp_eval/code/projector.py 표준 inference 절차:
    # 1. Y = (X - mu_in) @ W
    # 2. (optional) Y = (Y - mu_out) / std_out
    # 3. L2 normalize
    Y = (X - mu_in) @ W
    # NOTE: mu_out/std_out는 학습 시 출력 정규화 통계. 보통 inference에서는 적용하지 않음
    #       (paper eq. 10: g(x) = norm((x-mu_in)·S·W))
    # L2 normalize은 compute_anisotropy 내부에서 처리
    return Y


# ── 모델별 실험 ──────────────────────────────────────────────
results = []

for model_key, cfg in MODELS.items():
    print(f"\n{'='*55}")
    print(f"Model: {model_key} (base_dim={cfg['base_dim']})")
    print(f"{'='*55}")
    
    # 임베딩 (캐시 사용 가능 시)
    cache_emb  = os.path.join(CACHE_DIR, f"{model_key}_valid_embs.npy")
    cache_sent = os.path.join(CACHE_DIR, f"{model_key}_valid_sents.json")
    
    if os.path.exists(cache_emb) and os.path.exists(cache_sent):
        print("  validation 임베딩 캐시 로드...")
        with open(cache_sent) as f:
            cached = json.load(f)
        if cached == fit_sents:
            embs_full = np.load(cache_emb).astype(np.float32)
        else:
            print("  캐시 문장 불일치 → 재인코딩")
            embs_full = encode_sentences(fit_sents, model_key)
            np.save(cache_emb, embs_full)
            with open(cache_sent, "w") as f: json.dump(fit_sents, f)
    elif model_key == "qwen-4b":
        # 기존 ablation 캐시(전체 5385 문장)에서 valid 부분만 추출
        print("  기존 sts_ablation_cache에서 valid 부분 추출...")
        all_embs = np.load(os.path.join(CACHE_DIR, "qwen-4b_all_embs.npy"))
        with open(os.path.join(CACHE_DIR, "qwen-4b_sent_order.json")) as f:
            saved_sents = json.load(f)
        sent2idx = {s: i for i, s in enumerate(saved_sents)}
        embs_full = np.array([all_embs[sent2idx[s]] for s in fit_sents]).astype(np.float32)
        np.save(cache_emb, embs_full)
        with open(cache_sent, "w") as f: json.dump(fit_sents, f)
    else:
        print(f"  validation 임베딩 생성 ({len(fit_sents)}개)...")
        embs_full = encode_sentences(fit_sents, model_key)
        np.save(cache_emb, embs_full)
        with open(cache_sent, "w") as f: json.dump(fit_sents, f)
    
    print(f"  embs_full shape: {embs_full.shape}")
    
    # Base 측정
    aniso_b, self_b = compute_anisotropy(embs_full)
    print(f"  [Base dim={cfg['base_dim']}] anisotropy_mean={aniso_b:.4f}  self_sim={self_b:.4f}")
    results.append({
        "model": model_key,
        "method": "Base",
        "dim": cfg['base_dim'],
        "anisotropy_mean": round(aniso_b, 4),
        "self_sim": round(self_b, 4),
    })
    
    # NCWP 각 dim 측정
    for r in cfg['dims']:
        wpath = os.path.join(WEIGHT_DIR, f"{model_key}_ncwp_dim{r}.npz")
        if not os.path.exists(wpath):
            print(f"  [NCWP r={r}] weight 없음, 스킵")
            continue
        d = np.load(wpath)
        W = d['W'].astype(np.float32)
        mu_in  = d['mu_in'].astype(np.float32)
        embs_proj = apply_ncwp(embs_full, W, mu_in, None, None)
        aniso_p, self_p = compute_anisotropy(embs_proj)
        print(f"  [NCWP r={r:>4}] anisotropy_mean={aniso_p:.4f}  self_sim={self_p:.4f}")
        results.append({
            "model": model_key,
            "method": "NCWP",
            "dim": r,
            "anisotropy_mean": round(aniso_p, 4),
            "self_sim": round(self_p, 4),
        })

# 저장
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\n결과 저장: {OUT_CSV}")

# ── 표 출력 ───────────────────────────────────────────────────
print("\n" + "="*70)
print("T2: LLM Backbone Anisotropy on STSBenchmark validation (N=2910)")
print("="*70)

for model_key in MODELS.keys():
    sub = df[df['model']==model_key]
    base = sub[sub['method']=='Base'].iloc[0]
    print(f"\n[ {model_key} ]  Base(dim={int(base['dim'])}): "
          f"aniso_mean={base['anisotropy_mean']:.4f}  self_sim={base['self_sim']:.4f}")
    ncwp = sub[sub['method']=='NCWP'].sort_values('dim', ascending=False)
    print(f"  {'r':>5}  {'aniso_mean':>11}  {'self_sim':>10}")
    for _, row in ncwp.iterrows():
        print(f"  {int(row['dim']):>5}  {row['anisotropy_mean']:>11.4f}  {row['self_sim']:>10.4f}")
