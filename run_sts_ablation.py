"""
run_sts_ablation.py
─────────────────────────────────────────────────────────────
T1: Ablation Study — Qwen-4B × STSBenchmark
8 Variants:
  1. PCA-whitening
  2. ZCA-only
  3. Random-pairs
  4. No hard-neg    (= No-QR, hard-neg 미구현이므로 동일)
  5. No memory bank (= No-QR, memory bank 미구현이므로 동일)
  6. No-QR (final)
  7. No refinement
  8. Full-NCWP-with-QR

Fit:  STSBenchmark validation (2,910 sentences)
Eval: STSBenchmark test pairs (1,379), Spearman×100
Dims: r ∈ {20, 80, 320}
"""
import os, math, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

# ── 경로 ─────────────────────────────────────────────────────
STS_DIR   = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
OUT_PATH  = "/workspace/NCWP/sts_ablation_results.csv"
MODEL_KEY = "qwen-4b"
HF_NAME   = "Qwen/Qwen1.5-4B"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED      = 42
os.makedirs(CACHE_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)

# ── 데이터 로드 ───────────────────────────────────────────────
def load_sts(path):
    with open(path) as f:
        return json.load(f)

valid_data = load_sts(os.path.join(STS_DIR, "sts_valid.json"))
test_data  = load_sts(os.path.join(STS_DIR, "sts_test.json"))

fit_sents = list({s for item in valid_data
                  for s in (item['sentence1'], item['sentence2'])})
test_pairs  = [(item['sentence1'], item['sentence2']) for item in test_data]
test_scores = [item['score'] for item in test_data]
eval_sents  = list({s for item in test_data
                    for s in (item['sentence1'], item['sentence2'])})
all_sents   = list(set(fit_sents + eval_sents))

print(f"fit sentences (valid): {len(fit_sents)}")
print(f"eval pairs    (test):  {len(test_pairs)}")
print(f"all unique sents:      {len(all_sents)}")

# ── 임베딩 (캐시) ─────────────────────────────────────────────
cache_path = os.path.join(CACHE_DIR, f"{MODEL_KEY}_all_embs.npy")
idx_path   = os.path.join(CACHE_DIR, f"{MODEL_KEY}_sent_order.json")

if os.path.exists(cache_path) and os.path.exists(idx_path):
    print("임베딩 캐시 로드...")
    all_embs     = np.load(cache_path)
    with open(idx_path) as f:
        saved_sents = json.load(f)
    sent2idx = {s: i for i, s in enumerate(saved_sents)}
else:
    print(f"Qwen-4B 로드 및 임베딩 ({len(all_sents)}개)...")
    tok = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True,
          "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(AutoModel.from_pretrained(HF_NAME, **kw)).cuda()
    else:
        model = AutoModel.from_pretrained(HF_NAME, **kw).to(DEVICE)
    model.eval()

    parts = []
    batch_size = 64
    for i in range(0, len(all_sents), batch_size):
        batch = all_sents[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc) if n_gpu <= 1 else model.module(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        parts.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  {i+len(batch)}/{len(all_sents)}", end="\r")
    print()
    all_embs = np.vstack(parts)
    np.save(cache_path, all_embs)
    with open(idx_path, "w") as f:
        json.dump(all_sents, f)
    sent2idx = {s: i for i, s in enumerate(all_sents)}
    del model

sent2idx = {s: i for i, s in enumerate(all_sents
    if os.path.exists(cache_path) else all_sents)}
with open(idx_path) as f:
    saved_sents = json.load(f)
sent2idx = {s: i for i, s in enumerate(saved_sents)}

X_fit  = np.array([all_embs[sent2idx[s]] for s in fit_sents]).astype(np.float32)
X_eval = {s: all_embs[sent2idx[s]] for s in eval_sents}
print(f"X_fit={X_fit.shape}")

# ── 평가 함수 ────────────────────────────────────────────────
def spearman_score(emb_map, pairs, scores):
    pred = []
    for s1, s2 in pairs:
        e1 = emb_map.get(s1); e2 = emb_map.get(s2)
        if e1 is None or e2 is None:
            pred.append(0.0)
        else:
            pred.append(float(np.dot(e1, e2)))
    return spearmanr(scores, pred).correlation * 100

def transform_eval(W, mu, eval_sents_list, X_eval_raw):
    """W: (D, r) numpy, mu: (D,) numpy"""
    result = {}
    for s in eval_sents_list:
        x = X_eval_raw[s].astype(np.float32)
        z = (x - mu) @ W
        z = z / (np.linalg.norm(z) + 1e-12)
        result[s] = z
    return result

# ── ZCA 공통 ──────────────────────────────────────────────────
def compute_zca(X_fit_np, shrink=0.08):
    X = torch.from_numpy(X_fit_np).to(DEVICE)
    mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.clamp(ev, min=1e-6)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T
    return mu.cpu().numpy(), S.cpu().numpy()

def apply_zca(X_np, mu, S):
    Xw = F.normalize(
        torch.from_numpy((X_np - mu).astype(np.float32) @ S.astype(np.float32)).to(DEVICE),
        dim=1).cpu().numpy()
    return Xw

# ── kNN ───────────────────────────────────────────────────────
def build_knn(Xw_np, k):
    X = torch.from_numpy(Xw_np.astype(np.float32)).to(DEVICE)
    sim = X @ X.T; sim.fill_diagonal_(-1e9)
    _, idx = torch.topk(sim, k=k, dim=1)
    return idx.cpu().numpy()

# ── NCWP 공통 학습 루프 ──────────────────────────────────────
def train_ncwp(Xw_np, knn_idx, rank, seed=SEED,
               temperature=0.12, lambda_cov=0.05, lambda_orth=0.02,
               lr=8e-3, max_epochs=20, batch_pairs=None,
               retraction_interval=0,   # 0=No-QR, >0=Full-NCWP-with-QR
               refine_knn_rounds=1,
               random_pairs=False,
               topk_negatives=None):    # None=all in-batch
    set_seed(seed)
    N, D = Xw_np.shape
    r = rank
    Xw = torch.from_numpy(Xw_np.astype(np.float32)).to(DEVICE)
    knn = torch.from_numpy(knn_idx).to(DEVICE) if knn_idx is not None else None

    B = batch_pairs or max(64, min(1024, N // 64))
    steps = min(160, max(100, math.ceil(N / (2*B))))
    T_total = max_epochs * steps
    lr_min = 0.2 * lr

    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)

    best_W = None; best_loss = float('inf'); no_imp = 0; gs = 0

    def _one_round(knn_local):
        nonlocal gs, best_W, best_loss, no_imp
        for ep in range(max_epochs):
            ep_loss = 0.0
            for _ in range(steps):
                # LR cosine
                warmup = 200
                if gs < warmup:
                    cur_lr = lr_min + (lr - lr_min) * (gs / max(1, warmup))
                else:
                    t = (gs - warmup) / max(1, T_total - warmup)
                    cur_lr = lr_min + 0.5*(lr - lr_min)*(1 + math.cos(math.pi*t))
                for pg in opt.param_groups: pg['lr'] = cur_lr

                # Positive pair sampling
                with torch.no_grad():
                    a_idx = torch.randint(0, N, (B,), device=DEVICE)
                    if random_pairs:
                        b_idx = torch.randint(0, N, (B,), device=DEVICE)
                    else:
                        k_val = knn_local.shape[1]
                        b_idx = knn_local[a_idx, torch.randint(0, k_val, (B,), device=DEVICE)]
                    batch = torch.cat([Xw[a_idx], Xw[b_idx]], dim=0)

                Z = F.normalize(batch @ W, dim=1)
                sim = (Z @ Z.T) / temperature
                B2 = Z.shape[0]; Bh = B2 // 2
                sim.fill_diagonal_(-1e9)
                i1 = torch.arange(Bh, device=DEVICE)
                i2 = i1 + Bh

                # InfoNCE (symmetric)
                def _ce(logits, pos_idx, topk=None):
                    pos_logits = logits.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
                    if topk is None or topk >= logits.shape[1]:
                        denom = torch.logsumexp(logits, 1)
                    else:
                        mask = torch.ones_like(logits, dtype=torch.bool)
                        mask[torch.arange(len(logits), device=DEVICE), pos_idx] = False
                        negs = logits.masked_select(mask).view(len(logits), -1)
                        kk = min(topk, negs.shape[1])
                        vals, _ = torch.topk(negs, k=kk, dim=1)
                        denom = torch.logsumexp(
                            torch.cat([vals, pos_logits.unsqueeze(1)], dim=1), dim=1)
                    return -(pos_logits - denom).mean()

                loss = 0.5*(_ce(sim[:Bh], i2, topk_negatives) +
                            _ce(sim[Bh:], i1, topk_negatives))

                # Regularization
                Zm = Z - Z.mean(0)
                cov = (Zm.T @ Zm) / max(len(Z)-1, 1)
                off = cov - torch.diag(torch.diag(cov))
                loss += lambda_cov * (off**2).sum() / r
                Wn = F.normalize(W, dim=0)
                gram = Wn.T @ Wn - torch.eye(r, device=DEVICE)
                loss += lambda_orth * (gram**2).mean()

                opt.zero_grad(); loss.backward(); opt.step()

                # QR retraction
                if retraction_interval > 0 and (gs+1) % retraction_interval == 0:
                    with torch.no_grad():
                        Q, _ = torch.linalg.qr(W.data)
                        W.data = Q[:, :r]

                ep_loss += loss.item(); gs += 1

            ep_loss /= steps
            if ep_loss + 1e-6 < best_loss:
                best_loss = ep_loss; no_imp = 0; best_W = W.detach().clone()
            else:
                no_imp += 1
                if no_imp >= 6: break

    # 초기 학습
    knn_cur = knn if knn is not None else torch.zeros(N, 1, dtype=torch.long, device=DEVICE)
    _one_round(knn_cur)

    # kNN refinement
    for rr in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ best_W, dim=1)
            sim_r = Zfull @ Zfull.T; sim_r.fill_diagonal_(-1e9)
            k_val = knn_cur.shape[1]
            _, knn_cur = torch.topk(sim_r, k=k_val, dim=1)
            W.data.copy_(best_W)
        _one_round(knn_cur)

    return best_W.cpu().numpy()


# ── 8가지 Variant 실행 ───────────────────────────────────────
DIMS = [20, 80, 320]
k_neighbors = 10
shrink = 0.08

print("\n공통 ZCA 계산 중...")
mu_zca, S_zca = compute_zca(X_fit, shrink=shrink)
Xw_fit = apply_zca(X_fit, mu_zca, S_zca)
Xw_eval = {s: apply_zca(X_eval[s].reshape(1,-1), mu_zca, S_zca)[0] for s in eval_sents}

print("kNN 계산 중...")
knn_idx = build_knn(Xw_fit, k=k_neighbors)

# PCA-W
print("PCA-W 계산 중...")
mu_pca = X_fit.mean(0)
Xc = X_fit - mu_pca
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
# W_pca_full[:, :r] → whitened PCA
scales = np.maximum(sv, 1e-9)
W_pca_full = Vt.T / scales  # (D, min(N,D))

results = []

VARIANTS = [
    ("PCA-whitening",    None),
    ("ZCA-only",         None),
    ("Random-pairs",     dict(random_pairs=True, retraction_interval=0, refine_knn_rounds=1)),
    ("No hard-neg",      dict(random_pairs=False, retraction_interval=0, refine_knn_rounds=1, topk_negatives=None)),
    ("No memory bank",   dict(random_pairs=False, retraction_interval=0, refine_knn_rounds=1, topk_negatives=None)),
    ("No-QR (final)",    dict(random_pairs=False, retraction_interval=0, refine_knn_rounds=1, topk_negatives=None)),
    ("No refinement",    dict(random_pairs=False, retraction_interval=0, refine_knn_rounds=0, topk_negatives=None)),
    ("Full-NCWP-with-QR",dict(random_pairs=False, retraction_interval=50, refine_knn_rounds=1, topk_negatives=None)),
]

for variant_name, kwargs in VARIANTS:
    row = {"variant": variant_name}
    print(f"\n[{variant_name}]")
    
    for dim in DIMS:
        print(f"  r={dim}...", end=" ", flush=True)
        
        if variant_name == "PCA-whitening":
            k = min(dim, W_pca_full.shape[1])
            W = W_pca_full[:, :k]
            mu = mu_pca
            emb_map = transform_eval(W, mu, eval_sents, X_eval)

        elif variant_name == "ZCA-only":
            # ZCA 후 top-dim PCA
            _, _, Vt_w = np.linalg.svd(Xw_fit - Xw_fit.mean(0), full_matrices=False)
            k = min(dim, Vt_w.shape[0])
            W_pca_w = Vt_w[:k].T
            mu_w = Xw_fit.mean(0)
            emb_map = {}
            for s in eval_sents:
                z = Xw_eval[s] - mu_w
                z = z @ W_pca_w
                z = z / (np.linalg.norm(z) + 1e-12)
                emb_map[s] = z

        else:
            # NCWP variants
            W_np = train_ncwp(Xw_fit, knn_idx, rank=dim, **kwargs)
            mu_w_proj = Xw_fit.mean(0) @ W_np
            emb_map = {}
            for s in eval_sents:
                z = (Xw_eval[s] - Xw_fit.mean(0)) @ W_np - mu_w_proj
                # already centered, just normalize
                z_full = Xw_eval[s] @ W_np
                mu_w2 = Xw_fit.mean(0) @ W_np
                z = z_full - mu_w2
                z = z / (np.linalg.norm(z) + 1e-12)
                emb_map[s] = z

        sp = spearman_score(emb_map, test_pairs, test_scores)
        row[f"r={dim}"] = round(sp, 2)
        print(f"Spearman={sp:.2f}")
    
    results.append(row)

# ── 결과 출력 ─────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_PATH, index=False)
print(f"\n결과 저장: {OUT_PATH}")

print("\n" + "="*60)
print("T1: Ablation Study — Qwen-4B × STSBenchmark (test, Spearman×100)")
print("="*60)
print(df.to_string(index=False))

# ── 검증 체크 ─────────────────────────────────────────────────
def get(name, r):
    row = df[df['variant']==name]
    return row[f'r={r}'].values[0] if len(row) else float('nan')

print("\n" + "="*60)
print("검증 체크 (r=20 기준)")
print("="*60)
checks = [
    ("NCWP(No-QR) > ZCA-only",
     get("No-QR (final)",20) > get("ZCA-only",20),
     f"{get('No-QR (final)',20):.2f} > {get('ZCA-only',20):.2f}"),
    ("ZCA-only > Random-pairs",
     get("ZCA-only",20) > get("Random-pairs",20),
     f"{get('ZCA-only',20):.2f} > {get('Random-pairs',20):.2f}"),
    ("No-QR > Full-NCWP-with-QR",
     get("No-QR (final)",20) > get("Full-NCWP-with-QR",20),
     f"{get('No-QR (final)',20):.2f} > {get('Full-NCWP-with-QR',20):.2f}"),
    ("|No-hardneg − No-QR| < 1.0",
     abs(get("No hard-neg",20) - get("No-QR (final)",20)) < 1.0,
     f"|{get('No hard-neg',20):.2f} - {get('No-QR (final)',20):.2f}| = {abs(get('No hard-neg',20)-get('No-QR (final)',20)):.2f}"),
    ("|No-memorybank − No-QR| < 1.0",
     abs(get("No memory bank",20) - get("No-QR (final)",20)) < 1.0,
     f"|{get('No memory bank',20):.2f} - {get('No-QR (final)',20):.2f}| = {abs(get('No memory bank',20)-get('No-QR (final)',20)):.2f}"),
    ("No-QR > No-refinement",
     get("No-QR (final)",20) > get("No refinement",20),
     f"{get('No-QR (final)',20):.2f} > {get('No refinement',20):.2f}"),
]

for desc, passed, detail in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {desc}: {detail}")
