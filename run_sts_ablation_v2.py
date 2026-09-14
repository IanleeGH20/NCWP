"""
run_sts_ablation_v2.py
─────────────────────────────────────────────────────────────
T1: Ablation Study — Qwen-4B × STSBenchmark (개선된 버전)

하이퍼파라미터 개선:
  k: 10 → 40 (더 넓은 neighborhood)
  cosine_tau: 0 → 0.2 (false positive 필터)
  batch_pairs: 64 → 256 (더 많은 in-batch negative)
  temperature: 0.12 → 0.07 (positive 품질 향상 시 더 sharp)
  QR retraction_interval: 50 → 200 (gradient 충분히 수렴 후 retraction)

신규 구현:
  Memory Bank (size=4096): 과거 배치 임베딩을 queue에 보관 → negative 확장
  Hard Negative (K=32): 가장 어려운 negative를 우선 선택

8 Variants 구조:
  Full NCWP (variants 4,5,8 기준) = memory bank + hard neg + refinement
  1. PCA-whitening            : closed-form
  2. ZCA-only                 : closed-form
  3. Random-pairs             : ZCA + random positive
  4. No hard-neg              : memory bank, NO hard neg
  5. No memory bank           : NO memory bank, hard neg (in-batch only)
  6. No-QR (final)            : NO memory bank, NO hard neg, NO QR
  7. No refinement            : No-QR but refine_knn_rounds=0
  8. Full-NCWP-with-QR        : memory bank + hard neg + QR retraction (200 steps)
"""
import os, math, json, random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

STS_DIR   = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
OUT_PATH  = "/workspace/NCWP/sts_ablation_results_v2.csv"
HF_NAME   = "Qwen/Qwen1.5-4B"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED      = 42
os.makedirs(CACHE_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── 데이터 ────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f: return json.load(f)

valid_data = load_json(os.path.join(STS_DIR, "sts_valid.json"))
test_data  = load_json(os.path.join(STS_DIR, "sts_test.json"))

fit_sents   = list({s for item in valid_data for s in (item['sentence1'], item['sentence2'])})
test_pairs  = [(item['sentence1'], item['sentence2']) for item in test_data]
test_scores = [item['score'] for item in test_data]
eval_sents  = list({s for item in test_data for s in (item['sentence1'], item['sentence2'])})
all_sents   = list(set(fit_sents + eval_sents))

print(f"fit={len(fit_sents)}, eval_pairs={len(test_pairs)}, all_unique={len(all_sents)}")

# ── 임베딩 (캐시) ─────────────────────────────────────────────
cache_np  = os.path.join(CACHE_DIR, "qwen-4b_all_embs.npy")
cache_idx = os.path.join(CACHE_DIR, "qwen-4b_sent_order.json")

if os.path.exists(cache_np) and os.path.exists(cache_idx):
    print("임베딩 캐시 로드...")
    all_embs = np.load(cache_np)
    with open(cache_idx) as f: saved_sents = json.load(f)
else:
    print(f"Qwen-4B 임베딩 ({len(all_sents)}개)...")
    tok = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True,
          "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    n_gpu = torch.cuda.device_count()
    model = (torch.nn.DataParallel(AutoModel.from_pretrained(HF_NAME, **kw)).cuda()
             if n_gpu > 1 else
             AutoModel.from_pretrained(HF_NAME, **kw).to(DEVICE))
    model.eval()
    parts = []
    for i in range(0, len(all_sents), 64):
        batch = all_sents[i:i+64]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc) if n_gpu <= 1 else model.module(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        parts.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        if (i//64) % 10 == 0: print(f"  {i+len(batch)}/{len(all_sents)}", end="\r")
    print()
    all_embs = np.vstack(parts)
    np.save(cache_np, all_embs); saved_sents = all_sents
    with open(cache_idx, "w") as f: json.dump(saved_sents, f)
    del model

sent2idx = {s: i for i, s in enumerate(saved_sents)}
X_fit    = np.array([all_embs[sent2idx[s]] for s in fit_sents]).astype(np.float32)
X_eval   = {s: all_embs[sent2idx[s]].astype(np.float32) for s in eval_sents}
print(f"X_fit={X_fit.shape}")

# ── Spearman 평가 ─────────────────────────────────────────────
def spearman_score(emb_map):
    pred = [float(np.dot(emb_map.get(s1, np.zeros(1)), emb_map.get(s2, np.zeros(1))))
            for s1, s2 in test_pairs]
    return spearmanr(test_scores, pred).correlation * 100

# ── ZCA 공통 ─────────────────────────────────────────────────
def compute_zca(X_np, shrink=0.08):
    X = torch.from_numpy(X_np).to(DEVICE)
    mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    return mu.cpu().numpy(), S.cpu().numpy()

def apply_zca_np(X_np, mu, S):
    return F.normalize(torch.from_numpy(
        (X_np - mu).astype(np.float32) @ S.astype(np.float32)).to(DEVICE), dim=1).cpu().numpy()

print("ZCA 계산...")
mu_zca, S_zca = compute_zca(X_fit)
Xw_fit  = apply_zca_np(X_fit, mu_zca, S_zca)
Xw_eval = {s: apply_zca_np(X_eval[s].reshape(1,-1), mu_zca, S_zca)[0] for s in eval_sents}

# ── kNN (whitened space, cosine_tau 포함) ─────────────────────
K_NEIGH   = 40
COSINE_TAU = 0.2

print(f"kNN (k={K_NEIGH}, tau={COSINE_TAU}) 계산...")
Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
_, knn_idx_t = torch.topk(sim_all, k=K_NEIGH, dim=1)
knn_idx  = knn_idx_t.cpu().numpy()    # (N, 40)
knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()  # (N, 40)

# ── Memory Bank ───────────────────────────────────────────────
class MemoryBank:
    def __init__(self, size, dim):
        self.bank = torch.zeros(size, dim, device=DEVICE)
        self.ptr  = 0; self.full = False; self.size = size
    def update(self, z_detach):
        B = z_detach.shape[0]
        end = min(self.ptr + B, self.size)
        n1  = end - self.ptr
        self.bank[self.ptr:end] = z_detach[:n1]
        if B > n1:
            self.bank[:B-n1] = z_detach[n1:]
            self.ptr = B - n1; self.full = True
        else:
            self.ptr = end % self.size
            if self.ptr == 0: self.full = True
    def get(self):
        return self.bank if self.full else self.bank[:self.ptr]

# ── NCWP 학습 (개선 버전) ─────────────────────────────────────
def train_ncwp_v2(Xw_np, knn_idx_np, knn_sims_np, rank,
                  seed=SEED,
                  temperature=0.07,
                  lambda_cov=0.05, lambda_orth=0.02,
                  lr=8e-3, max_epochs=20,
                  batch_pairs=256,
                  cosine_tau=COSINE_TAU,
                  retraction_interval=0,
                  refine_knn_rounds=1,
                  random_pairs=False,
                  use_memory_bank=False,  # memory bank 사용 여부
                  use_hard_neg=False,     # hard negative 선택 여부
                  hard_neg_k=32,          # hard negative 수
                  memory_bank_size=4096):
    set_seed(seed)
    N, D = Xw_np.shape; r = rank
    Xw = torch.from_numpy(Xw_np.astype(np.float32)).to(DEVICE)
    knn  = torch.from_numpy(knn_idx_np).to(DEVICE)
    sims = torch.from_numpy(knn_sims_np.astype(np.float32)).to(DEVICE)

    B = batch_pairs
    steps = min(160, max(100, math.ceil(N / (2*B))))
    T_total = max_epochs * steps
    lr_min  = 0.2 * lr
    warmup  = 200

    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)

    mbank = MemoryBank(memory_bank_size, r) if use_memory_bank else None
    best_W = None; best_loss = float('inf'); no_imp = 0; gs = 0

    def _one_round(knn_loc, sims_loc):
        nonlocal gs, best_W, best_loss, no_imp
        for ep in range(max_epochs):
            ep_loss = 0.0
            for _ in range(steps):
                # LR cosine warmup
                if gs < warmup:
                    cur_lr = lr_min + (lr - lr_min) * (gs / max(1, warmup))
                else:
                    t = (gs - warmup) / max(1, T_total - warmup)
                    cur_lr = lr_min + 0.5*(lr - lr_min)*(1 + math.cos(math.pi*t))
                for pg in opt.param_groups: pg['lr'] = cur_lr

                # Positive pair sampling (cosine_tau 필터)
                with torch.no_grad():
                    a_idx = torch.randint(0, N, (B,), device=DEVICE)
                    if random_pairs:
                        b_idx = torch.randint(0, N, (B,), device=DEVICE)
                    else:
                        b_list = []
                        for anc in a_idx.tolist():
                            valid = (sims_loc[anc] >= cosine_tau).nonzero(as_tuple=True)[0]
                            if len(valid) == 0:
                                valid = torch.arange(knn_loc.shape[1], device=DEVICE)
                            pick = valid[torch.randint(0, len(valid), (1,)).item()]
                            b_list.append(knn_loc[anc, pick].item())
                        b_idx = torch.tensor(b_list, device=DEVICE)
                    batch = torch.cat([Xw[a_idx], Xw[b_idx]], dim=0)

                # Forward
                Z   = F.normalize(batch @ W, dim=1)
                Za  = Z[:B]; Zb = Z[B:]

                # ── 독립 ablation 로직 ────────────────────────
                sim_a_to_b = (Za @ Zb.T) / temperature
                sim_a_to_a = (Za @ Za.T) / temperature
                sim_a_to_a.fill_diagonal_(-1e9)
                sim_b_to_a = (Zb @ Za.T) / temperature
                sim_b_to_b = (Zb @ Zb.T) / temperature
                sim_b_to_b.fill_diagonal_(-1e9)

                if use_memory_bank and mbank is not None and len(mbank.get()) > 0:
                    bank = mbank.get().detach()
                    sim_a_bank = (Za @ bank.T) / temperature
                    sim_b_bank = (Zb @ bank.T) / temperature
                else:
                    sim_a_bank = None; sim_b_bank = None

                pos_logit_ab = sim_a_to_b.diag()
                neg_a_inb = sim_a_to_b.clone(); neg_a_inb.fill_diagonal_(-1e9)
                neg_pool_a = torch.cat([neg_a_inb, sim_a_to_a], dim=1)
                if sim_a_bank is not None:
                    neg_pool_a = torch.cat([neg_pool_a, sim_a_bank], dim=1)

                pos_logit_ba = sim_b_to_a.diag()
                neg_b_inb = sim_b_to_a.clone(); neg_b_inb.fill_diagonal_(-1e9)
                neg_pool_b = torch.cat([neg_b_inb, sim_b_to_b], dim=1)
                if sim_b_bank is not None:
                    neg_pool_b = torch.cat([neg_pool_b, sim_b_bank], dim=1)

                if use_hard_neg:
                    K = min(hard_neg_k, neg_pool_a.shape[1])
                    neg_a_sel, _ = torch.topk(neg_pool_a, k=K, dim=1)
                    neg_b_sel, _ = torch.topk(neg_pool_b, k=K, dim=1)
                else:
                    neg_a_sel = neg_pool_a
                    neg_b_sel = neg_pool_b

                denom_a = torch.logsumexp(torch.cat([pos_logit_ab.unsqueeze(1), neg_a_sel], dim=1), dim=1)
                denom_b = torch.logsumexp(torch.cat([pos_logit_ba.unsqueeze(1), neg_b_sel], dim=1), dim=1)
                loss = 0.5*( -(pos_logit_ab - denom_a).mean()
                           + -(pos_logit_ba - denom_b).mean() )

                # Regularization
                Z_all = Z
                Zm = Z_all - Z_all.mean(0)
                cov = (Zm.T @ Zm) / max(len(Z_all)-1, 1)
                off = cov - torch.diag(torch.diag(cov))
                loss += lambda_cov * (off**2).sum() / r
                Wn = F.normalize(W, dim=0)
                gram = Wn.T @ Wn - torch.eye(r, device=DEVICE)
                loss += lambda_orth * (gram**2).mean()

                opt.zero_grad(); loss.backward(); opt.step()

                # Memory bank 업데이트 (Za, Zb 모두 큐에 추가)
                if use_memory_bank and mbank is not None:
                    mbank.update(Z.detach())

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
    _one_round(knn, sims)

    # kNN refinement
    for _ in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ best_W, dim=1)
            sim_r = Zfull @ Zfull.T; sim_r.fill_diagonal_(-1e9)
            _, knn_new = torch.topk(sim_r, k=K_NEIGH, dim=1)
            sims_new = sim_r.gather(1, knn_new)
            W.data.copy_(best_W)
        if use_memory_bank and mbank is not None:
            mbank.ptr = 0; mbank.full = False  # reset bank
        _one_round(knn_new, sims_new)

    return best_W.cpu().numpy()

# ── 공통 eval ─────────────────────────────────────────────────
def make_emb_map(W_np, Xw_mean):
    mu_w = Xw_mean @ W_np
    result = {}
    for s in eval_sents:
        z = Xw_eval[s] @ W_np - mu_w
        result[s] = z / (np.linalg.norm(z) + 1e-12)
    return result

Xw_mean = Xw_fit.mean(0)

# ── PCA-W ─────────────────────────────────────────────────────
mu_pca = X_fit.mean(0)
Xc_pca = X_fit - mu_pca
_, sv, Vt_pca = np.linalg.svd(Xc_pca, full_matrices=False)
W_pca_full = Vt_pca.T / np.maximum(sv, 1e-9)

# ── ZCA-only PCA 행렬 ─────────────────────────────────────────
Xwc = Xw_fit - Xw_fit.mean(0)
_, _, Vt_w = np.linalg.svd(Xwc, full_matrices=False)
Xw_mu = Xw_fit.mean(0)

# ── 8 Variants ───────────────────────────────────────────────
DIMS = [20, 80, 320]
results = []

VARIANTS = [
    # (name, ncwp_kwargs or None for special)
    ("PCA-whitening",     "pca"),
    ("ZCA-only",          "zca"),
    ("Random-pairs",      dict(random_pairs=True,  use_memory_bank=False, use_hard_neg=False, retraction_interval=0, refine_knn_rounds=1)),
    ("No hard-neg",       dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=False, retraction_interval=0, refine_knn_rounds=1)),
    ("No memory bank",    dict(random_pairs=False, use_memory_bank=False, use_hard_neg=True,  retraction_interval=0, refine_knn_rounds=1)),
    ("No-QR (final)",     dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0, refine_knn_rounds=1)),
    ("No refinement",     dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0, refine_knn_rounds=0)),
    ("Full-NCWP-with-QR", dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=True,  retraction_interval=200, refine_knn_rounds=1)),
]

for variant_name, cfg in VARIANTS:
    row = {"variant": variant_name}
    print(f"\n[{variant_name}]")
    for dim in DIMS:
        print(f"  r={dim}...", end=" ", flush=True)
        if cfg == "pca":
            k = min(dim, W_pca_full.shape[1])
            emb_map = {}
            for s in eval_sents:
                z = (X_eval[s] - mu_pca) @ W_pca_full[:, :k]
                emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
        elif cfg == "zca":
            k = min(dim, Vt_w.shape[0])
            W_zca = Vt_w[:k].T; mu_zca_w = Xw_mu @ W_zca
            emb_map = {}
            for s in eval_sents:
                z = Xw_eval[s] @ W_zca - mu_zca_w
                emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
        else:
            W_np = train_ncwp_v2(Xw_fit, knn_idx, knn_sims, rank=dim, **cfg)
            emb_map = make_emb_map(W_np, Xw_mean)
        sp = spearman_score(emb_map)
        row[f"r={dim}"] = round(sp, 2)
        print(f"Spearman={sp:.2f}")
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(OUT_PATH, index=False)
print(f"\n결과 저장: {OUT_PATH}")

print("\n" + "="*62)
print("T1: Ablation — Qwen-4B × STSBenchmark (test, Spearman×100)")
print("="*62)
print(df.to_string(index=False))

def g(name, r): 
    row = df[df['variant']==name]; return row[f'r={r}'].values[0] if len(row) else float('nan')

print("\n검증 체크 (r=20):")
checks = [
    ("No-QR > ZCA-only",           g("No-QR (final)",20) > g("ZCA-only",20)),
    ("ZCA-only > Random-pairs",     g("ZCA-only",20) > g("Random-pairs",20)),
    ("No-QR > Full-NCWP-with-QR",  g("No-QR (final)",20) > g("Full-NCWP-with-QR",20)),
    ("|No-hardneg − No-QR| < 1.0", abs(g("No hard-neg",20)-g("No-QR (final)",20)) < 1.0),
    ("|No-membank − No-QR| < 1.0", abs(g("No memory bank",20)-g("No-QR (final)",20)) < 1.0),
    ("No-QR > No-refinement",       g("No-QR (final)",20) > g("No refinement",20)),
]
for desc, ok in checks:
    v1 = g(desc.split(">")[0].split("|")[0].strip().replace("No-","No-").strip()
           .replace("No-QR","No-QR (final)").replace("ZCA","ZCA-only")
           .replace("No-hardneg","No hard-neg").replace("No-membank","No memory bank")
           .replace("No-refinement","No refinement"), 20)
    print(f"  {'✅' if ok else '❌'} {desc}")
