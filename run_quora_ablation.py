"""
run_quora_ablation.py
─────────────────────────────────────────────────────────────
T1(Quora): Ablation Study — Qwen-4B × Quora

STS 실험과 동일한 8 Variants:
  1. PCA-whitening
  2. ZCA-only
  3. Random-pairs
  4. No hard-neg       (memory bank O, hard neg X)
  5. No memory bank    (memory bank X, hard neg O)
  6. No-QR (final)     (memory bank X, hard neg X, QR X)
  7. No refinement     (No-QR + refinement X)
  8. Full-NCWP-with-QR (memory bank O, hard neg O, QR O)

Fit:  query_sample N=2000 (qwen-4b, 2560dim)
Eval: nDCG@10, full corpus 522K, test queries 10K
Dims: r ∈ {20, 80, 320}

하이퍼파라미터 (STS v2와 동일):
  k=40, cosine_tau=0.2, batch_pairs=256, temperature=0.07
  QR retraction_interval=200, memory_bank_size=4096, hard_neg_k=32
"""
import os, math, json, random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from datasets import load_dataset

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--N", type=int, default=2000)
_p.add_argument("--model", type=str, default="qwen-4b")
_args = _p.parse_args()

CACHE   = f'/workspace/NCWP/quora_results/embedding_cache/{_args.model}'
OUT_DIR = '/workspace/NCWP/quora_results/fit_query_sample'
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED    = 42
FIT_N   = _args.N
_m_suffix = "" if _args.model == "qwen-4b" else f"_{_args.model}"
OUT_CSV = f'/workspace/NCWP/quora_ablation_N{FIT_N}{_m_suffix}.csv'

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── 임베딩 로드 ───────────────────────────────────────────────
print("임베딩 로드...")
corpus_np = np.load(f'{CACHE}/corpus_base.npy').astype(np.float32)
query_np  = np.load(f'{CACHE}/test_queries_base.npy').astype(np.float32)
fit_np    = np.load(f'{CACHE}/fit_query_sample_N{FIT_N}.npy').astype(np.float32)
print(f"  corpus={corpus_np.shape}, query={query_np.shape}, fit={fit_np.shape}")

# ── qrels ─────────────────────────────────────────────────────
print("qrels 로드...")
ds_corp = load_dataset("mteb/quora", "corpus",  split="corpus")
ds_qs   = load_dataset("mteb/quora", "queries", split="queries")
ds_qrel = load_dataset("mteb/quora", split="test")
corpus_ids = [str(r['_id']) for r in ds_corp]
qid2tx     = {str(r['_id']): r['text'] for r in ds_qs}
qrels = defaultdict(set)
for r in ds_qrel:
    qrels[str(r['query-id'])].add(str(r['corpus-id']))
test_qids = sorted([q for q in qrels if q in qid2tx])
print(f"  corpus={len(corpus_ids)}, test_qids={len(test_qids)}")

# ── 평가 함수 ─────────────────────────────────────────────────
def evaluate_ndcg(c_t, q_t, chunk=500):
    """GPU topk 기반 nDCG@10"""
    ndcg10 = []; recall100 = []; ap_list = []
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi+chunk)
        with torch.no_grad():
            sim = (q_t[qi:qe] @ c_t.T).float()
            _, top_idx = torch.topk(sim, k=100, dim=1)
            top100 = top_idx.cpu().numpy()
        for ci, q2 in enumerate(range(qi, qe)):
            qid = test_qids[q2]
            if qid not in qrels: continue
            rel = qrels[qid]
            ranked = [corpus_ids[i] for i in top100[ci]]
            h10 = [1.0 if d in rel else 0.0 for d in ranked[:10]]
            dcg  = sum(h/np.log2(r+2) for r,h in enumerate(h10))
            idcg = sum(1.0/np.log2(r+2) for r in range(min(len(rel),10)))
            ndcg10.append(dcg/idcg if idcg>0 else 0.0)
            recall100.append(sum(1.0 if d in rel else 0.0 for d in ranked)/len(rel))
            nr=0; ps=0.0
            for r,d in enumerate(ranked):
                if d in rel: nr+=1; ps+=nr/(r+1)
            ap_list.append(ps/len(rel))
    return np.mean(ndcg10)*100

def to_gpu_half(X_np, mu, W):
    """numpy 임베딩 → ZCA 투영 → GPU half tensor"""
    chunk = 2000
    parts = []
    for i in range(0, len(X_np), chunk):
        xb = torch.from_numpy(X_np[i:i+chunk]).to(DEVICE)
        xw = F.normalize((xb - mu) @ W, dim=1)
        parts.append(xw.half())
    return torch.cat(parts, dim=0)

# ── ZCA 공통 ─────────────────────────────────────────────────
def compute_zca(X_np, shrink=0.08):
    X = torch.from_numpy(X_np).to(DEVICE)
    mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    return mu, S

print("ZCA 계산...")
mu_zca, S_zca = compute_zca(fit_np)

def apply_zca(X_np):
    X = torch.from_numpy(X_np).to(DEVICE)
    return F.normalize((X - mu_zca) @ S_zca, dim=1).cpu().numpy()

Xw_fit = apply_zca(fit_np)

# ── kNN (whitened, cosine_tau) ────────────────────────────────
K_NEIGH    = 40
COSINE_TAU = 0.2
print(f"kNN (k={K_NEIGH}, tau={COSINE_TAU}) 계산...")
Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
_, knn_idx_t  = torch.topk(sim_all, k=K_NEIGH, dim=1)
knn_idx  = knn_idx_t.cpu().numpy()
knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()

# ── Memory Bank ───────────────────────────────────────────────
class MemoryBank:
    def __init__(self, size, dim):
        self.bank = torch.zeros(size, dim, device=DEVICE)
        self.ptr = 0; self.full = False; self.size = size
    def update(self, z):
        B = z.shape[0]
        end = min(self.ptr + B, self.size)
        n1 = end - self.ptr
        self.bank[self.ptr:end] = z[:n1]
        if B > n1:
            self.bank[:B-n1] = z[n1:]
            self.ptr = B - n1; self.full = True
        else:
            self.ptr = end % self.size
            if self.ptr == 0: self.full = True
    def get(self):
        return self.bank if self.full else self.bank[:self.ptr]

# ── NCWP 학습 ─────────────────────────────────────────────────
def train_ncwp(Xw_np, knn_idx_np, knn_sims_np, rank, seed=SEED,
               temperature=0.07, lambda_cov=0.05, lambda_orth=0.02,
               lr=8e-3, max_epochs=20, batch_pairs=256,
               cosine_tau=COSINE_TAU, retraction_interval=0,
               refine_knn_rounds=1, random_pairs=False,
               use_memory_bank=False, use_hard_neg=False,
               hard_neg_k=32, memory_bank_size=4096):
    set_seed(seed)
    N, D = Xw_np.shape; r = rank
    Xw   = torch.from_numpy(Xw_np.astype(np.float32)).to(DEVICE)
    knn  = torch.from_numpy(knn_idx_np).to(DEVICE)
    sims = torch.from_numpy(knn_sims_np.astype(np.float32)).to(DEVICE)

    B = batch_pairs
    steps   = min(160, max(100, math.ceil(N / (2*B))))
    T_total = max_epochs * steps
    lr_min  = 0.2 * lr; warmup = 200

    W   = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    mbank = MemoryBank(memory_bank_size, r) if use_memory_bank else None
    best_W = None; best_loss = float('inf'); no_imp = 0; gs = 0

    def _one_round(knn_loc, sims_loc):
        nonlocal gs, best_W, best_loss, no_imp
        for ep in range(max_epochs):
            ep_loss = 0.0
            for _ in range(steps):
                if gs < warmup:
                    cur_lr = lr_min + (lr - lr_min) * gs / max(1, warmup)
                else:
                    t = (gs - warmup) / max(1, T_total - warmup)
                    cur_lr = lr_min + 0.5*(lr - lr_min)*(1+math.cos(math.pi*t))
                for pg in opt.param_groups: pg['lr'] = cur_lr

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

                Z  = F.normalize(batch @ W, dim=1)
                Za = Z[:B]; Zb = Z[B:]

                # ── 독립 ablation 로직 ────────────────────────
                # Step 1: negative pool 구성 (memory bank 사용 시 확장)
                # 기본: in-batch (Za, Zb) 모두 negative 후보로 사용
                # in-batch negative for anchor a_i: Zb (B개) + Za (B개, 자기 자신 제외)
                sim_a_to_b = (Za @ Zb.T) / temperature   # (B, B) — positive 포함
                sim_a_to_a = (Za @ Za.T) / temperature   # (B, B) — anchor-anchor neg
                sim_a_to_a.fill_diagonal_(-1e9)          # 자기자신 제외

                # symmetric direction (b → a)
                sim_b_to_a = (Zb @ Za.T) / temperature   # (B, B) — positive 포함
                sim_b_to_b = (Zb @ Zb.T) / temperature   # (B, B)
                sim_b_to_b.fill_diagonal_(-1e9)

                # Step 2: memory bank 추가 (옵션)
                if use_memory_bank and mbank is not None and len(mbank.get()) > 0:
                    bank = mbank.get().detach()
                    sim_a_bank = (Za @ bank.T) / temperature   # (B, M)
                    sim_b_bank = (Zb @ bank.T) / temperature   # (B, M)
                else:
                    sim_a_bank = None; sim_b_bank = None

                # Step 3: anchor a → positive Zb, negatives 구성
                pos_logit_ab = sim_a_to_b.diag()          # 대각이 positive
                # negative 집합: Zb의 비대각 + Za의 비대각 + (memory bank)
                neg_a_inb = sim_a_to_b.clone()
                neg_a_inb.fill_diagonal_(-1e9)            # positive 제거
                neg_pool_a = torch.cat([neg_a_inb, sim_a_to_a], dim=1)  # (B, 2B)
                if sim_a_bank is not None:
                    neg_pool_a = torch.cat([neg_pool_a, sim_a_bank], dim=1)

                pos_logit_ba = sim_b_to_a.diag()
                neg_b_inb = sim_b_to_a.clone()
                neg_b_inb.fill_diagonal_(-1e9)
                neg_pool_b = torch.cat([neg_b_inb, sim_b_to_b], dim=1)
                if sim_b_bank is not None:
                    neg_pool_b = torch.cat([neg_pool_b, sim_b_bank], dim=1)

                # Step 4: hard negative 선택 (옵션)
                if use_hard_neg:
                    K = min(hard_neg_k, neg_pool_a.shape[1])
                    neg_a_sel, _ = torch.topk(neg_pool_a, k=K, dim=1)
                    neg_b_sel, _ = torch.topk(neg_pool_b, k=K, dim=1)
                else:
                    neg_a_sel = neg_pool_a
                    neg_b_sel = neg_pool_b

                # Step 5: InfoNCE (symmetric)
                denom_a = torch.logsumexp(
                    torch.cat([pos_logit_ab.unsqueeze(1), neg_a_sel], dim=1), dim=1)
                denom_b = torch.logsumexp(
                    torch.cat([pos_logit_ba.unsqueeze(1), neg_b_sel], dim=1), dim=1)
                loss = 0.5*( -(pos_logit_ab - denom_a).mean()
                           + -(pos_logit_ba - denom_b).mean() )

                Zm = Z - Z.mean(0)
                cov = (Zm.T @ Zm) / max(len(Z)-1,1)
                off = cov - torch.diag(torch.diag(cov))
                loss += lambda_cov*(off**2).sum()/r
                Wn = F.normalize(W, dim=0)
                loss += lambda_orth*((Wn.T @ Wn - torch.eye(r,device=DEVICE))**2).mean()

                opt.zero_grad(); loss.backward(); opt.step()
                if use_memory_bank and mbank is not None:
                    mbank.update(torch.cat([Za, Zb], dim=0).detach())
                if retraction_interval > 0 and (gs+1) % retraction_interval == 0:
                    with torch.no_grad():
                        Q, _ = torch.linalg.qr(W.data); W.data = Q[:, :r]
                ep_loss += loss.item(); gs += 1

            ep_loss /= steps
            if ep_loss + 1e-6 < best_loss:
                best_loss = ep_loss; no_imp = 0; best_W = W.detach().clone()
            else:
                no_imp += 1
                if no_imp >= 6: break

    _one_round(knn, sims)
    for _ in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ best_W, dim=1)
            sim_r = Zfull @ Zfull.T; sim_r.fill_diagonal_(-1e9)
            _, knn_new = torch.topk(sim_r, k=K_NEIGH, dim=1)
            sims_new = sim_r.gather(1, knn_new)
            W.data.copy_(best_W)
        if mbank: mbank.ptr=0; mbank.full=False
        _one_round(knn_new, sims_new)
    return best_W.cpu().numpy()

# ── 공통 투영 적용 ─────────────────────────────────────────────
def project_and_eval(W_learned, use_zca=True):
    """W_learned: (D, r) — ZCA 공간 기준 or original 공간"""
    Xw_mean = Xw_fit.mean(0)
    mu_w = torch.from_numpy(Xw_mean).to(DEVICE) @ torch.from_numpy(W_learned).to(DEVICE)
    W_t  = torch.from_numpy(W_learned).to(DEVICE)

    def _project(X_np):
        parts = []
        for i in range(0, len(X_np), 2000):
            xb = torch.from_numpy(X_np[i:i+2000]).to(DEVICE)
            if use_zca:
                xw = F.normalize((xb - mu_zca) @ S_zca, dim=1)
                xp = F.normalize(xw @ W_t - mu_w, dim=1)
            else:
                xp = F.normalize(xb @ W_t, dim=1)
            parts.append(xp.half())
        return torch.cat(parts, dim=0)

    c_t = _project(corpus_np)
    q_t = _project(query_np)
    return evaluate_ndcg(c_t, q_t)

# PCA-W 준비
mu_pca = fit_np.mean(0)
Xc_pca = fit_np - mu_pca
_, sv, Vt_pca = np.linalg.svd(Xc_pca, full_matrices=False)
W_pca_full = Vt_pca.T / np.maximum(sv, 1e-9)  # (D, min(N,D))

# ZCA-only PCA 행렬
Xwc = Xw_fit - Xw_fit.mean(0)
_, _, Vt_w = np.linalg.svd(Xwc, full_matrices=False)
Xw_mu = Xw_fit.mean(0)

DIMS = [20, 80, 320]

VARIANTS = [
    # PCA-whitening, ZCA-only은 기존 scaling 실험에서 계산됨 → 스킵
    ("Random-pairs",      dict(random_pairs=True,  use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No hard-neg",       dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No memory bank",    dict(random_pairs=False, use_memory_bank=False, use_hard_neg=True,  retraction_interval=0,   refine_knn_rounds=1)),
    ("No-QR (final)",     dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No refinement",     dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=0)),
    ("Full-NCWP-with-QR", dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=True,  retraction_interval=200, refine_knn_rounds=1)),
]

results = []
for variant_name, cfg in VARIANTS:
    row = {"variant": variant_name}
    print(f"\n[{variant_name}]")
    for dim in DIMS:
        print(f"  r={dim}...", end=" ", flush=True)
        if cfg == "pca":
            k = min(dim, W_pca_full.shape[1])
            W_t = torch.from_numpy(W_pca_full[:, :k]).to(DEVICE)
            mu_t = torch.from_numpy(mu_pca).to(DEVICE)
            def _pca_proj(X_np):
                parts = []
                for i in range(0, len(X_np), 2000):
                    xb = torch.from_numpy(X_np[i:i+2000]).to(DEVICE)
                    xp = F.normalize((xb - mu_t) @ W_t, dim=1)
                    parts.append(xp.half())
                return torch.cat(parts, dim=0)
            ndcg = evaluate_ndcg(_pca_proj(corpus_np), _pca_proj(query_np))
        elif cfg == "zca":
            k = min(dim, Vt_w.shape[0])
            W_zca = torch.from_numpy(Vt_w[:k].T.astype(np.float32)).to(DEVICE)
            mu_zca_w = torch.from_numpy(Xw_mu).to(DEVICE) @ W_zca
            def _zca_proj(X_np):
                parts = []
                for i in range(0, len(X_np), 2000):
                    xb = torch.from_numpy(X_np[i:i+2000]).to(DEVICE)
                    xw = F.normalize((xb - mu_zca) @ S_zca, dim=1)
                    xp = F.normalize(xw @ W_zca - mu_zca_w, dim=1)
                    parts.append(xp.half())
                return torch.cat(parts, dim=0)
            ndcg = evaluate_ndcg(_zca_proj(corpus_np), _zca_proj(query_np))
        else:
            W_np = train_ncwp(Xw_fit, knn_idx, knn_sims, rank=dim, **cfg)
            ndcg = project_and_eval(W_np, use_zca=True)
        row[f"r={dim}"] = round(ndcg, 2)
        print(f"nDCG@10={ndcg:.2f}")
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\n결과 저장: {OUT_CSV}")

print("\n" + "="*62)
print("T1(Quora): Ablation — Qwen-4B × Quora / nDCG@10")
print(f"Fit: N={FIT_N} queries | Eval: 10K queries × 522K corpus")
print("="*62)
print(df.to_string(index=False))

def g(name, r):
    row = df[df['variant']==name]
    return row[f'r={r}'].values[0] if len(row) else float('nan')

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
    print(f"  {'✅' if ok else '❌'} {desc}")
