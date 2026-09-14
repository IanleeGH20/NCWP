"""
STS Ablation — Main과 완전 동일한 protocol로 통일
  fit: valid 2,910 unique sentences (Main과 동일)
  eval: train+test 7,128 pairs (Main과 동일)
  model: Qwen-4B
  dims: [20, 80, 320]

12 Variants:
  Baselines (closed-form): PCA-whitening, Soft-White, Random, LPP, ZCA-only
  NCWP variants:           Random-pairs, No hard-neg, No memory bank, No-QR (final),
                           No refinement, QR + No refinement, Full-NCWP-with-QR
"""
import os, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr
from scipy.linalg import eigh as scipy_eigh

STS_DIR    = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR  = "/workspace/NCWP/sts_ablation_cache"
OUT_CSV    = "/workspace/NCWP/sts_ablation_unified_eval7128.csv"
DEVICE     = torch.device("cuda")
SEED       = 42
DIMS       = [20, 80, 320]
K_NEIGH    = 40
COSINE_TAU = 0.2

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── 데이터 ───────────────────────────────────────────────────
valid = json.load(open(f"{STS_DIR}/sts_valid.json"))
train = json.load(open(f"{STS_DIR}/sts_train.json"))
test  = json.load(open(f"{STS_DIR}/sts_test.json"))

fit_sents = list({s for item in valid for s in (item['sentence1'], item['sentence2'])})
eval_pairs  = [(item['sentence1'], item['sentence2']) for item in train + test]
eval_scores = [item['score'] for item in train + test]
eval_sents  = list({s for s1, s2 in eval_pairs for s in (s1, s2)})
all_sents   = list(set(fit_sents + eval_sents))
print(f"fit={len(fit_sents)}  eval_pairs={len(eval_pairs)}  all_unique={len(all_sents)}")

# 캐시 (Main 업데이트에서 사용한 것 재사용)
cache_emb = f"{CACHE_DIR}/qwen-4b_main_embs.npy"
cache_idx = f"{CACHE_DIR}/qwen-4b_main_sents.json"
embs = np.load(cache_emb).astype(np.float32)
saved = json.load(open(cache_idx))
sent2idx = {s: i for i, s in enumerate(saved)}
# 모든 fit_sents, eval_sents가 캐시에 있는지 확인
missing = [s for s in fit_sents + eval_sents if s not in sent2idx]
assert len(missing) == 0, f"캐시에 없는 문장 {len(missing)}개"
X_fit  = np.array([embs[sent2idx[s]] for s in fit_sents]).astype(np.float32)
X_eval = {s: embs[sent2idx[s]].astype(np.float32) for s in eval_sents}
print(f"X_fit={X_fit.shape}")

# ── ZCA ──────────────────────────────────────────────────────
def compute_zca(X, shrink=0.08):
    Xt = torch.from_numpy(X).to(DEVICE)
    mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(Xt.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    return mu, S

mu_zca, S_zca = compute_zca(X_fit)
def apply_zca_np(X):
    Xt = torch.from_numpy(X).to(DEVICE)
    return F.normalize((Xt - mu_zca) @ S_zca, dim=1).cpu().numpy()
Xw_fit = apply_zca_np(X_fit)
Xw_eval = {s: apply_zca_np(X_eval[s].reshape(1,-1))[0] for s in eval_sents}

# ── kNN ──────────────────────────────────────────────────────
print(f"kNN (k={K_NEIGH})...")
Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
_, knn_idx_t = torch.topk(sim_all, k=K_NEIGH, dim=1)
knn_idx  = knn_idx_t.cpu().numpy()
knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()

# ── Spearman ─────────────────────────────────────────────────
def spearman_score(emb_map):
    pred = [float(np.dot(emb_map[s1], emb_map[s2])) for s1, s2 in eval_pairs]
    return spearmanr(eval_scores, pred).correlation * 100

# ── Baselines (closed-form) ──────────────────────────────────
def pca_white(dim, shrink=0.08):
    Xt = torch.from_numpy(X_fit).to(DEVICE).double()
    N, D = Xt.shape; mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(N-1, 1)
    tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE, dtype=Xt.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(dim, D)
    W = evec[:, :k] @ torch.diag(1.0 / torch.sqrt(torch.clamp(ev[:k], min=1e-6)))
    W_np = W.cpu().numpy().astype(np.float32); mu_np = mu.cpu().numpy().astype(np.float32)
    emb_map = {}
    for s in eval_sents:
        z = (X_eval[s] - mu_np) @ W_np
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

def soft_white(dim, shrink=0.08):
    Xt = torch.from_numpy(X_fit).to(DEVICE).double()
    N, D = Xt.shape; mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(N-1, 1); tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE, dtype=Xt.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(dim, D)
    W = evec[:, :k] @ torch.diag(1.0 / torch.pow(torch.clamp(ev[:k], min=1e-6), 0.25))
    W_np = W.cpu().numpy().astype(np.float32); mu_np = mu.cpu().numpy().astype(np.float32)
    emb_map = {}
    for s in eval_sents:
        z = (X_eval[s] - mu_np) @ W_np
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

def random_proj(dim):
    rng = np.random.RandomState(SEED)
    D = X_fit.shape[1]
    R = rng.randn(D, dim).astype(np.float32) / np.sqrt(D)
    Q, _ = np.linalg.qr(R)
    emb_map = {}
    for s in eval_sents:
        z = X_eval[s] @ Q
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

def lpp_proj(dim, k=10):
    N, D = X_fit.shape
    Xn = X_fit / (np.linalg.norm(X_fit, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    knn = np.argsort(-sim, axis=1)[:, :k]
    W_adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in knn[i]:
            W_adj[i, j] = 1.0; W_adj[j, i] = 1.0
    D_diag = W_adj.sum(1)
    L = np.diag(D_diag) - W_adj
    XtX  = X_fit.T @ (D_diag[:, None] * X_fit)
    XtLX = X_fit.T @ L @ X_fit
    try:
        eigvals, eigvecs = scipy_eigh(XtLX, XtX)
        W = eigvecs[:, :dim]
    except:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(XtX) @ XtLX)
        W = eigvecs[:, :dim]
    emb_map = {}
    for s in eval_sents:
        z = X_eval[s] @ W
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

def zca_only_proj(dim, shrink=0.08):
    # ZCA whitening + PCA top-dim
    Xw = Xw_fit
    mu_w = Xw.mean(0); Xwc = Xw - mu_w
    _, _, Vt = np.linalg.svd(Xwc, full_matrices=False)
    k = min(dim, Vt.shape[0])
    W_pca = Vt[:k].T
    emb_map = {}
    for s in eval_sents:
        xw = Xw_eval[s] - mu_w
        z = xw @ W_pca
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── Memory Bank ──────────────────────────────────────────────
class MemoryBank:
    def __init__(self, size, dim):
        self.bank = torch.zeros(size, dim, device=DEVICE)
        self.ptr = 0; self.full = False; self.size = size
    def update(self, z):
        B = z.shape[0]; end = min(self.ptr + B, self.size); n1 = end - self.ptr
        self.bank[self.ptr:end] = z[:n1]
        if B > n1: self.bank[:B-n1] = z[n1:]; self.ptr = B - n1; self.full = True
        else:
            self.ptr = end % self.size
            if self.ptr == 0: self.full = True
    def get(self): return self.bank if self.full else self.bank[:self.ptr]

# ── NCWP 학습 ────────────────────────────────────────────────
def train_ncwp(rank, seed=SEED,
               temperature=0.07, lambda_cov=0.05, lambda_orth=0.02,
               lr=8e-3, max_epochs=20, batch_pairs=256,
               cosine_tau=COSINE_TAU, retraction_interval=0,
               refine_knn_rounds=1, random_pairs=False,
               use_memory_bank=False, use_hard_neg=False,
               hard_neg_k=32, memory_bank_size=4096):
    set_seed(seed)
    N, D = Xw_fit.shape; r = rank
    Xw   = torch.from_numpy(Xw_fit).to(DEVICE)
    knn  = torch.from_numpy(knn_idx).to(DEVICE)
    sims = torch.from_numpy(knn_sims.astype(np.float32)).to(DEVICE)
    B = batch_pairs
    steps = min(160, max(100, math.ceil(N / (2*B))))
    round_T = max_epochs * steps; lr_min = 0.2 * lr; warmup = 200
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    mbank = MemoryBank(memory_bank_size, r) if use_memory_bank else None
    overall_best_W = None; overall_best_loss = float('inf')

    def _one_round(knn_loc, sims_loc):
        # Round-local LR schedule (A안: 각 round 독립 학습)
        nonlocal overall_best_W, overall_best_loss
        round_gs = 0
        round_best_loss = float('inf')
        round_best_W = None
        no_imp = 0
        for ep in range(max_epochs):
            ep_loss = 0.0
            for _ in range(steps):
                if round_gs < warmup:
                    cur_lr = lr_min + (lr - lr_min) * round_gs / max(1, warmup)
                else:
                    t = (round_gs - warmup) / max(1, round_T - warmup)
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
                Z = F.normalize(batch @ W, dim=1)
                Za = Z[:B]; Zb = Z[B:]
                sim_a_to_b = (Za @ Zb.T) / temperature
                sim_a_to_a = (Za @ Za.T) / temperature; sim_a_to_a.fill_diagonal_(-1e9)
                sim_b_to_a = (Zb @ Za.T) / temperature
                sim_b_to_b = (Zb @ Zb.T) / temperature; sim_b_to_b.fill_diagonal_(-1e9)
                if use_memory_bank and mbank is not None and len(mbank.get()) > 0:
                    bank = mbank.get().detach()
                    sim_a_bank = (Za @ bank.T) / temperature
                    sim_b_bank = (Zb @ bank.T) / temperature
                else:
                    sim_a_bank = None; sim_b_bank = None
                pos_logit_ab = sim_a_to_b.diag()
                neg_a_inb = sim_a_to_b.clone(); neg_a_inb.fill_diagonal_(-1e9)
                neg_pool_a = torch.cat([neg_a_inb, sim_a_to_a], dim=1)
                if sim_a_bank is not None: neg_pool_a = torch.cat([neg_pool_a, sim_a_bank], dim=1)
                pos_logit_ba = sim_b_to_a.diag()
                neg_b_inb = sim_b_to_a.clone(); neg_b_inb.fill_diagonal_(-1e9)
                neg_pool_b = torch.cat([neg_b_inb, sim_b_to_b], dim=1)
                if sim_b_bank is not None: neg_pool_b = torch.cat([neg_pool_b, sim_b_bank], dim=1)
                if use_hard_neg:
                    K = min(hard_neg_k, neg_pool_a.shape[1])
                    neg_a_sel, _ = torch.topk(neg_pool_a, k=K, dim=1)
                    neg_b_sel, _ = torch.topk(neg_pool_b, k=K, dim=1)
                else:
                    neg_a_sel = neg_pool_a; neg_b_sel = neg_pool_b
                denom_a = torch.logsumexp(torch.cat([pos_logit_ab.unsqueeze(1), neg_a_sel], dim=1), dim=1)
                denom_b = torch.logsumexp(torch.cat([pos_logit_ba.unsqueeze(1), neg_b_sel], dim=1), dim=1)
                loss = 0.5*(-(pos_logit_ab - denom_a).mean() + -(pos_logit_ba - denom_b).mean())
                Zm = Z - Z.mean(0)
                cov = (Zm.T @ Zm) / max(len(Z)-1,1)
                off = cov - torch.diag(torch.diag(cov))
                loss += lambda_cov*(off**2).sum()/r
                Wn = F.normalize(W, dim=0)
                loss += lambda_orth*((Wn.T @ Wn - torch.eye(r,device=DEVICE))**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if use_memory_bank and mbank is not None:
                    mbank.update(torch.cat([Za, Zb], dim=0).detach())
                if retraction_interval > 0 and (round_gs+1) % retraction_interval == 0:
                    with torch.no_grad():
                        Q, _ = torch.linalg.qr(W.data); W.data = Q[:, :r]
                ep_loss += loss.item(); round_gs += 1
            ep_loss /= steps
            if ep_loss + 1e-6 < round_best_loss:
                round_best_loss = ep_loss; no_imp = 0
                round_best_W = W.detach().clone()
                # overall best W도 갱신
                if ep_loss < overall_best_loss:
                    overall_best_loss = ep_loss
                    overall_best_W = round_best_W.clone()
            else:
                no_imp += 1
                if no_imp >= 6: break

    _one_round(knn, sims)
    for _ in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ overall_best_W, dim=1)
            sim_r = Zfull @ Zfull.T; sim_r.fill_diagonal_(-1e9)
            _, knn_new = torch.topk(sim_r, k=K_NEIGH, dim=1)
            sims_new = sim_r.gather(1, knn_new)
            W.data.copy_(overall_best_W)
        # optimizer 재생성 (round-local 학습 위해, Adam momentum 초기화)
        opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
        if mbank: mbank.ptr=0; mbank.full=False
        _one_round(knn_new, sims_new)
    return overall_best_W.cpu().numpy()

def ncwp_eval(W_np):
    mu_w = Xw_fit.mean(0) @ W_np
    emb_map = {}
    for s in eval_sents:
        z = Xw_eval[s] @ W_np - mu_w
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── 12 Variants ──────────────────────────────────────────────
VARIANTS = [
    ("PCA-whitening",     "closed", pca_white),
    ("Soft-White",        "closed", soft_white),
    ("Random",            "closed", random_proj),
    ("LPP",               "closed", lpp_proj),
    ("ZCA-only",          "closed", zca_only_proj),
    ("Random-pairs",      "ncwp", dict(random_pairs=True,  use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No hard-neg",       "ncwp", dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No memory bank",    "ncwp", dict(random_pairs=False, use_memory_bank=False, use_hard_neg=True,  retraction_interval=0,   refine_knn_rounds=1)),
    ("No-QR (final)",     "ncwp", dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No refinement",     "ncwp", dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=0)),
    ("QR + No refinement","ncwp", dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=True,  retraction_interval=200, refine_knn_rounds=0)),
    ("Full-NCWP-with-QR", "ncwp", dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=True,  retraction_interval=200, refine_knn_rounds=1)),
]

results = []
for name, kind, cfg in VARIANTS:
    row = {"variant": name}
    print(f"\n[{name}]")
    for d in DIMS:
        print(f"  r={d}...", end=" ", flush=True)
        if kind == "closed":
            emb_map = cfg(d)
        else:
            W_np = train_ncwp(rank=d, **cfg)
            emb_map = ncwp_eval(W_np)
        sp = spearman_score(emb_map)
        row[f"r={d}"] = round(sp, 2)
        print(f"Spearman={sp:.2f}")
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\n저장: {OUT_CSV}")
print(df.to_string(index=False))
