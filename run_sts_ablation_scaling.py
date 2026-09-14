"""
STS Ablation Scaling — N별 fit 데이터 크기 비교
fit pool: train+valid+test 합집합 15,487 → N개 random sample
eval: test 1,379 쌍 (고정)

6 NCWP variants:
  Random-pairs, No hard-neg, No memory bank, No-QR, No refinement, Full-NCWP-with-QR
(PCA/ZCA는 기존 결과에서 가져옴)
"""
import os, sys, json, math, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr

argp = argparse.ArgumentParser()
argp.add_argument("--N", type=int, required=True)
args = argp.parse_args()

STS_DIR   = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
OUT_CSV   = f"/workspace/NCWP/sts_ablation_N{args.N}.csv"
DEVICE    = torch.device("cuda")
SEED      = 42
FIT_N     = args.N

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── 데이터 ───────────────────────────────────────────────────
def load_json(p):
    with open(p) as f: return json.load(f)

test_data  = load_json(f"{STS_DIR}/sts_test.json")
test_pairs  = [(item['sentence1'], item['sentence2']) for item in test_data]
test_scores = [item['score'] for item in test_data]

with open(f"{CACHE_DIR}/qwen-4b_full_sents.json") as f:
    full_sents = json.load(f)
full_embs = np.load(f"{CACHE_DIR}/qwen-4b_full_embs.npy").astype(np.float32)
sent2idx = {s: i for i, s in enumerate(full_sents)}

# fit: 전체 풀에서 N개 random sample (seed 고정)
rng = np.random.RandomState(SEED)
fit_idx = rng.choice(len(full_sents), size=FIT_N, replace=False)
X_fit = full_embs[fit_idx]
print(f"fit pool={len(full_sents)}, sample={len(X_fit)}")
print(f"eval pairs={len(test_pairs)}")

# eval 임베딩 추출
eval_sents = list({s for s1, s2 in test_pairs for s in (s1, s2)})
X_eval = {s: full_embs[sent2idx[s]] for s in eval_sents}

# ── 평가 ─────────────────────────────────────────────────────
def spearman_score(emb_map):
    pred = [float(np.dot(emb_map.get(s1, np.zeros(1)), emb_map.get(s2, np.zeros(1))))
            for s1, s2 in test_pairs]
    return spearmanr(test_scores, pred).correlation * 100

# ── ZCA ─────────────────────────────────────────────────────
def compute_zca(X, shrink=0.08):
    Xt = torch.from_numpy(X).to(DEVICE)
    mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(Xt.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    return mu.cpu().numpy(), S.cpu().numpy()

def apply_zca(X, mu, S):
    return F.normalize(torch.from_numpy(
        (X - mu).astype(np.float32) @ S.astype(np.float32)).to(DEVICE), dim=1).cpu().numpy()

print("ZCA 계산...")
mu_zca, S_zca = compute_zca(X_fit)
Xw_fit  = apply_zca(X_fit, mu_zca, S_zca)
Xw_eval = {s: apply_zca(X_eval[s].reshape(1,-1), mu_zca, S_zca)[0] for s in eval_sents}

# kNN
K_NEIGH    = 40
COSINE_TAU = 0.2
print(f"kNN (k={K_NEIGH}, tau={COSINE_TAU})...")
Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
_, knn_idx_t = torch.topk(sim_all, k=K_NEIGH, dim=1)
knn_idx  = knn_idx_t.cpu().numpy()
knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()

# Memory Bank
class MemoryBank:
    def __init__(self, size, dim):
        self.bank = torch.zeros(size, dim, device=DEVICE)
        self.ptr = 0; self.full = False; self.size = size
    def update(self, z):
        B = z.shape[0]
        end = min(self.ptr + B, self.size); n1 = end - self.ptr
        self.bank[self.ptr:end] = z[:n1]
        if B > n1:
            self.bank[:B-n1] = z[n1:]; self.ptr = B - n1; self.full = True
        else:
            self.ptr = end % self.size
            if self.ptr == 0: self.full = True
    def get(self):
        return self.bank if self.full else self.bank[:self.ptr]

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
    T_total = max_epochs * steps; lr_min = 0.2 * lr; warmup = 200
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

def make_emb_map(W_np):
    Xw_mean = Xw_fit.mean(0)
    mu_w = Xw_mean @ W_np
    out = {}
    for s in eval_sents:
        z = Xw_eval[s] @ W_np - mu_w
        out[s] = z / (np.linalg.norm(z) + 1e-12)
    return out

DIMS = [20, 80, 320]
VARIANTS = [
    ("Random-pairs",      dict(random_pairs=True,  use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No hard-neg",       dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No memory bank",    dict(random_pairs=False, use_memory_bank=False, use_hard_neg=True,  retraction_interval=0,   refine_knn_rounds=1)),
    ("No-QR (final)",     dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=1)),
    ("No refinement",     dict(random_pairs=False, use_memory_bank=False, use_hard_neg=False, retraction_interval=0,   refine_knn_rounds=0)),
    ("Full-NCWP-with-QR", dict(random_pairs=False, use_memory_bank=True,  use_hard_neg=True,  retraction_interval=200, refine_knn_rounds=1)),
]

results = []
for name, cfg in VARIANTS:
    row = {"variant": name}
    print(f"\n[{name}]")
    for d in DIMS:
        print(f"  r={d}...", end=" ", flush=True)
        W_np = train_ncwp(Xw_fit, knn_idx, knn_sims, rank=d, **cfg)
        sp = spearman_score(make_emb_map(W_np))
        row[f"r={d}"] = round(sp, 2)
        print(f"Spearman={sp:.2f}")
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\n저장: {OUT_CSV}")
print(df.to_string(index=False))
