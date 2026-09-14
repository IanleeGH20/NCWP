"""
Update Main STS NCWP results with new v4 implementation.
  - fit: valid 2,910 sentences (기존 Main과 동일)
  - eval: train+test 7,128 pairs (기존 Main과 동일)
  - NCWP final = "QR + No refinement" variant
    (use_memory_bank=True, use_hard_neg=True, retraction_interval=200, refine_knn_rounds=0)
  - Hyperparameters: k=40, tau=0.2, temp=0.07, warmup=200

기존 결과 CSV: /workspace/RAG/code/Make_embedding/nanoGPT/ncwp_extended_results/{model}_results_main.csv
→ NCWP 행만 새 값으로 덮어쓰기. PCA/ZCA 등 closed-form은 그대로 유지.
"""
import os, sys, json, math, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

argp = argparse.ArgumentParser()
argp.add_argument("--model", choices=["qwen-4b", "qwen-8b", "llama-8b"], required=True)
args = argp.parse_args()

STS_DIR    = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR  = "/workspace/NCWP/sts_ablation_cache"
RESULTS_DIR= "/workspace/RAG/code/Make_embedding/nanoGPT/ncwp_extended_results"
DEVICE     = torch.device("cuda")
SEED       = 42

MODELS = {
    "qwen-4b":  {"hf":"Qwen/Qwen1.5-4B",              "base_dim":2560, "dims":[5,10,20,40,80,160,320,640,1280]},
    "qwen-8b":  {"hf":"Qwen/Qwen2-7B",                "base_dim":3584, "dims":[7,14,28,56,112,224,448,896,1792]},
    "llama-8b": {"hf":"meta-llama/Meta-Llama-3.1-8B", "base_dim":4096, "dims":[4,8,16,32,64,128,256,512,1024,2048]},
}

cfg_m = MODELS[args.model]

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── 데이터 ───────────────────────────────────────────────────
print(f"[{args.model}] STS 데이터 로드...")
valid = json.load(open(f"{STS_DIR}/sts_valid.json"))
train = json.load(open(f"{STS_DIR}/sts_train.json"))
test  = json.load(open(f"{STS_DIR}/sts_test.json"))

# fit = valid의 unique sentences
fit_sents = list({s for item in valid for s in (item['sentence1'], item['sentence2'])})
# eval = train + test pairs (기존 Main과 동일)
eval_pairs  = [(item['sentence1'], item['sentence2']) for item in train + test]
eval_scores = [item['score'] for item in train + test]
eval_sents = list({s for s1, s2 in eval_pairs for s in (s1, s2)})
all_sents  = list(set(fit_sents + eval_sents))
print(f"  fit_sents={len(fit_sents)}  eval_pairs={len(eval_pairs)}  all_unique={len(all_sents)}")

# ── 임베딩 (캐시) ─────────────────────────────────────────────
cache_emb  = f"{CACHE_DIR}/{args.model}_main_embs.npy"
cache_idx  = f"{CACHE_DIR}/{args.model}_main_sents.json"

if os.path.exists(cache_emb) and os.path.exists(cache_idx):
    print("  Main 임베딩 캐시 로드...")
    saved = json.load(open(cache_idx))
    if saved == all_sents:
        embs = np.load(cache_emb).astype(np.float32)
    else:
        os.remove(cache_emb); os.remove(cache_idx)
        embs = None
else:
    embs = None

# qwen-4b는 기존 full_embs(15,487) 캐시 활용 가능
if embs is None and args.model == "qwen-4b":
    print("  기존 qwen-4b_full_embs 캐시에서 추출...")
    full_embs = np.load(f"{CACHE_DIR}/qwen-4b_full_embs.npy").astype(np.float32)
    full_sents = json.load(open(f"{CACHE_DIR}/qwen-4b_full_sents.json"))
    s2i = {s: i for i, s in enumerate(full_sents)}
    embs = np.array([full_embs[s2i[s]] for s in all_sents]).astype(np.float32)
    np.save(cache_emb, embs); json.dump(all_sents, open(cache_idx, 'w'))

if embs is None:
    print(f"  Qwen/Llama 모델 로드 + 임베딩 ({len(all_sents)}개)...")
    tok = AutoTokenizer.from_pretrained(cfg_m['hf'], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True,
          "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    n_gpu = torch.cuda.device_count()
    model = (torch.nn.DataParallel(AutoModel.from_pretrained(cfg_m['hf'], **kw)).cuda()
             if n_gpu > 1 else AutoModel.from_pretrained(cfg_m['hf'], **kw).to(DEVICE))
    model.eval()
    parts = []
    for i in range(0, len(all_sents), 32):
        batch = all_sents[i:i+32]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc) if n_gpu <= 1 else model.module(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        parts.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        if (i//32) % 20 == 0: print(f"    {i+len(batch)}/{len(all_sents)}", end="\r")
    print()
    embs = np.vstack(parts)
    np.save(cache_emb, embs); json.dump(all_sents, open(cache_idx, 'w'))
    del model; torch.cuda.empty_cache()

sent2idx = {s: i for i, s in enumerate(all_sents)}
X_fit  = np.array([embs[sent2idx[s]] for s in fit_sents]).astype(np.float32)
X_eval = {s: embs[sent2idx[s]].astype(np.float32) for s in eval_sents}
print(f"  X_fit={X_fit.shape}")

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
K_NEIGH = 40; COSINE_TAU = 0.2
print(f"  kNN(k={K_NEIGH})...")
Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
_, knn_idx_t = torch.topk(sim_all, k=K_NEIGH, dim=1)
knn_idx  = knn_idx_t.cpu().numpy()
knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()

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
def train_ncwp(rank, seed=SEED, temperature=0.07, lambda_cov=0.05, lambda_orth=0.02,
               lr=8e-3, max_epochs=20, batch_pairs=256,
               cosine_tau=COSINE_TAU, retraction_interval=200,
               refine_knn_rounds=0,  # ← "QR + No refinement"
               use_memory_bank=True, use_hard_neg=True,
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
            opt.state = {}
        if mbank: mbank.ptr=0; mbank.full=False
        _one_round(knn_new, sims_new)
    return overall_best_W.cpu().numpy()

# ── eval (Spearman) ──────────────────────────────────────────
def evaluate(W_np):
    Xw_mean = Xw_fit.mean(0); mu_w = Xw_mean @ W_np
    emb_map = {}
    for s in eval_sents:
        z = Xw_eval[s] @ W_np - mu_w
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    pred = [float(np.dot(emb_map[s1], emb_map[s2])) for s1, s2 in eval_pairs]
    return spearmanr(eval_scores, pred).correlation * 100

# ── dim별 측정 ───────────────────────────────────────────────
new_rows = []
for r in cfg_m['dims']:
    print(f"  NCWP r={r}...", end=" ", flush=True)
    W_np = train_ncwp(rank=r)
    sp = evaluate(W_np)
    new_rows.append({'dim': r, 'method': 'NCWP', 'spearman': sp, 'anisotropy': float('nan'), 'self_sim': float('nan')})
    print(f"Spearman={sp:.2f}")

# ── CSV 업데이트 ────────────────────────────────────────────
csv_path = f"{RESULTS_DIR}/{args.model}_results_main.csv"
df = pd.read_csv(csv_path)
# 기존 NCWP 백업
df_old = df[df['method']=='NCWP'].copy()
df_old.to_csv(f"{csv_path}.backup_old_ncwp.csv", index=False)
# NCWP 제거 후 새 결과 추가
df = df[df['method'] != 'NCWP']
df_new = pd.DataFrame(new_rows)
df = pd.concat([df, df_new], ignore_index=True)
df.to_csv(csv_path, index=False)
print(f"\n저장: {csv_path}")
print("\n=== Before vs After ===")
for r in cfg_m['dims']:
    old = float(df_old[df_old['dim']==r]['spearman'].values[0])
    new_v = float(df_new[df_new['dim']==r]['spearman'].values[0])
    diff = new_v - old
    print(f"  r={r:>5}  OLD={old:6.2f}  NEW={new_v:6.2f}  Δ={diff:+.2f}")
