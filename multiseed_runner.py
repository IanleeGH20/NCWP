"""
Multi-seed runner — 단일 실험(model, dataset, dim, seed)을 빠르게 측정.
- STS / Quora 둘 다 지원
- Method: NCWP "QR + No refinement" (memory bank + hard-neg + QR ON, refinement OFF)
- 캐시된 임베딩 사용

Usage:
  python3 multiseed_runner.py --dataset sts   --model qwen-4b  --dim 320 --seed 42
  python3 multiseed_runner.py --dataset quora --model qwen-4b  --dim 80  --seed 42
  python3 multiseed_runner.py --dataset sts   --model llama-8b --dim 512 --seed 42
"""
import os, json, math, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from datasets import load_dataset

argp = argparse.ArgumentParser()
argp.add_argument("--dataset", choices=["sts", "quora"], required=True)
argp.add_argument("--model", required=True)
argp.add_argument("--dim", type=int, required=True)
argp.add_argument("--seed", type=int, required=True)
argp.add_argument("--N", type=int, default=1000, help="fit sample size for Quora")
argp.add_argument("--variant", choices=["qr_norefine", "zca_contrastive"], default="qr_norefine",
                  help="qr_norefine = NCWP final (memory bank+hard-neg+QR, no refine); "
                       "zca_contrastive = No-QR (ZCA + plain contrastive, no memory bank/hard-neg/QR)")
args = argp.parse_args()

DEVICE = torch.device("cuda")
SEED = args.seed
K_NEIGH = 40
COSINE_TAU = 0.2

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)


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


def train_ncwp_qr_norefine(Xw_np, knn_idx_np, knn_sims_np, rank, seed,
                            use_memory_bank=True, use_hard_neg=True,
                            retraction_interval=200, refine_knn_rounds=0):
    """NCWP variant (기본: QR + No refinement)"""
    set_seed(seed)
    N, D = Xw_np.shape; r = rank
    Xw = torch.from_numpy(Xw_np.astype(np.float32)).to(DEVICE)
    knn = torch.from_numpy(knn_idx_np).to(DEVICE)
    sims = torch.from_numpy(knn_sims_np.astype(np.float32)).to(DEVICE)
    
    temperature = 0.07; lambda_cov=0.05; lambda_orth=0.02
    lr = 8e-3; lr_min = 0.2 * lr; warmup = 200
    max_epochs = 20; batch_pairs = 256
    hard_neg_k = 32; memory_bank_size = 4096
    
    B = batch_pairs
    steps = min(160, max(100, math.ceil(N / (2*B))))
    round_T = max_epochs * steps
    
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    mbank = MemoryBank(memory_bank_size, r) if use_memory_bank else None
    
    best_W = None; best_loss = float('inf'); no_imp = 0; gs = 0
    
    for ep in range(max_epochs):
        ep_loss = 0.0
        for _ in range(steps):
            if gs < warmup:
                cur_lr = lr_min + (lr - lr_min) * gs / max(1, warmup)
            else:
                t = (gs - warmup) / max(1, round_T - warmup)
                cur_lr = lr_min + 0.5*(lr - lr_min)*(1+math.cos(math.pi*t))
            for pg in opt.param_groups: pg['lr'] = cur_lr
            
            with torch.no_grad():
                a_idx = torch.randint(0, N, (B,), device=DEVICE)
                b_list = []
                for anc in a_idx.tolist():
                    valid = (sims[anc] >= COSINE_TAU).nonzero(as_tuple=True)[0]
                    if len(valid) == 0:
                        valid = torch.arange(knn.shape[1], device=DEVICE)
                    pick = valid[torch.randint(0, len(valid), (1,)).item()]
                    b_list.append(knn[anc, pick].item())
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
                sim_a_bank = sim_b_bank = None
            
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
                neg_a_sel = neg_pool_a
                neg_b_sel = neg_pool_b
            
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
    
    return best_W.cpu().numpy()


# ─── ZCA + kNN 공통 ────────────────────────────────────────
def compute_zca(X, shrink=0.08):
    Xt = torch.from_numpy(X).to(DEVICE)
    mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(Xt.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    return mu, S


# ─── 데이터셋 분기 ─────────────────────────────────────────
if args.dataset == "sts":
    from scipy.stats import spearmanr
    STS_DIR = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
    CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
    
    valid = json.load(open(f"{STS_DIR}/sts_valid.json"))
    train = json.load(open(f"{STS_DIR}/sts_train.json"))
    test  = json.load(open(f"{STS_DIR}/sts_test.json"))
    
    fit_sents = list({s for item in valid for s in (item['sentence1'], item['sentence2'])})
    eval_pairs  = [(item['sentence1'], item['sentence2']) for item in train + test]
    eval_scores = [item['score'] for item in train + test]
    eval_sents  = list({s for s1, s2 in eval_pairs for s in (s1, s2)})
    
    # 캐시
    cache_emb = f"{CACHE_DIR}/{args.model}_main_embs.npy"
    cache_idx = f"{CACHE_DIR}/{args.model}_main_sents.json"
    embs = np.load(cache_emb).astype(np.float32)
    saved = json.load(open(cache_idx))
    sent2idx = {s: i for i, s in enumerate(saved)}
    X_fit  = np.array([embs[sent2idx[s]] for s in fit_sents]).astype(np.float32)
    X_eval = {s: embs[sent2idx[s]].astype(np.float32) for s in eval_sents}
    
    print(f"[STS/{args.model}/r={args.dim}/seed={args.seed}] fit={X_fit.shape}, eval_pairs={len(eval_pairs)}")
    
    mu_zca, S_zca = compute_zca(X_fit)
    def apply_zca(X):
        Xt = torch.from_numpy(X).to(DEVICE)
        return F.normalize((Xt - mu_zca) @ S_zca, dim=1).cpu().numpy()
    Xw_fit = apply_zca(X_fit)
    Xw_eval = {s: apply_zca(X_eval[s].reshape(1,-1))[0] for s in eval_sents}
    
    Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
    sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
    _, knn_idx_t = torch.topk(sim_all, k=K_NEIGH, dim=1)
    knn_idx = knn_idx_t.cpu().numpy()
    knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()
    
    if args.variant == "zca_contrastive":
        W_np = train_ncwp_qr_norefine(Xw_fit, knn_idx, knn_sims, args.dim, args.seed,
                                       use_memory_bank=False, use_hard_neg=False,
                                       retraction_interval=0, refine_knn_rounds=1)
    else:
        W_np = train_ncwp_qr_norefine(Xw_fit, knn_idx, knn_sims, args.dim, args.seed)
    
    mu_w = Xw_fit.mean(0) @ W_np
    emb_map = {}
    for s in eval_sents:
        z = Xw_eval[s] @ W_np - mu_w
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    pred = [float(np.dot(emb_map[s1], emb_map[s2])) for s1, s2 in eval_pairs]
    score = spearmanr(eval_scores, pred).correlation * 100
    
    print(f"RESULT  dataset=sts  model={args.model}  dim={args.dim}  seed={args.seed}  variant={args.variant}  Spearman={score:.4f}")

else:  # quora
    CACHE = f'/workspace/NCWP/quora_results/embedding_cache/{args.model}'
    corpus_np = np.load(f'{CACHE}/corpus_base.npy').astype(np.float32)
    query_np  = np.load(f'{CACHE}/test_queries_base.npy').astype(np.float32)
    X_fit     = np.load(f'{CACHE}/fit_query_sample_N{args.N}.npy').astype(np.float32)
    
    print(f"[Quora/{args.model}/r={args.dim}/N={args.N}/seed={args.seed}] corpus={corpus_np.shape}, fit={X_fit.shape}")
    
    ds_corp = load_dataset("mteb/quora", "corpus",  split="corpus")
    ds_qs   = load_dataset("mteb/quora", "queries", split="queries")
    ds_qrel = load_dataset("mteb/quora", split="test")
    corpus_ids = [str(r['_id']) for r in ds_corp]
    qid2tx     = {str(r['_id']): r['text'] for r in ds_qs}
    qrels = defaultdict(set)
    for r in ds_qrel: qrels[str(r['query-id'])].add(str(r['corpus-id']))
    test_qids = sorted([q for q in qrels if q in qid2tx])
    
    mu_zca, S_zca = compute_zca(X_fit)
    def apply_zca_np(X):
        Xt = torch.from_numpy(X).to(DEVICE)
        return F.normalize((Xt - mu_zca) @ S_zca, dim=1).cpu().numpy()
    Xw_fit = apply_zca_np(X_fit)
    
    Xw_t = torch.from_numpy(Xw_fit).to(DEVICE)
    sim_all = Xw_t @ Xw_t.T; sim_all.fill_diagonal_(-1e9)
    _, knn_idx_t = torch.topk(sim_all, k=K_NEIGH, dim=1)
    knn_idx = knn_idx_t.cpu().numpy()
    knn_sims = sim_all.gather(1, knn_idx_t).cpu().numpy()
    
    W_np = train_ncwp_qr_norefine(Xw_fit, knn_idx, knn_sims, args.dim, args.seed)
    
    # 평가
    Xw_mean_t = torch.from_numpy(Xw_fit.mean(0)).to(DEVICE)
    W_t = torch.from_numpy(W_np).to(DEVICE)
    mu_w = Xw_mean_t @ W_t
    def _proj(Xnp, chunk=2000):
        parts = []
        for i in range(0, len(Xnp), chunk):
            xb = torch.from_numpy(Xnp[i:i+chunk]).to(DEVICE)
            xw = F.normalize((xb - mu_zca) @ S_zca, dim=1)
            xp = F.normalize(xw @ W_t - mu_w, dim=1)
            parts.append(xp.half())
        return torch.cat(parts, dim=0)
    c_t = _proj(corpus_np); q_t = _proj(query_np)
    
    ndcg10 = []
    for qi in range(0, len(test_qids), 500):
        qe = min(len(test_qids), qi+500)
        with torch.no_grad():
            sim = (q_t[qi:qe] @ c_t.T).float()
            _, top_idx = torch.topk(sim, k=10, dim=1)
            top10 = top_idx.cpu().numpy()
        for ci, q2 in enumerate(range(qi, qe)):
            qid = test_qids[q2]
            if qid not in qrels: continue
            rel = qrels[qid]
            ranked = [corpus_ids[i] for i in top10[ci]]
            h10 = [1.0 if d in rel else 0.0 for d in ranked]
            dcg = sum(h/np.log2(r+2) for r,h in enumerate(h10))
            idcg = sum(1.0/np.log2(r+2) for r in range(min(len(rel),10)))
            ndcg10.append(dcg/idcg if idcg>0 else 0.0)
    score = np.mean(ndcg10)*100
    
    print(f"RESULT  dataset=quora  model={args.model}  dim={args.dim}  N={args.N}  seed={args.seed}  nDCG@10={score:.4f}")
