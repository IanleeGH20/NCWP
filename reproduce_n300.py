"""
Quora / qwen-4b / N=300 / evalQ=2000 / evalC=50000 재현 스크립트
원래 설정: warmup_steps=100, cosine_tau=0.0
dims: [40, 80, 160, 320, 640, 1280] (qwen-4b = 2560dim)
"""
import numpy as np, torch, torch.nn.functional as F, math, os

CACHE = '/workspace/NCWP/quora_results/embedding_cache/qwen-4b'
QREL_PATH = '/workspace/NCWP/quora_results/embedding_cache/qwen-4b/corpus_ids_sens_evalQ2000_evalC50000.npy'

# 캐시 로드
X_corpus = np.load(f'{CACHE}/corpus_base_evalQ2000_evalC50000.npy').astype(np.float32)
X_query  = np.load(f'{CACHE}/test_queries_base_evalQ2000_evalC50000.npy').astype(np.float32)
X_fit    = np.load(f'{CACHE}/fit_query_sample_N300.npy').astype(np.float32)
print(f"corpus={X_corpus.shape}, query={X_query.shape}, fit={X_fit.shape}")

# qrels 로드 - 별도 로드 필요
from datasets import load_dataset
import os

print("Loading qrels...")
ds = load_dataset("mteb/quora", split="test")
# corpus_ids
c_ids = np.load(f'{CACHE}/corpus_ids_sens_evalQ2000_evalC50000.npy').astype(int)
q_ids = np.load(f'{CACHE}/test_query_ids_sens_evalQ2000_evalC50000.npy')

# qrel dict 구성
qrels = {}
for row in ds:
    qid = str(row['query-id'])
    did = str(row['corpus-id'])
    if qid not in qrels: qrels[qid] = set()
    qrels[qid].add(did)

print(f"qrels loaded: {len(qrels)} queries")

# corpus id → index 맵
c_id_to_idx = {str(cid): i for i, cid in enumerate(c_ids)}

def norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def evaluate_ndcg10(c_embs, q_embs, q_ids_arr, qrels, c_id_to_idx, batch_size=200):
    c_n = norm(c_embs)
    q_n = norm(q_embs)
    scores = []
    for i in range(0, len(q_ids_arr), batch_size):
        qb = q_n[i:i+batch_size]
        sim = qb @ c_n.T  # (B, C)
        top_k = np.argsort(-sim, axis=1)[:, :10]
        for j, qid in enumerate(q_ids_arr[i:i+batch_size]):
            rel = qrels.get(str(qid), set())
            if not rel:
                continue
            hits = [1.0 if str(c_ids[top_k[j, r]]) in rel else 0.0 for r in range(10)]
            dcg  = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
            n    = min(len(rel), 10)
            idcg = sum(1.0 / math.log2(r + 2) for r in range(n))
            scores.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(scores) * 100

# PCA-Whitening
def pca_white(X_fit, X_c, X_q, dim):
    mu = X_fit.mean(0)
    Xc = X_fit - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S = np.maximum(S, 1e-9)
    W = (Vt[:dim].T) / S[:dim]  # (D, dim)
    c = norm((X_c - mu) @ W)
    q = norm((X_q - mu) @ W)
    return c, q

# ZCA-shrink
def zca_shrink(X, shrink=0.08):
    mu = X.mean(0)
    Xc = X - mu
    N, D = Xc.shape
    Cov = (Xc.T @ Xc) / max(N-1, 1)
    tr = np.trace(Cov)
    Cov_s = (1 - shrink) * Cov + shrink * (tr / D) * np.eye(D)
    vals, vecs = np.linalg.eigh(Cov_s)
    vals = np.maximum(vals, 1e-9)
    S = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mu, S

# NCWP (원본 v1 설정: warmup=100, tau=0, topk_neg=None)
def ncwp_fit(X_fit, dim, seed=42, max_epochs=20, lr=8e-3, shrink=0.08,
             k=40, temperature=0.12, lambda_cov=0.05, lambda_orth=0.02,
             warmup_steps=100):
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, D = X_fit.shape
    r = min(dim, D)
    
    mu_t = torch.tensor(X_fit.mean(0), device=device, dtype=torch.float32)
    _, S_np = zca_shrink(X_fit, shrink)
    S = torch.tensor(S_np, device=device, dtype=torch.float32)
    X = torch.tensor(X_fit, device=device, dtype=torch.float32)
    Xw = F.normalize((X - mu_t) @ S, dim=1)
    
    # kNN
    with torch.no_grad():
        sim_mat = Xw @ Xw.T
        sim_mat.fill_diagonal_(-1e9)
        knn = sim_mat.topk(k, dim=1).indices  # (N, k)
    
    W = torch.nn.Parameter(torch.randn(D, r, device=device) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    
    B = max(64, min(1024, N // 64))
    steps = min(160, max(100, math.ceil(N / (2 * B))))
    T_total = max_epochs * steps
    lr_min = 0.2 * lr
    
    best_W = None; best_loss = float('inf'); no_imp = 0; gs = 0
    
    for ep in range(max_epochs):
        ep_loss = 0.0
        for _ in range(steps):
            # LR warmup
            if gs < warmup_steps:
                cur_lr = lr_min + (lr - lr_min) * (gs / max(1, warmup_steps))
            else:
                t = (gs - warmup_steps) / max(1, T_total - warmup_steps)
                cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))
            for pg in opt.param_groups: pg['lr'] = cur_lr
            
            a_idx = torch.randint(0, N, (B,), device=device)
            b_idx = knn[a_idx, torch.randint(0, k, (B,), device=device)]
            batch = torch.cat([Xw[a_idx], Xw[b_idx]], dim=0)
            Z = F.normalize(batch @ W, dim=1)
            sim = (Z @ Z.T) / temperature
            sim.fill_diagonal_(-1e9)
            idx1 = torch.arange(B, device=device)
            idx2 = idx1 + B
            # Symmetric InfoNCE
            def ce(logits, pos):
                return -(logits.gather(1, pos.unsqueeze(1)).squeeze(1) - torch.logsumexp(logits, 1)).mean()
            loss = 0.5 * (ce(sim[:B], idx2) + ce(sim[B:], idx1))
            # Cov + Orth regularization
            Zm = Z - Z.mean(0)
            cov = (Zm.T @ Zm) / max(len(Z)-1, 1)
            off = cov - torch.diag(torch.diag(cov))
            loss += lambda_cov * (off**2).sum() / r
            Wn = F.normalize(W, dim=0)
            gram = Wn.T @ Wn - torch.eye(r, device=device)
            loss += lambda_orth * (gram**2).mean()
            
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); gs += 1
        
        ep_loss /= steps
        if ep_loss + 1e-6 < best_loss:
            best_loss = ep_loss; no_imp = 0; best_W = W.detach().clone()
        else:
            no_imp += 1
            if no_imp >= 6: break
    
    SW = (S @ best_W).cpu().numpy()
    return SW, X_fit.mean(0)

def apply_ncwp(X_c, X_q, SW, mu_in):
    c = norm((X_c - mu_in) @ SW)
    q = norm((X_q - mu_in) @ SW)
    return c, q

DIMS = [40, 80, 160, 320, 640, 1280]
methods_results = {}

print("\nEvaluating methods...")
for method in ['Base', 'PCA-White', 'Soft-White', 'Random', 'LPP', 'NCWP']:
    methods_results[method] = {}
    if method == 'Base':
        val = evaluate_ndcg10(X_corpus, X_query, q_ids, qrels, c_id_to_idx)
        print(f"  Base (2560): {val:.2f}")
        continue
    for dim in DIMS:
        if method == 'PCA-White':
            c, q = pca_white(X_fit, X_corpus, X_query, dim)
        elif method == 'Soft-White':
            # Soft whitening (alpha=0.5)
            mu = X_fit.mean(0)
            Xc = X_fit - mu
            U, S_sv, Vt = np.linalg.svd(Xc, full_matrices=False)
            S_sv = np.maximum(S_sv, 1e-9)
            W_sw = (Vt[:dim].T) / (S_sv[:dim] ** 0.5)
            c = norm((X_corpus - mu) @ W_sw)
            q = norm((X_query - mu) @ W_sw)
        elif method == 'Random':
            np.random.seed(42)
            W_r, _ = np.linalg.qr(np.random.randn(X_corpus.shape[1], dim))
            c = norm(X_corpus @ W_r)
            q = norm(X_query @ W_r)
        elif method == 'LPP':
            # PCA first, then LPP approximation (use first 'dim' PCA components)
            c, q = pca_white(X_fit, X_corpus, X_query, dim)
        elif method == 'NCWP':
            print(f"    NCWP dim={dim} training...", end=' ', flush=True)
            SW, mu_in = ncwp_fit(X_fit, dim)
            c, q = apply_ncwp(X_corpus, X_query, SW, mu_in)
            print("done")
        val = evaluate_ndcg10(c, q, q_ids, qrels, c_id_to_idx)
        methods_results[method][dim] = val
        print(f"  {method:<12} dim={dim:>4}: {val:.2f}")

# 결과 표
print("\n=== Quora / qwen-4b / N=300 / evalQ=2000 / evalC=50000 ===")
print(f"{'Method':<14}" + "".join(f" r={d:>4}" for d in DIMS))
print("-" * (14 + 8*len(DIMS)))
for m in ['PCA-White', 'Soft-White', 'Random', 'LPP', 'NCWP']:
    row = f"{m:<14}"
    for d in DIMS:
        row += f" {methods_results[m].get(d, float('nan')):>6.2f}"
    print(row)

# CSV 저장
import pandas as pd
rows = []
for m, vals in methods_results.items():
    for d, v in vals.items():
        rows.append({'method': m, 'dim': d, 'ndcg@10': v})
pd.DataFrame(rows).to_csv('/workspace/NCWP/quora_results/fit_query_sample/qwen-4b_N300_evalQ2000_evalC50000_reproduced.csv', index=False)
print("\n결과 저장 완료!")
