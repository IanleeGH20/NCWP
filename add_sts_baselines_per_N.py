"""
STS scaling N별 baseline (PCA-White, Soft-White, Random, LPP, ZCA-only) 계산
NCWP scaling 실험과 동일한 N별 fit pool 사용.
결과를 sts_ablation_N{N}.csv에 추가.
"""
import os, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import spearmanr

STS_DIR   = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
DEVICE    = torch.device("cuda")
SEED      = 42
NS        = [300, 500, 1000, 2000, 5000]
DIMS      = [20, 80, 320]

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── 데이터 로드 ──────────────────────────────────────────────
with open(f"{STS_DIR}/sts_test.json") as f:
    test_data = json.load(f)
test_pairs  = [(item['sentence1'], item['sentence2']) for item in test_data]
test_scores = [item['score'] for item in test_data]

with open(f"{CACHE_DIR}/qwen-4b_full_sents.json") as f:
    full_sents = json.load(f)
full_embs = np.load(f"{CACHE_DIR}/qwen-4b_full_embs.npy").astype(np.float32)
sent2idx = {s: i for i, s in enumerate(full_sents)}

eval_sents = list({s for s1, s2 in test_pairs for s in (s1, s2)})
X_eval = {s: full_embs[sent2idx[s]] for s in eval_sents}

# ── 평가 ─────────────────────────────────────────────────────
def spearman_score(emb_map):
    pred = [float(np.dot(emb_map.get(s1, np.zeros(1)), emb_map.get(s2, np.zeros(1))))
            for s1, s2 in test_pairs]
    return spearmanr(test_scores, pred).correlation * 100

def normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# ── ZCA + PCA (ZCA-only baseline) ───────────────────────────
def zca_only(X_fit, dim, shrink=0.08):
    Xt = torch.from_numpy(X_fit).to(DEVICE)
    mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(Xt.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    mu_np = mu.cpu().numpy(); S_np = S.cpu().numpy()

    # ZCA whitening + L2 norm
    Xw_fit = F.normalize(((Xt - mu) @ S), dim=1).cpu().numpy()
    mu_w = Xw_fit.mean(0)
    Xwc = Xw_fit - mu_w
    _, _, Vt = np.linalg.svd(Xwc, full_matrices=False)
    k = min(dim, Vt.shape[0])
    W_pca = Vt[:k].T  # (D, k)

    emb_map = {}
    for s in eval_sents:
        x = X_eval[s]
        xw = (x - mu_np) @ S_np; xw = xw / (np.linalg.norm(xw) + 1e-12)
        z = xw @ W_pca - mu_w @ W_pca
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── PCA-whitening ───────────────────────────────────────────
def pca_white(X_fit, dim, shrink=0.08):
    Xt = torch.from_numpy(X_fit).to(DEVICE).double()
    N, D = Xt.shape; mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(N-1, 1)
    tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE, dtype=Xt.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(dim, D)
    W = (evec[:, :k] @ torch.diag(1.0 / torch.sqrt(torch.clamp(ev[:k], min=1e-6))))
    W_np = W.cpu().numpy().astype(np.float32)
    mu_np = mu.cpu().numpy().astype(np.float32)
    emb_map = {}
    for s in eval_sents:
        x = X_eval[s]
        z = (x - mu_np) @ W_np
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── Soft-whitening (eigenvalue^0.5 scaling) ─────────────────
def soft_white(X_fit, dim, shrink=0.08):
    Xt = torch.from_numpy(X_fit).to(DEVICE).double()
    N, D = Xt.shape; mu = Xt.mean(0); Xc = Xt - mu
    Cov = (Xc.T @ Xc) / max(N-1, 1)
    tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE, dtype=Xt.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(dim, D)
    # soft = eigenvalue^(-1/4)  (PCA-white의 절반 강도)
    W = (evec[:, :k] @ torch.diag(1.0 / torch.pow(torch.clamp(ev[:k], min=1e-6), 0.25)))
    W_np = W.cpu().numpy().astype(np.float32)
    mu_np = mu.cpu().numpy().astype(np.float32)
    emb_map = {}
    for s in eval_sents:
        x = X_eval[s]
        z = (x - mu_np) @ W_np
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── Random projection ───────────────────────────────────────
def random_proj(X_fit, dim, seed=SEED):
    rng = np.random.RandomState(seed)
    D = X_fit.shape[1]
    R = rng.randn(D, dim).astype(np.float32) / np.sqrt(D)
    Q, _ = np.linalg.qr(R)
    emb_map = {}
    for s in eval_sents:
        z = X_eval[s] @ Q
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── LPP (graph laplacian projection) ────────────────────────
def lpp_proj(X_fit, dim, k=10):
    N, D = X_fit.shape
    Xt = torch.from_numpy(X_fit).to(DEVICE)
    Xn = F.normalize(Xt, dim=1)
    sim = (Xn @ Xn.T).cpu().numpy()
    np.fill_diagonal(sim, -np.inf)
    # k-NN graph
    knn = np.argsort(-sim, axis=1)[:, :k]
    W_adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in knn[i]:
            W_adj[i, j] = 1.0; W_adj[j, i] = 1.0
    D_diag = W_adj.sum(1)
    L = np.diag(D_diag) - W_adj
    # Generalized eigenvalue: X^T L X w = lam X^T D X w
    XtX = X_fit.T @ (D_diag[:, None] * X_fit)
    XtLX = X_fit.T @ L @ X_fit
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(XtLX, XtX)
        # smallest eigenvalues = best preserving locality
        W = eigvecs[:, :dim]
    except Exception as e:
        # fallback
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(XtX) @ XtLX)
        W = eigvecs[:, :dim]
    emb_map = {}
    for s in eval_sents:
        z = X_eval[s] @ W
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map

# ── 각 N별 실행 ──────────────────────────────────────────────
for N in NS:
    csv_path = f"/workspace/NCWP/sts_ablation_N{N}.csv"
    print(f"\n{'='*55}")
    print(f"N={N}  (csv: {csv_path})")
    print(f"{'='*55}")

    # 동일 seed로 동일한 fit sample
    rng = np.random.RandomState(SEED)
    fit_idx = rng.choice(len(full_sents), size=N, replace=False)
    X_fit = full_embs[fit_idx]

    new_rows = []
    BASELINES = [
        ("PCA-whitening", pca_white),
        ("Soft-White",    soft_white),
        ("Random",        random_proj),
        ("LPP",           lpp_proj),
        ("ZCA-only",      zca_only),
    ]
    for name, fn in BASELINES:
        row = {"variant": name}
        print(f"  [{name}]")
        for d in DIMS:
            print(f"    r={d}...", end=" ", flush=True)
            emb_map = fn(X_fit, d)
            sp = spearman_score(emb_map)
            row[f"r={d}"] = round(sp, 2)
            print(f"Spearman={sp:.2f}")
        new_rows.append(row)

    # 기존 CSV에 추가
    df = pd.read_csv(csv_path)
    df = df[~df['variant'].isin([r['variant'] for r in new_rows])]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"  → 업데이트: {csv_path}")

print("\n전체 완료!")
