"""
add_zca_only.py — ZCA-only 결과를 두 CSV에 추가
1. qwen-4b / Quora 전체 / query_sample N=300
2. E5-base  / Quora v4  / corpus_sample N=20000
"""
import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os
from collections import defaultdict
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 공통 유틸 ─────────────────────────────────────────────────
def compute_zca_proj(corpus_np, query_np, fit_np, dim, shrink=0.08, chunk=2000):
    """ZCA whitening + PCA top-dim → normalized GPU half tensors"""
    X = torch.from_numpy(fit_np.astype(np.float32)).to(DEVICE)
    mu = X.mean(0)
    Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.clamp(ev, min=1e-6)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T       # (D, D) whitening

    # fit → ZCA → PCA
    Xw_fit = F.normalize((X - mu) @ S, dim=1)               # (N, D)
    mu_w   = Xw_fit.mean(0)                                  # (D,)
    Xwc    = (Xw_fit - mu_w).cpu().numpy()
    _, _, Vt = np.linalg.svd(Xwc, full_matrices=False)
    W_pca  = torch.from_numpy(Vt[:dim].T.astype(np.float32)).to(DEVICE)  # (D, dim)
    mu_w_proj = (mu_w @ W_pca)                               # (dim,)

    def _transform(Xnp):
        parts = []
        for i in range(0, len(Xnp), chunk):
            xb = torch.from_numpy(Xnp[i:i+chunk].astype(np.float32)).to(DEVICE)
            xw = F.normalize((xb - mu) @ S, dim=1)          # ZCA whitening + norm
            xp = F.normalize(xw @ W_pca - mu_w_proj, dim=1) # PCA centering + norm
            parts.append(xp.half())
        return torch.cat(parts, dim=0)                       # (N, dim) GPU half

    return _transform(corpus_np), _transform(query_np)


def evaluate_gpu(c_t, q_t, corpus_ids, test_qids, qrels, chunk=500):
    """GPU topk 기반 nDCG@10, Recall@100, MAP 계산"""
    ndcg10 = []; recall100 = []; ap_list = []
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi+chunk)
        with torch.no_grad():
            sim = (q_t[qi:qe] @ c_t.T).float()              # (B, C)
            _, top_idx = torch.topk(sim, k=min(100, sim.shape[1]), dim=1)
            top100 = top_idx.cpu().numpy()
        for ci, q2 in enumerate(range(qi, qe)):
            qid = test_qids[q2]
            if qid not in qrels: continue
            rel = qrels[qid]
            ranked = [corpus_ids[i] for i in top100[ci]]
            # nDCG@10
            h10 = [1.0 if d in rel else 0.0 for d in ranked[:10]]
            dcg  = sum(h/np.log2(r+2) for r,h in enumerate(h10))
            n    = min(len(rel), 10)
            idcg = sum(1.0/np.log2(r+2) for r in range(n))
            ndcg10.append(dcg/idcg if idcg > 0 else 0.0)
            # Recall@100
            recall100.append(sum(1.0 if d in rel else 0.0 for d in ranked)/len(rel))
            # MAP@100
            nr=0; ps=0.0
            for r,d in enumerate(ranked): nr += d in rel; ps += nr/(r+1) if d in rel else 0
            ap_list.append(ps/len(rel))
    return {'ndcg@10':    np.mean(ndcg10)*100,
            'recall@100': np.mean(recall100)*100,
            'map':        np.mean(ap_list)*100}


def upsert(csv_path, new_rows):
    df = pd.read_csv(csv_path)
    df = df[df['method'] != 'ZCA-only']
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"  → 저장: {csv_path}")


# ── Quora 공통 ID/qrels 로드 ─────────────────────────────────
print("Quora ID/qrels 로드 중...")
ds_corp  = load_dataset("mteb/quora", "corpus",  split="corpus")
ds_qs    = load_dataset("mteb/quora", "queries", split="queries")
ds_qrel  = load_dataset("mteb/quora", split="test")

corpus_ids = [str(r['_id']) for r in ds_corp]         # 522,931개
qid2tx     = {str(r['_id']): r['text'] for r in ds_qs}

qrels = defaultdict(set)
for r in ds_qrel:
    qrels[str(r['query-id'])].add(str(r['corpus-id']))

test_qids = sorted([q for q in qrels if q in qid2tx]) # 10,000개 (정렬 일치)
print(f"  corpus={len(corpus_ids)}, test_qids={len(test_qids)}")


# ═══════════════════════════════════════════════════════════════
# 1. qwen-4b / Quora 전체 / query_sample N=300
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("1. qwen-4b / Quora / query_sample N=300")
print("="*55)

CACHE_Q4 = '/workspace/NCWP/quora_results/embedding_cache/qwen-4b'
CSV_Q4   = '/workspace/NCWP/quora_results/fit_query_sample/qwen-4b_N300_results_main.csv'

corpus_q4 = np.load(f'{CACHE_Q4}/corpus_base.npy').astype(np.float32)
query_q4  = np.load(f'{CACHE_Q4}/test_queries_base.npy').astype(np.float32)
fit_q4    = np.load(f'{CACHE_Q4}/fit_query_sample_N300.npy').astype(np.float32)
print(f"  corpus={corpus_q4.shape}, query={query_q4.shape}, fit={fit_q4.shape}")

DIMS_Q4 = [40, 80, 160, 320, 640, 1280]
rows_q4 = []
for dim in DIMS_Q4:
    print(f"  ZCA-only dim={dim}...", end=' ', flush=True)
    c_t, q_t = compute_zca_proj(corpus_q4, query_q4, fit_q4, dim)
    r = evaluate_gpu(c_t, q_t, corpus_ids, test_qids, qrels)
    rows_q4.append({'method':'ZCA-only','dim':dim, **r})
    print(f"nDCG@10={r['ndcg@10']:.2f}")

upsert(CSV_Q4, rows_q4)


# ═══════════════════════════════════════════════════════════════
# 2. E5-base / Quora v4 / corpus_sample N=20000
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("2. E5-base / Quora v4 / corpus_sample N=20000")
print("="*55)

CACHE_E5 = '/workspace/NCWP/quora_results/embedding_cache/e5-base'
CSV_E5   = '/workspace/NCWP/quora_results/fit_corpus_sample/e5-base_results_main.csv'

corpus_e5 = np.load(f'{CACHE_E5}/corpus_base.npy').astype(np.float32)
query_e5  = np.load(f'{CACHE_E5}/test_queries_base.npy').astype(np.float32)
fit_e5    = np.load(f'{CACHE_E5}/fit_corpus_sample_N20000.npy').astype(np.float32)
print(f"  corpus={corpus_e5.shape}, query={query_e5.shape}, fit={fit_e5.shape}")

DIMS_E5 = [24, 48, 96, 192, 384]
rows_e5 = []
for dim in DIMS_E5:
    print(f"  ZCA-only dim={dim}...", end=' ', flush=True)
    c_t, q_t = compute_zca_proj(corpus_e5, query_e5, fit_e5, dim)
    r = evaluate_gpu(c_t, q_t, corpus_ids, test_qids, qrels)
    rows_e5.append({'method':'ZCA-only','dim':dim, **r})
    print(f"nDCG@10={r['ndcg@10']:.2f}")

upsert(CSV_E5, rows_e5)


# ── 최종 출력 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 결과")
print("="*60)

print(f"\n[qwen-4b / Quora / N=300]  Base(2560)=43.70")
df = pd.read_csv(CSV_Q4)
for m in ['PCA-White','Soft-White','Random','LPP','ZCA-only','NCWP']:
    sub = df[df['method']==m].set_index('dim')['ndcg@10']
    row = f"  {m:<14}"
    for d in DIMS_Q4:
        row += f" {sub.get(d, float('nan')):>6.2f}"
    print(row)

print(f"\n[E5-base / Quora v4]  Base(768)=84.20")
df2 = pd.read_csv(CSV_E5)
for m in ['PCA-White','ZCA-only','NCWP']:
    sub = df2[df2['method']==m].set_index('dim')['ndcg@10']
    row = f"  {m:<14}"
    for d in DIMS_E5:
        row += f" {sub.get(d, float('nan')):>6.2f}"
    print(row)
