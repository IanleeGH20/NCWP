"""
ZCA-only 미완료 N (300, 3000, 5000) 계산 후 CSV에 추가
"""
import numpy as np, torch, torch.nn.functional as F, pandas as pd, os
from datasets import load_dataset
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE  = '/workspace/NCWP/quora_results'
CACHE = f'{BASE}/embedding_cache/qwen-4b'

def zca_pca(X_fit, dim, shrink=0.08, chunk=2000):
    X = torch.from_numpy(X_fit.astype(np.float32)).to(DEVICE)
    mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.clamp(ev, min=1e-6)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T
    Xw = F.normalize((X - mu) @ S, dim=1)
    mu_w = Xw.mean(0)
    Xwc  = (Xw - mu_w).cpu().numpy()
    _, _, Vt = np.linalg.svd(Xwc, full_matrices=False)
    k = min(dim, Vt.shape[0])
    W_pca = torch.from_numpy(Vt[:k].T.astype(np.float32)).to(DEVICE)
    mu_w_proj = mu_w @ W_pca
    def _t(Xnp):
        parts = []
        for i in range(0, len(Xnp), chunk):
            xb = torch.from_numpy(Xnp[i:i+chunk].astype(np.float32)).to(DEVICE)
            xw = F.normalize((xb - mu) @ S, dim=1)
            xp = F.normalize(xw @ W_pca - mu_w_proj, dim=1)
            parts.append(xp.half())
        return torch.cat(parts, dim=0)
    return _t

def evaluate(c_t, q_t, corpus_ids, test_qids, qrels, chunk=500):
    ndcg10=[]; recall100=[]; ap_list=[]
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi+chunk)
        with torch.no_grad():
            sim = (q_t[qi:qe] @ c_t.T).float()
            _, top_idx = torch.topk(sim, k=min(100, sim.shape[1]), dim=1)
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
    return {'ndcg@10': np.mean(ndcg10)*100,
            'recall@100': np.mean(recall100)*100,
            'map': np.mean(ap_list)*100}

# qrels 로드
print("qrels 로드 중...")
ds_corp = load_dataset("mteb/quora", "corpus", split="corpus")
ds_qs   = load_dataset("mteb/quora", "queries", split="queries")
ds_qrel = load_dataset("mteb/quora", split="test")
corpus_ids = [str(r['_id']) for r in ds_corp]
qid2tx     = {str(r['_id']): r['text'] for r in ds_qs}
qrels = defaultdict(set)
for r in ds_qrel:
    qrels[str(r['query-id'])].add(str(r['corpus-id']))
test_qids = sorted([q for q in qrels if q in qid2tx])
print(f"  corpus={len(corpus_ids)}, test_qids={len(test_qids)}")

corpus_np = np.load(f'{CACHE}/corpus_base.npy').astype(np.float32)
query_np  = np.load(f'{CACHE}/test_queries_base.npy').astype(np.float32)

DIMS = [1280, 640, 320, 160, 80, 40, 20, 10, 5]

TARGETS = {
    300:  f'{CACHE}/fit_query_sample_N300.npy',
    3000: f'{CACHE}/fit_query_sample_N3000.npy',
    5000: f'{CACHE}/fit_query_sample_N5000.npy',
}

for N, fit_path in TARGETS.items():
    csv_path = f'{BASE}/fit_query_sample/qwen-4b_N{N}_results_main.csv'
    if not os.path.exists(fit_path):
        # N=5000 캐시 없을 수 있음
        print(f"N={N}: fit 캐시 없음, 스킵")
        continue
    fit_np = np.load(fit_path).astype(np.float32)
    print(f"\nN={N} (fit={fit_np.shape})")
    rows = []
    for dim in DIMS:
        print(f"  dim={dim}...", end=' ', flush=True)
        transform = zca_pca(fit_np, dim)
        c_t = transform(corpus_np)
        q_t = transform(query_np)
        r = evaluate(c_t, q_t, corpus_ids, test_qids, qrels)
        rows.append({'method': 'ZCA-only', 'dim': dim, **r})
        print(f"nDCG@10={r['ndcg@10']:.2f}")
    df = pd.read_csv(csv_path)
    df = df[df['method'] != 'ZCA-only']
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path} 저장")

print("\n=== 완료 ===")
