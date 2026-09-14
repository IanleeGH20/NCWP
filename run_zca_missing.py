"""
run_zca_missing.py
ZCA-only (ZCA whitening + PCA 차원축소)를
SCIDOCS, NFCorpus, FiQA, MRPC, SICK 에 대해 계산.
이미 있는 임베딩 캐시를 사용하므로 모델 로딩 불필요.
결과를 각 데이터셋의 *_results_main.csv 에 ZCA-only 행으로 추가.
"""
import os, math, numpy as np, torch, torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT   = "/workspace/NCWP"
os.environ["HF_HOME"] = "/workspace/RAG/code/Make_embedding/nanoGPT"

# ── helpers ──────────────────────────────────────────────────
def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

@torch.no_grad()
def _zca_shrink(X_np, shrink=0.08, eps=1e-6):
    X = torch.from_numpy(X_np).to(DEVICE)
    mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1,1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs); ev = torch.clamp(ev, min=eps)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T
    return mu.cpu().numpy(), S.cpu().numpy()

@torch.no_grad()
def zca_pca(corpus, query, X_fit, dim):
    mu, S = _zca_shrink(X_fit)
    def _xform(X_np):
        Xw = F.normalize(torch.from_numpy(
            (X_np - mu) @ S).to(DEVICE), dim=1).cpu().numpy()
        # PCA on whitened space
        Xwc = Xw - Xw.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xwc, full_matrices=False)
        W_pca = Vt[:dim].T
        return _l2(Xwc @ W_pca)
    # fit PCA from X_fit whitened
    Xw_fit = F.normalize(torch.from_numpy(
        (X_fit - mu) @ S).to(DEVICE), dim=1).cpu().numpy()
    Xwc_fit = Xw_fit - Xw_fit.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xwc_fit, full_matrices=False)
    def _xform2(X_np):
        Xw = F.normalize(torch.from_numpy(
            (X_np - mu) @ S).to(DEVICE), dim=1).cpu().numpy()
        Xwc = Xw - Xw_fit.mean(0, keepdims=True)
        return _l2(Xwc @ Vt[:dim].T)
    return _xform2(corpus), _xform2(query)


def evaluate(c_embs, q_embs, corpus_ids, test_qids, qrels,
             k_values=(10,100), chunk=500):
    max_k = max(k_values)
    ndcg_at={k:[] for k in k_values}; recall_at={k:[] for k in k_values}; ap=[]
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi+chunk)
        sim = q_embs[qi:qe] @ c_embs.T
        if sim.shape[1] <= max_k: top_idx = np.argsort(-sim, axis=1)
        else:
            part = np.argpartition(-sim, max_k, axis=1)[:,:max_k]
            order = np.argsort(-sim[np.arange(len(part))[:,None], part], axis=1)
            top_idx = part[np.arange(len(part))[:,None], order]
        for ci, q in enumerate(range(qi, qe)):
            qid = test_qids[q]
            if qid not in qrels: continue
            rel = qrels[qid]; ranked = [corpus_ids[i] for i in top_idx[ci]]
            ideal = sorted(rel.values(), reverse=True)
            for k in k_values:
                top = ranked[:k]
                dcg  = sum((2**rel.get(d,0)-1)/math.log2(r+2) for r,d in enumerate(top))
                idcg = sum((2**rv-1)/math.log2(r+2) for r,rv in enumerate(ideal[:k]))
                ndcg_at[k].append(dcg/idcg if idcg>0 else 0.)
                rs = {d for d,s in rel.items() if s>0}
                recall_at[k].append(len(set(top)&rs)/len(rs) if rs else 0.)
            tr=sum(1 for s in rel.values() if s>0); nr=0; a=0.
            for r,d in enumerate(ranked,1):
                if rel.get(d,0)>0: nr+=1; a+=nr/r
            ap.append(a/tr if tr>0 else 0.)
    m={f"ndcg@{k}":np.mean(ndcg_at[k])*100 for k in k_values}
    m.update({f"recall@{k}":np.mean(recall_at[k])*100 for k in k_values})
    m["map"]=np.mean(ap)*100; return m


# ── Dataset configs ──────────────────────────────────────────
BEIR_DATASETS = {
    "scidocs":  ("mteb/scidocs",  "test",  None,  [24,48,96,192,384]),
    "nfcorpus": ("mteb/nfcorpus", "test",  None,  [24,48,96,192,384]),
    "fiqa":     ("mteb/fiqa",     "test",  None,  [24,48,96,192,384]),
}
STS_DATASETS = {
    "mrpc": None,  # loaded separately
    "sick": None,
}
MODELS = ["e5-base", "bge-base"]

def load_beir_qrels(mteb_name, test_split="test"):
    qr = load_dataset(mteb_name)
    qrels = defaultdict(dict)
    for r in qr[test_split]:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    c_ds = load_dataset(mteb_name, "corpus", split="corpus")
    corpus_ids = [r["_id"] for r in c_ds]
    q_ds = load_dataset(mteb_name, "queries", split="queries")
    qid2tx = {r["_id"]: r["text"] for r in q_ds}
    test_qids = [q for q in sorted(qrels) if q in qid2tx]
    return corpus_ids, test_qids, dict(qrels)

def load_mrpc_qrels():
    ds = load_dataset("glue","mrpc")
    sents = set()
    for split in ["train","validation","test"]:
        for r in ds[split]:
            sents.add(r["sentence1"]); sents.add(r["sentence2"])
    corpus_ids = sorted(sents)
    qrels = defaultdict(dict)
    for r in ds["test"]:
        if r["label"]==1:
            qrels[r["sentence1"]][r["sentence2"]] = 1
            qrels[r["sentence2"]][r["sentence1"]] = 1
    test_qids = [q for q in sorted(qrels) if q in set(corpus_ids)]
    return corpus_ids, test_qids, dict(qrels)

def load_sick_qrels():
    ds = load_dataset("sick", trust_remote_code=True)
    sents = set()
    for split in ["train","validation","test"]:
        for r in ds[split]:
            sents.add(r["sentence_A"]); sents.add(r["sentence_B"])
    corpus_ids = sorted(sents)
    qrels = defaultdict(dict)
    for r in ds["test"]:
        if r["relatedness_score"] >= 4.0:
            qrels[r["sentence_A"]][r["sentence_B"]] = int(round(r["relatedness_score"]))
            qrels[r["sentence_B"]][r["sentence_A"]] = int(round(r["relatedness_score"]))
    test_qids = [q for q in sorted(qrels) if q in set(corpus_ids)]
    return corpus_ids, test_qids, dict(qrels)


def update_csv(csv_path, new_rows):
    """결과 CSV에 ZCA-only 행 추가 (중복 방지)."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df.method != "ZCA-only"]   # 기존 ZCA-only 제거 후 덮어쓰기
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"  저장: {csv_path}  (+{len(new_rows)} rows)")


def run_zca_for_dataset(ds_name, corpus_ids, test_qids, qrels, target_dims, model):
    cache_dir = f"{ROOT}/{ds_name}_results/embedding_cache/{model}"
    corpus_np = np.load(f"{cache_dir}/corpus_base.npy").astype(np.float32)
    query_np  = np.load(f"{cache_dir}/test_queries_base.npy").astype(np.float32)
    fit_np    = np.load(f"{cache_dir}/fit_corpus_sample.npy").astype(np.float32) \
                if os.path.exists(f"{cache_dir}/fit_corpus_sample.npy") else \
                np.load(f"{cache_dir}/fit_corpus_sample_N20000.npy").astype(np.float32) \
                if os.path.exists(f"{cache_dir}/fit_corpus_sample_N20000.npy") else None
    if fit_np is None:
        # fallback: try any fit file
        import glob
        fits = glob.glob(f"{cache_dir}/fit_*.npy")
        if not fits: print(f"  ❌ fit 캐시 없음: {cache_dir}"); return []
        fit_np = np.load(fits[0]).astype(np.float32)

    rows = []
    for dim in target_dims:
        c, q = zca_pca(corpus_np, query_np, fit_np, dim)
        m = evaluate(c, q, corpus_ids, test_qids, qrels)
        rows.append({"dim":dim,"method":"ZCA-only",**m,"anisotropy":float("nan"),"self_sim":float("nan")})
        print(f"    dim={dim:>4}  NDCG@10={m['ndcg@10']:.2f}")
    return rows


# ── BEIR datasets ─────────────────────────────────────────────
for ds_name, (mteb_name, test_split, _, target_dims) in BEIR_DATASETS.items():
    print(f"\n{'='*55}")
    print(f"  {ds_name.upper()}")
    print(f"{'='*55}")
    corpus_ids, test_qids, qrels = load_beir_qrels(mteb_name, test_split)
    print(f"  corpus={len(corpus_ids):,}  queries={len(test_qids):,}")
    for model in MODELS:
        cache = f"{ROOT}/{ds_name}_results/embedding_cache/{model}"
        if not os.path.exists(cache):
            print(f"  ❌ {model}: 캐시 없음"); continue
        print(f"\n  [{model}]")
        rows = run_zca_for_dataset(ds_name, corpus_ids, test_qids, qrels,
                                   target_dims, model)
        if rows:
            csv_path = f"{ROOT}/{ds_name}_results/fit_corpus_sample/{model}_results_main.csv"
            update_csv(csv_path, rows)

# ── STS datasets ──────────────────────────────────────────────
STS_LOADERS = {"mrpc": load_mrpc_qrels, "sick": load_sick_qrels}
for ds_name, loader in STS_LOADERS.items():
    print(f"\n{'='*55}")
    print(f"  {ds_name.upper()}")
    print(f"{'='*55}")
    corpus_ids, test_qids, qrels = loader()
    print(f"  corpus={len(corpus_ids):,}  queries={len(test_qids):,}")
    for model in MODELS:
        cache = f"{ROOT}/{ds_name}_results/embedding_cache/{model}"
        if not os.path.exists(cache):
            print(f"  ❌ {model}: 캐시 없음"); continue
        print(f"\n  [{model}]")
        fit_file = f"{cache}/fit_corpus_sample.npy"
        if not os.path.exists(fit_file):
            print(f"  ❌ fit 캐시 없음"); continue
        rows = run_zca_for_dataset(ds_name, corpus_ids, test_qids, qrels,
                                   [24,48,96,192,384], model)
        if rows:
            csv_path = f"{ROOT}/{ds_name}_results/fit_corpus_sample/{model}_results_main.csv"
            update_csv(csv_path, rows)

print("\n=== ZCA-only 계산 완료 ===")
