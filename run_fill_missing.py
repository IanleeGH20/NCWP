"""
run_fill_missing.py — 부족한 ZCA-only 값 채우기

1. Quora E5-base ZCA-only: ablation_ext → main CSV 병합
2. Quora BGE-base ZCA-only: 새로 계산
3. LLM (Llama-8B, Qwen-4B, Qwen-8B) ZCA-only: FiQA + NFCorpus
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
def zca_pca(corpus_np, query_np, fit_np, dim):
    """ZCA shrink whitening + PCA top-k → L2 norm"""
    X = torch.from_numpy(fit_np).to(DEVICE)
    mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc)/max(X.shape[0]-1,1)
    D = Cov.shape[0]; tr = torch.trace(Cov); shrink=0.08; eps=1e-6
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D,device=DEVICE)
    ev, evec = torch.linalg.eigh(Cs); ev = torch.clamp(ev,min=eps)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T
    mu_np = mu.cpu().numpy(); S_np = S.cpu().numpy()

    # PCA on ZCA-whitened fit
    Xw_fit = F.normalize(torch.from_numpy((fit_np - mu_np) @ S_np).to(DEVICE), dim=1)
    Xwc = (Xw_fit - Xw_fit.mean(0, keepdim=True)).cpu().numpy()
    _, _, Vt = np.linalg.svd(Xwc, full_matrices=False)
    W_pca = Vt[:dim].T  # (D, dim)
    mu_w  = Xw_fit.mean(0).cpu().numpy()

    def _transform(Xnp):
        Xw = F.normalize(torch.from_numpy((Xnp - mu_np) @ S_np).to(DEVICE), dim=1).cpu().numpy()
        return _l2((Xw - mu_w) @ W_pca)

    return _transform(corpus_np), _transform(query_np)


def evaluate(c, q, corpus_ids, test_qids, qrels, chunk=500, k_values=(10,100)):
    max_k = max(k_values)
    ndcg={k:[] for k in k_values}; recall={k:[] for k in k_values}; ap=[]
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi+chunk)
        sim = q[qi:qe] @ c.T
        if sim.shape[1] <= max_k: top = np.argsort(-sim,axis=1)
        else:
            part  = np.argpartition(-sim, max_k, axis=1)[:,:max_k]
            order = np.argsort(-sim[np.arange(len(part))[:,None],part],axis=1)
            top   = part[np.arange(len(part))[:,None],order]
        for ci,q2 in enumerate(range(qi,qe)):
            qid = test_qids[q2]
            if qid not in qrels: continue
            rel = qrels[qid]; ranked=[corpus_ids[i] for i in top[ci]]
            ideal=sorted(rel.values(),reverse=True)
            for k in k_values:
                t=ranked[:k]
                dcg =sum((2**rel.get(d,0)-1)/math.log2(r+2) for r,d in enumerate(t))
                idcg=sum((2**rv-1)/math.log2(r+2) for r,rv in enumerate(ideal[:k]))
                ndcg[k].append(dcg/idcg if idcg>0 else 0.)
                rs={d for d,s in rel.items() if s>0}
                recall[k].append(len(set(t)&rs)/len(rs) if rs else 0.)
            tr=sum(1 for s in rel.values() if s>0); nr=0; a=0.
            for r,d in enumerate(ranked,1):
                if rel.get(d,0)>0: nr+=1; a+=nr/r
            ap.append(a/tr if tr>0 else 0.)
    m={f"ndcg@{k}":np.mean(ndcg[k])*100 for k in k_values}
    m.update({f"recall@{k}":np.mean(recall[k])*100 for k in k_values})
    m["map"]=np.mean(ap)*100; return m


def upsert_rows(csv_path, new_rows):
    """ZCA-only 행을 CSV에 추가(기존 ZCA-only 행 덮어쓰기)"""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df.method != "ZCA-only"]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"  → 저장: {csv_path} (+{len(new_rows)} rows)")


# ═══════════════════════════════════════════════════════════════
# Step 1: Quora E5-base ZCA-only — ablation_ext → main CSV 병합
# ═══════════════════════════════════════════════════════════════
print("="*60)
print("Step 1: Quora E5-base ZCA-only 병합")
print("="*60)
ext = pd.read_csv(f"{ROOT}/quora_results/ablation/ablation_ext_results.csv")
zca_e5 = ext[(ext.model=="e5-base")&(ext.variant=="ZCA-only")].copy()
zca_e5["method"] = "ZCA-only"
main_cols_path = f"{ROOT}/quora_results/fit_corpus_sample/e5-base_results_main.csv"
existing = pd.read_csv(main_cols_path)
# 공통 컬럼만 유지
keep_cols = [c for c in existing.columns if c in zca_e5.columns]
zca_rows = zca_e5[keep_cols].to_dict("records")
upsert_rows(main_cols_path, zca_rows)
print(f"  병합된 ZCA-only dims: {sorted(zca_e5.dim.tolist())}")


# ═══════════════════════════════════════════════════════════════
# Step 2: Quora BGE-base ZCA-only
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("Step 2: Quora BGE-base ZCA-only")
print("="*60)
print("Loading Quora data...")
qr = load_dataset("mteb/quora")
qrels_q = defaultdict(dict)
for r in qr["test"]:
    qrels_q[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
c_ds = load_dataset("mteb/quora","corpus",split="corpus")
corpus_ids_q = [r["_id"] for r in c_ds]
q_ds = load_dataset("mteb/quora","queries",split="queries")
qid2tx = {r["_id"]:r["text"] for r in q_ds}
test_qids_q = [q for q in sorted(qrels_q) if q in qid2tx]
print(f"  corpus={len(corpus_ids_q):,}  queries={len(test_qids_q):,}")

cache_bge = f"{ROOT}/quora_results/embedding_cache/bge-base"
corpus_bge = np.load(f"{cache_bge}/corpus_base.npy").astype(np.float32)
query_bge  = np.load(f"{cache_bge}/test_queries_base.npy").astype(np.float32)
fit_bge    = np.load(f"{cache_bge}/fit_corpus_sample_N20000.npy").astype(np.float32)

TARGET_DIMS = [24, 48, 96, 192, 384]
rows_bge = []
for dim in TARGET_DIMS:
    print(f"  dim={dim:>4} ... ", end="", flush=True)
    c, q = zca_pca(corpus_bge, query_bge, fit_bge, dim)
    m = evaluate(c, q, corpus_ids_q, test_qids_q, dict(qrels_q), chunk=500)
    rows_bge.append({"dim":dim,"method":"ZCA-only",**m})
    print(f"NDCG@10={m['ndcg@10']:.2f}")

upsert_rows(f"{ROOT}/quora_results/fit_corpus_sample/bge-base_results_main.csv", rows_bge)


# ═══════════════════════════════════════════════════════════════
# Step 3: LLM ZCA-only — FiQA + NFCorpus
# ═══════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("Step 3: LLM ZCA-only (FiQA + NFCorpus)")
print("="*60)

def load_beir_qrels(mteb_name, test_split="test"):
    qr = load_dataset(mteb_name)
    qrels = defaultdict(dict)
    for r in qr[test_split]:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    c_ds = load_dataset(mteb_name,"corpus",split="corpus")
    corpus_ids = [r["_id"] for r in c_ds]
    q_ds = load_dataset(mteb_name,"queries",split="queries")
    qid2tx = {r["_id"]:r["text"] for r in q_ds}
    test_qids = [q for q in sorted(qrels) if q in qid2tx]
    return corpus_ids, test_qids, dict(qrels)

LLM_DIMS = {
    "llama-8b": [64, 128, 256, 512, 1024, 2048],
    "qwen-4b":  [40, 80, 160, 320, 640, 1280],
    "qwen-8b":  [56, 112, 224, 448, 896, 1792],
}
FIT_NAMES = {
    "corpus_sample": "fit_corpus_sample_N10000.npy",
    "dev_queries":   "fit_dev_queries.npy",
}

BEIR_DS = {
    "fiqa":     ("mteb/fiqa",     "test"),
    "nfcorpus": ("mteb/nfcorpus", "test"),
}

for ds_name, (mteb_name, test_split) in BEIR_DS.items():
    print(f"\n  [{ds_name.upper()}]")
    corpus_ids, test_qids, qrels = load_beir_qrels(mteb_name, test_split)
    print(f"  corpus={len(corpus_ids):,}  queries={len(test_qids):,}")

    for model in ["llama-8b","qwen-4b","qwen-8b"]:
        dims = LLM_DIMS[model]
        cache = f"{ROOT}/{ds_name}_results/embedding_cache/{model}"
        if not os.path.exists(cache):
            print(f"    ❌ {model}: 캐시 없음"); continue

        corpus_np = np.load(f"{cache}/corpus_base.npy").astype(np.float32)
        query_np  = np.load(f"{cache}/test_queries_base.npy").astype(np.float32)

        for fm, fit_fname in FIT_NAMES.items():
            fit_path = os.path.join(cache, fit_fname)
            if not os.path.exists(fit_path):
                # fallback
                alts = [f for f in os.listdir(cache) if f.startswith("fit_") and f.endswith(".npy")]
                if alts: fit_path = os.path.join(cache, alts[0])
                else: print(f"    ❌ {model}/{fm}: fit 캐시 없음"); continue
            fit_np = np.load(fit_path).astype(np.float32)

            csv_path = f"{ROOT}/{ds_name}_results/fit_{fm}/{model}_results_main.csv"
            if not os.path.exists(csv_path):
                print(f"    ❌ {model}/{fm}: CSV 없음"); continue

            print(f"    {model} / {fm}:")
            rows = []
            for dim in dims:
                print(f"      dim={dim:>5} ... ", end="", flush=True)
                c, q = zca_pca(corpus_np, query_np, fit_np, dim)
                eval_chunk = 200 if ds_name == "nfcorpus" else 500
                m = evaluate(c, q, corpus_ids, test_qids, qrels, chunk=eval_chunk)
                rows.append({"dim":dim,"method":"ZCA-only",**m})
                print(f"NDCG@10={m['ndcg@10']:.2f}")
            upsert_rows(csv_path, rows)

print("\n\n=== 모든 ZCA-only 채우기 완료 ===")
