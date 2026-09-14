"""
run_beir_small.py  ── FiQA-2018 / SCIDOCS 경량 실험

리버탈용 추가 BEIR 벤치마크 실험 (속도 최우선)

설계:
  - 차원: [32, 64, 128] (3개)
  - 방법: Base / PCA-White / Soft-White / NCWP
  - Fit: corpus 랜덤 샘플 (FiQA=1000, SCIDOCS=500)
  - 평가: 전체 corpus (FiQA≈57K, SCIDOCS≈25K)
  - 캐시: embedding_cache 재사용

실행:
  python run_beir_small.py --dataset fiqa   --model qwen-4b
  python run_beir_small.py --dataset fiqa   --model llama-8b
  python run_beir_small.py --dataset scidocs --model qwen-4b
  python run_beir_small.py --dataset scidocs --model llama-8b
"""

import os, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from collections import defaultdict

# ── NVRTC fix ──────────────────────────────────────────────
for _p in ["/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
           "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"]:
    if os.path.exists(_p):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _p not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_p}:{_ld}"
        break

# ── Config ─────────────────────────────────────────────────
HF_CACHE   = "/workspace/RAG/code/Make_embedding/nanoGPT"
OUT_ROOT   = "/workspace/NCWP"
os.environ["HF_HOME"] = HF_CACHE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

TARGET_DIMS = [32, 64, 128]

DATASET_CONFIGS = {
    "fiqa": {
        "mteb_name": "mteb/fiqa",
        "fit_n":     1000,
        "eval_chunk": 200,
        "label": "FiQA-2018",
    },
    "scidocs": {
        "mteb_name": "mteb/scidocs",
        "fit_n":     500,
        "eval_chunk": 200,
        "label": "SCIDOCS",
    },
    # ── CQADupStack (Quora와 동일 구조: 질문 중복 탐지) ──
    "cqa-english": {
        "mteb_name": "mteb/cqadupstack-english",
        "fit_n":     1000,
        "eval_chunk": 200,
        "label": "CQADupStack-English",
    },
    "cqa-gaming": {
        "mteb_name": "mteb/cqadupstack-gaming",
        "fit_n":     1000,
        "eval_chunk": 200,
        "label": "CQADupStack-Gaming",
    },
    "cqa-physics": {
        "mteb_name": "mteb/cqadupstack-physics",
        "fit_n":     1000,
        "eval_chunk": 200,
        "label": "CQADupStack-Physics",
    },
}

MODEL_MAP = {
    "qwen-4b":  "Qwen/Qwen1.5-4B",
    "qwen-8b":  "Qwen/Qwen2-7B",
    "llama-8b": "meta-llama/Meta-Llama-3.1-8B",
}


# ── Reproducibility ───────────────────────────────────────
def _set_seed(s=SEED):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

_set_seed()


# ── Data Loading ──────────────────────────────────────────
def load_beir_data(dataset_name):
    """MTEB 형식 데이터 로드 (mteb/fiqa, mteb/scidocs)."""
    ds_cfg   = DATASET_CONFIGS[dataset_name]
    mteb_name = ds_cfg["mteb_name"]
    print(f"Loading {ds_cfg['label']} from {mteb_name} ...")

    # corpus
    corp_ds = load_dataset(mteb_name, "corpus", split="corpus")
    corpus_ids   = [r["_id"] for r in corp_ds]
    corpus_texts = [(r.get("title","") + " " + r.get("text","")).strip()
                    for r in corp_ds]

    # queries
    q_ds = load_dataset(mteb_name, "queries", split="queries")
    qid2text = {r["_id"]: r["text"] for r in q_ds}

    # qrels (test split)
    qrel_ds = load_dataset(mteb_name, split="test")
    qrels = defaultdict(dict)
    for r in qrel_ds:
        qid = str(r["query-id"]); cid = str(r["corpus-id"])
        qrels[qid][cid] = int(r["score"])
    qrels = dict(qrels)

    # test 쿼리 = qrels에 있는 것
    test_qids   = sorted([q for q in qrels if q in qid2text])
    test_qtexts = [qid2text[q] for q in test_qids]

    print(f"  corpus: {len(corpus_ids):,}  test_queries: {len(test_qids):,}")
    return corpus_ids, corpus_texts, test_qids, test_qtexts, qrels


# ── LLM Embedder ──────────────────────────────────────────
class HFLLMPoolEmbedder:
    def __init__(self, model_name):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        n = torch.cuda.device_count(); print(f"[Info] {n} GPUs.")
        kw = {"trust_remote_code": True,
              "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported()
                             else torch.float16,
              "attn_implementation": "eager",
              "device_map": "auto"}   # DataParallel 대신 device_map=auto (NCCL 오류 방지)
        self.model = AutoModel.from_pretrained(model_name, **kw)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences, batch_size=256):
        dev = getattr(self.model, "device", DEVICE)
        embs = []
        for i in range(0, len(sentences), batch_size):
            enc = self.tok(sentences[i:i+batch_size], padding=True,
                           truncation=True, max_length=128,
                           return_tensors="pt").to(dev)
            h   = self.model(**enc).last_hidden_state
            mask= enc["attention_mask"].unsqueeze(-1).float()
            x   = (h*mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embs.append(F.normalize(x,dim=1).cpu().float().numpy())
        return np.concatenate(embs) if embs else np.array([])


# ── Projection helpers ────────────────────────────────────
def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _zca_shrink(X, shrink=0.08, eps=1e-6):
    orig=X.dtype; X=X.double()
    mu=X.mean(0,keepdim=True); Xc=X-mu
    Cov=(Xc.T@Xc)/max(X.shape[0]-1,1)
    D_=Cov.shape[0]; tr=torch.trace(Cov)
    Cs=(1-shrink)*Cov+shrink*(tr/D_)*torch.eye(D_,device=X.device,dtype=X.dtype)
    ev,evec=torch.linalg.eigh(Cs)
    S=evec@torch.diag(1./torch.sqrt(torch.clamp(ev,min=eps)))@evec.T
    return mu.squeeze(0).to(orig), S.to(orig)

def pca_white(X_np, dim, shrink=0.08, eps=1e-6):
    X=torch.from_numpy(X_np).to(DEVICE).double()
    N,D=X.shape; mu=X.mean(0,keepdim=True); Xc=X-mu
    Cov=(Xc.T@Xc)/max(N-1,1); tr=torch.trace(Cov)
    Cs=(1-shrink)*Cov+shrink*(tr/D)*torch.eye(D,device=X.device,dtype=X.dtype)
    ev,evec=torch.linalg.eigh(Cs)
    ev=torch.flip(ev,[0]); evec=torch.flip(evec,[1])
    k=min(dim,D)
    W=evec[:,:k]@torch.diag(1./torch.sqrt(torch.clamp(ev[:k],min=eps)))
    return mu.squeeze(0).cpu().float().numpy(), W.cpu().float().numpy()

def soft_white(X_np, dim, shrink=0.08, eps=1e-6):
    X=torch.from_numpy(X_np).to(DEVICE).double()
    N,D=X.shape; mu=X.mean(0,keepdim=True); Xc=X-mu
    Cov=(Xc.T@Xc)/max(N-1,1)
    ev,evec=torch.linalg.eigh(Cov)
    ev=torch.flip(ev,[0]); evec=torch.flip(evec,[1])
    k=min(dim,D)
    W=evec[:,:k]@torch.diag(1./(torch.clamp(ev[:k],min=eps)**0.25))
    return mu.squeeze(0).cpu().float().numpy(), W.cpu().float().numpy()

def proj_linear(X, mu, W):
    return _l2((X - mu) @ W)

@torch.no_grad()
def _topk_knn(X, k, chunk=2048):
    N=X.shape[0]; idx=torch.empty((N,k),dtype=torch.long,device="cpu")
    for s in range(0,N,chunk):
        e=min(N,s+chunk); sim=X[s:e]@X.T
        sim[:,torch.arange(s,e,device=X.device)]=-1e9
        idx[s:e]=torch.topk(sim,k=k,dim=1).indices.cpu()
    return idx

def _orth_penalty(W):
    WT_W=W.T@W; return ((WT_W-torch.eye(WT_W.shape[0],device=W.device))**2).mean()

def _cov_penalty(Z):
    B=Z.shape[0]
    if B<=1: return torch.tensor(0.,device=Z.device)
    Zc=Z-Z.mean(0,keepdim=True); Cov=(Zc.T@Zc)/(B-1)
    return ((Cov-torch.eye(Cov.shape[0],device=Z.device))**2).mean()

def train_ncwp(X_np, dim, k=10, temp=0.12, lc=1.0, lo=1.0, lk=1.0,
               epochs=20, lr=8e-3, shrink=0.08):
    X=torch.from_numpy(X_np).to(DEVICE); N,D=X.shape
    mu,S=_zca_shrink(X,shrink)
    Xw=F.normalize((X-mu)@S,dim=1)
    k_=max(1,min(k,N-1)); knn=_topk_knn(Xw,k_)
    r=min(dim,D)
    W=torch.nn.Parameter(torch.randn(D,r,device=DEVICE)/math.sqrt(D))
    opt=torch.optim.AdamW([W],lr=lr); B=64
    for _ in range(epochs):
        idx=torch.randperm(N)[:B]
        pos=knn[idx.cpu(),torch.randint(0,k_,(B,))].to(DEVICE)
        Z=F.normalize(torch.cat([Xw[idx.to(DEVICE)],Xw[pos]],0)@W,dim=1)
        sim=(Z@Z.T)/temp; sim.fill_diagonal_(-1e9)
        loss=(lk*(-(sim[:B].gather(1,torch.arange(B,2*B,device=DEVICE).view(-1,1)).squeeze(1)
                    -torch.logsumexp(sim[:B],1)).mean())
              +lc*_cov_penalty(Z)+lo*_orth_penalty(W))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        SW=S@W; Y=(X-mu)@SW
    return SW.cpu().numpy(), mu.cpu().numpy(), Y.mean(0).cpu().numpy(), \
           torch.clamp(Y.std(0),min=1e-6).cpu().numpy()

def proj_ncwp(X, W, mu_in, mu_out, std_out):
    return _l2(((X-mu_in)@W - mu_out)/std_out)


# ── Evaluation ────────────────────────────────────────────
def evaluate(corpus_embs, corpus_ids, query_embs, query_ids, qrels,
             k_values=(10,100), chunk=200):
    max_k=max(k_values)
    ndcg_at={k:[] for k in k_values}
    recall_at={k:[] for k in k_values}
    ap_list=[]
    for qi in range(0,len(query_ids),chunk):
        qe=min(len(query_ids),qi+chunk)
        sim=query_embs[qi:qe]@corpus_embs.T
        if sim.shape[1]<=max_k:
            top_idx=np.argsort(-sim,axis=1)
        else:
            part=np.argpartition(-sim,max_k,axis=1)[:,:max_k]
            order=np.argsort(-sim[np.arange(len(part))[:,None],part],axis=1)
            top_idx=part[np.arange(len(part))[:,None],order]
        for ci,q in enumerate(range(qi,qe)):
            qid=query_ids[q]
            if qid not in qrels: continue
            rel=qrels[qid]; ranked=[corpus_ids[i] for i in top_idx[ci]]
            ideal=sorted(rel.values(),reverse=True)
            for k in k_values:
                top=ranked[:k]
                dcg=sum((2**rel.get(d,0)-1)/math.log2(r+2) for r,d in enumerate(top))
                idcg=sum((2**rv-1)/math.log2(r+2) for r,rv in enumerate(ideal[:k]))
                ndcg_at[k].append(dcg/idcg if idcg>0 else 0.)
                relset={d for d,s in rel.items() if s>0}
                recall_at[k].append(len(set(top)&relset)/len(relset) if relset else 0.)
            total_rel=sum(1 for s in rel.values() if s>0)
            nr=0; ap=0.
            for r,d in enumerate(ranked,1):
                if rel.get(d,0)>0: nr+=1; ap+=nr/r
            ap_list.append(ap/total_rel if total_rel>0 else 0.)
    m={f"ndcg@{k}":np.mean(ndcg_at[k])*100 for k in k_values}
    m.update({f"recall@{k}":np.mean(recall_at[k])*100 for k in k_values})
    m["map"]=np.mean(ap_list)*100
    return m


# ── Main ──────────────────────────────────────────────────
def run(dataset_name, model_key):
    ds_cfg = DATASET_CONFIGS[dataset_name]
    model_name = MODEL_MAP[model_key]

    out_dir = os.path.join(OUT_ROOT, f"{dataset_name}_results")
    cache_dir = os.path.join(out_dir, "embedding_cache", model_key)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 데이터 로드
    (corpus_ids, corpus_texts,
     test_qids, test_qtexts, qrels) = load_beir_data(dataset_name)

    # 2. 임베딩 (캐시 우선)
    print(f"\n[임베딩] {model_key} 캐시 확인...")
    c_path = os.path.join(cache_dir, "corpus_base.npy")
    q_path = os.path.join(cache_dir, "query_base.npy")

    embedder = None
    def _get_embedder():
        nonlocal embedder
        if embedder is None:
            print(f"[인코딩] {model_key} 모델 로드...")
            embedder = HFLLMPoolEmbedder(model_name)
        return embedder

    if os.path.exists(c_path):
        print(f"  corpus cache 로드")
        base_corpus = np.load(c_path)
    else:
        print(f"  corpus {len(corpus_ids):,}개 인코딩...")
        base_corpus = _l2(_get_embedder().encode(corpus_texts))
        np.save(c_path, base_corpus)

    if os.path.exists(q_path):
        print(f"  query cache 로드")
        base_query = np.load(q_path)
    else:
        print(f"  query {len(test_qids):,}개 인코딩...")
        base_query = _l2(_get_embedder().encode(test_qtexts))
        np.save(q_path, base_query)

    # fit data: corpus 랜덤 샘플
    fit_n = min(ds_cfg["fit_n"], len(corpus_ids))
    fit_path = os.path.join(cache_dir, f"fit_corpus_N{fit_n}.npy")
    if os.path.exists(fit_path):
        print(f"  fit cache 로드 (N={fit_n})")
        X_fit = np.load(fit_path)
    else:
        rng = np.random.RandomState(SEED)
        fit_idx = rng.choice(len(corpus_ids), fit_n, replace=False)
        fit_texts = [corpus_texts[i] for i in fit_idx]
        print(f"  fit {fit_n}개 인코딩...")
        X_fit = _l2(_get_embedder().encode(fit_texts))
        np.save(fit_path, X_fit)

    base_dim = base_corpus.shape[1]
    print(f"\n  base_dim={base_dim}  corpus={base_corpus.shape}  "
          f"query={base_query.shape}  fit={X_fit.shape}")

    # 3. Base 성능
    base_m = evaluate(base_corpus, corpus_ids, base_query, test_qids, qrels)
    print(f"\n[Base] NDCG@10={base_m['ndcg@10']:.2f}  "
          f"Recall@100={base_m['recall@100']:.2f}  MAP={base_m['map']:.2f}")

    # 4. 실험
    records = []
    total = len(TARGET_DIMS) * 3  # PCA-W, Soft-W, NCWP
    done  = 0

    for dim in TARGET_DIMS:
        _set_seed(SEED)

        # PCA-Whitening
        mu_pca, W_pca = pca_white(X_fit, dim)
        c = proj_linear(base_corpus, mu_pca, W_pca)
        q = proj_linear(base_query,  mu_pca, W_pca)
        m = evaluate(c, corpus_ids, q, test_qids, qrels)
        done += 1
        print(f"  [{done:>2}/{total}] PCA-White   dim={dim:<4} "
              f"NDCG@10={m['ndcg@10']:.2f}  Recall@100={m['recall@100']:.2f}")
        records.append({"method":"PCA-Whitening","dim":dim,**m})

        # Soft-Whitening
        mu_sw, W_sw = soft_white(X_fit, dim)
        c = proj_linear(base_corpus, mu_sw, W_sw)
        q = proj_linear(base_query,  mu_sw, W_sw)
        m = evaluate(c, corpus_ids, q, test_qids, qrels)
        done += 1
        print(f"  [{done:>2}/{total}] Soft-White  dim={dim:<4} "
              f"NDCG@10={m['ndcg@10']:.2f}  Recall@100={m['recall@100']:.2f}")
        records.append({"method":"Soft-Whitening","dim":dim,**m})

        # NCWP
        W, mu_in, mu_out, std_out = train_ncwp(X_fit, dim)
        c = proj_ncwp(base_corpus, W, mu_in, mu_out, std_out)
        q = proj_ncwp(base_query,  W, mu_in, mu_out, std_out)
        m = evaluate(c, corpus_ids, q, test_qids, qrels)
        done += 1
        print(f"  [{done:>2}/{total}] NCWP        dim={dim:<4} "
              f"NDCG@10={m['ndcg@10']:.2f}  Recall@100={m['recall@100']:.2f}")
        records.append({"method":"NCWP","dim":dim,**m})

    # 5. 저장
    df = pd.DataFrame(records)
    # base 행 추가
    base_row = {"method":"Base","dim":base_dim,**base_m}
    df_all = pd.concat([pd.DataFrame([base_row]), df], ignore_index=True)

    csv_path = os.path.join(out_dir, f"{model_key}_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    # 6. 요약 출력
    print(f"\n{'='*60}")
    print(f"  {ds_cfg['label']} — {model_key} — NDCG@10 Summary")
    print(f"{'='*60}")
    print(f"  {'Method':<18} " + "  ".join(f"dim={d:<4}" for d in TARGET_DIMS))
    print(f"  {'─'*55}")
    for method in ["PCA-Whitening","Soft-Whitening","NCWP"]:
        sub = df[df["method"]==method].set_index("dim")
        vals = [f"{sub.loc[d,'ndcg@10']:.2f}" if d in sub.index else "  N/A"
                for d in TARGET_DIMS]
        print(f"  {method:<18} " + "  ".join(f"{v:>7}" for v in vals))
    print(f"  {'Base (full dim)':<18} {base_m['ndcg@10']:>7.2f}")

    # 7. 플롯
    pdf_path = os.path.join(out_dir, f"{model_key}_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_results(df, base_m, model_key, ds_cfg["label"], out_dir, pdf=pdf)
    print(f"PDF 저장: {pdf_path}")
    return df_all


# ── Plotting ─────────────────────────────────────────────
def plot_results(df, base_m, model_key, dataset_label, out_dir, pdf=None):
    methods = ["PCA-Whitening", "Soft-Whitening", "NCWP"]
    colors  = {"PCA-Whitening":"#4CAF50","Soft-Whitening":"#FF9800","NCWP":"#E91E63"}
    markers = {"PCA-Whitening":"s",      "Soft-Whitening":"^",      "NCWP":"o"}
    lws     = {"PCA-Whitening":1.5,      "Soft-Whitening":1.5,      "NCWP":2.5}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.set_theme(style="whitegrid")

    metrics = [("ndcg@10","NDCG@10"), ("recall@100","Recall@100"), ("map","MAP")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in methods:
            sub = df[df["method"]==method].sort_values("dim")
            ax.plot(sub["dim"], sub[metric],
                    color=colors[method], marker=markers[method],
                    lw=lws[method], markersize=8, label=method, zorder=3)
        ax.axhline(base_m[metric], ls="--", color="gray", lw=1.5,
                   label=f"Base full-dim ({base_m[metric]:.1f})", alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TARGET_DIMS)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{int(v)}"))
        ax.set_xlabel("Projected Dimension", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{dataset_label} — {model_key}\n"
                 f"(corpus fit, dims={TARGET_DIMS})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    else:
        fname = os.path.join(out_dir, f"{model_key}_main.pdf")
        plt.savefig(fname, bbox_inches="tight")
        print(f"그래프 저장: {fname}")
    plt.close()


# ── Entry ─────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=list(DATASET_CONFIGS.keys()) + ["cqa-all"])
    p.add_argument("--model",   required=True, choices=list(MODEL_MAP.keys()))
    args = p.parse_args()
    if args.dataset == "cqa-all":
        for ds in ["cqa-english", "cqa-gaming", "cqa-physics"]:
            run(ds, args.model)
    else:
        run(args.dataset, args.model)
