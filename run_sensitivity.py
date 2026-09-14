"""
run_sensitivity.py  ── Quora Hyperparameter Sensitivity Analysis

리뷰어 질문 대응:
  "How sensitive is NCWP performance to hyperparameter choices
   (top-m filter, cosine threshold τ, top-K hard negatives, regularizer weights)
   on a new unlabeled corpus?"

파라미터 매핑:
  - top-m filter / top-K hard negatives → k  (kNN 이웃 크기, 동일 파라미터)
  - cosine threshold τ                  → cosine_tau  (positive 최소 유사도 필터)
  - regularizer weights                 → λ_contrast, λ_cov, λ_orth

실험 설계:
  - One-At-a-Time (OAT) sensitivity: 파라미터 하나씩 변화, 나머지 고정
  - 고정 dims: [32, 64]  (2**5, 2**6)
  - 캐시된 base embedding 재사용 → 인코딩 없이 수 분 내 완료

스윕 대상 (각 6값):
  k            : [2, 5, 10, 20, 40, 80]    ← top-m / top-K hard negatives
  cosine_tau   : [-2.0, 0.0, 0.3, 0.5, 0.7, 0.8]  ← cosine threshold τ (NEW)
  temperature  : [0.02, 0.05, 0.08, 0.12, 0.20, 0.40]
  λ_contrast   : [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
  λ_cov        : [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
  λ_orth       : [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

기본값 (center point):
  k=10, cosine_tau=-2.0(no filter), temperature=0.12,
  λ_contrast=1.0, λ_cov=1.0, λ_orth=1.0

결과 저장:
  /workspace/NCWP/quora_results/sensitivity/
    {model}_sensitivity_results_v2.csv
    {model}_sensitivity_ndcg10_v2.png

실행:
  python run_sensitivity.py --model qwen-4b
  python run_sensitivity.py --model llama-8b
"""

import os, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# ── NVRTC fix ──────────────────────────────────────────────
for _p in ["/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
           "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"]:
    if os.path.exists(_p):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _p not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_p}:{_ld}"
        break

# ── Config ────────────────────────────────────────────────
CACHE_ROOT  = "/workspace/NCWP/quora_results/embedding_cache"
OUT_ROOT    = "/workspace/NCWP/quora_results/sensitivity"
HF_CACHE    = "/workspace/RAG/code/Make_embedding/nanoGPT"
os.environ["HF_HOME"] = HF_CACHE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

# 고정 평가 크기 (기존 실험과 동일)
EVAL_Q = 2000
EVAL_C = 50000
FIT_N  = 3000    # fit용 query 샘플 수

# 민감도 분석 타겟 dim
TARGET_DIMS = [32, 64]   # 2**5, 2**6

# 기본 하이퍼파라미터 (center point)
DEFAULT_HP = dict(k=10, cosine_tau=-2.0, temperature=0.12, lc=1.0, lo=1.0, lk=1.0)

# OAT 스윕 정의
# cosine_tau=-2.0 은 "필터 없음" 기준점 (cosine ∈ [-1,1] 이므로 -2 = no filter)
SWEEPS = {
    "k (top-m/top-K)":  [2, 5, 10, 20, 40, 80],
    "cosine_tau (τ)":    [-2.0, 0.0, 0.3, 0.5, 0.7, 0.8],
    "temperature":       [0.02, 0.05, 0.08, 0.12, 0.20, 0.40],
    "λ_contrast":        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "λ_cov":             [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "λ_orth":            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}
# sweep key → HP dict key 매핑
SWEEP_KEY_MAP = {
    "k (top-m/top-K)": "k",
    "cosine_tau (τ)":   "cosine_tau",
    "temperature":      "temperature",
    "λ_contrast": "lk", "λ_cov": "lc", "λ_orth": "lo",
}

MODEL_MAP = {
    "qwen-4b":  ("Qwen/Qwen1.5-4B",              True),
    "qwen-8b":  ("Qwen/Qwen2-7B",                True),
    "llama-8b": ("meta-llama/Meta-Llama-3.1-8B", True),
}


# ── Reproducibility ──────────────────────────────────────
def _set_seed(s=SEED):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

_set_seed()


# ── LLM Embedder ─────────────────────────────────────────
class HFLLMPoolEmbedder:
    def __init__(self, model_name, use_auto_device_map=True):
        self.device = DEVICE
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"trust_remote_code": True,
              "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
              "attn_implementation": "eager"}
        n = torch.cuda.device_count(); print(f"[Info] {n} GPUs.")
        if n > 1:
            self.model = torch.nn.DataParallel(
                AutoModel.from_pretrained(model_name, **kw)).to(DEVICE)
            self.is_parallel = True
        else:
            if use_auto_device_map: kw["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(model_name, **kw)
            if not use_auto_device_map: self.model.to(DEVICE)
            self.is_parallel = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences, batch_size=256):
        embs, dev = [], DEVICE if self.is_parallel else getattr(self.model, "device", DEVICE)
        for i in range(0, len(sentences), batch_size):
            enc = self.tok(sentences[i:i+batch_size], padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(dev)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            x = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embs.append(F.normalize(x, dim=1).cpu().float().numpy())
        return np.concatenate(embs) if embs else np.array([])


# ── Data Loading ─────────────────────────────────────────
def load_quora_data():
    print("Loading Quora data...")
    ds = load_dataset("mteb/quora", "corpus", split="corpus")
    corpus_ids   = [r["_id"] for r in ds]
    corpus_texts = [(r["title"] + " " + r["text"]).strip() for r in ds]

    q_ds = load_dataset("mteb/quora", "queries", split="queries")
    qid2tx = {r["_id"]: r["text"] for r in q_ds}

    qr_ds = load_dataset("mteb/quora")
    def _build(split):
        d = defaultdict(dict)
        for r in qr_ds[split]:
            d[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
        return dict(d)

    qrels_test = _build("test")
    qrels_dev  = _build("dev")

    test_qids   = [q for q in sorted(qrels_test) if q in qid2tx]
    test_qtexts = [qid2tx[q] for q in test_qids]
    dev_qtexts  = [qid2tx[q] for q in sorted(qrels_dev) if q in qid2tx]

    print(f"  corpus: {len(corpus_ids):,}  test: {len(test_qids):,}  dev: {len(dev_qtexts):,}")
    return corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test, dev_qtexts


# ── Embedding Cache ───────────────────────────────────────
def _build_corpus_subset(corpus_ids, qrels_test, sub_test_qids, n_corpus, seed):
    """
    relevant 문서가 반드시 포함되도록 corpus subset 구성.
    : relevant 전체 + 랜덤 non-relevant (합계 = n_corpus)
    """
    relevant_set = set()
    for qid in sub_test_qids:
        if qid in qrels_test:
            relevant_set.update(qrels_test[qid].keys())

    cid2idx = {cid: i for i, cid in enumerate(corpus_ids)}
    rel_idx  = [cid2idx[cid] for cid in relevant_set if cid in cid2idx]
    non_rel_idx = [i for i in range(len(corpus_ids)) if corpus_ids[i] not in relevant_set]

    rng = np.random.RandomState(seed)
    remaining = max(0, n_corpus - len(rel_idx))
    sampled_nr = rng.choice(len(non_rel_idx), min(remaining, len(non_rel_idx)), replace=False)
    nr_idx = [non_rel_idx[i] for i in sampled_nr]

    c_idx = sorted(rel_idx + nr_idx)
    print(f"  corpus subset: relevant={len(rel_idx):,}  non-rel={len(nr_idx):,}  total={len(c_idx):,}")
    return c_idx


def get_embeddings(model_key, corpus_ids, corpus_texts,
                   test_qids, test_qtexts, qrels_test, dev_qtexts):
    """
    캐시 있으면 로드 (IDs 파일도 함께), 없으면 인코딩 후 저장.
    corpus subset은 relevant 문서 포함을 보장.
    """
    cache_dir = os.path.join(CACHE_ROOT, model_key)
    os.makedirs(cache_dir, exist_ok=True)

    tag_c  = f"sens_evalQ{EVAL_Q}_evalC{EVAL_C}"   # sensitivity 전용 tag
    tag_f  = f"fit_query_sample_N{FIT_N}"

    c_emb_path  = os.path.join(cache_dir, f"corpus_base_{tag_c}.npy")
    c_ids_path  = os.path.join(cache_dir, f"corpus_ids_{tag_c}.npy")
    q_emb_path  = os.path.join(cache_dir, f"test_queries_base_{tag_c}.npy")
    q_ids_path  = os.path.join(cache_dir, f"test_query_ids_{tag_c}.npy")
    ft_path     = os.path.join(cache_dir, f"{tag_f}.npy")

    # ── test query subset (먼저 결정해야 corpus relevant 계산 가능) ──
    if os.path.exists(q_ids_path):
        sub_test_qids = list(np.load(q_ids_path, allow_pickle=True))
        print(f"  query IDs cache 로드: {len(sub_test_qids):,}개")
    else:
        rng = np.random.RandomState(SEED)
        q_idx = rng.choice(len(test_qids), min(EVAL_Q, len(test_qids)), replace=False)
        sub_test_qids = [test_qids[i] for i in q_idx]
        np.save(q_ids_path, np.array(sub_test_qids))

    sub_test_qtexts = []
    qid2tx = {qid: txt for qid, txt in zip(test_qids, test_qtexts)}
    sub_test_qtexts = [qid2tx[qid] for qid in sub_test_qids]

    # ── corpus subset (relevant 보장) ──
    if os.path.exists(c_ids_path):
        c_idx = list(np.load(c_ids_path, allow_pickle=True).astype(int))
        sub_corpus_ids = [corpus_ids[i] for i in c_idx]
        print(f"  corpus IDs cache 로드: {len(sub_corpus_ids):,}개")
    else:
        c_idx = _build_corpus_subset(corpus_ids, qrels_test, sub_test_qids, EVAL_C, SEED)
        sub_corpus_ids = [corpus_ids[i] for i in c_idx]
        np.save(c_ids_path, np.array(c_idx))

    sub_corpus_texts = [corpus_texts[i] for i in c_idx]

    # ── fit query sample ──
    query_pool = test_qtexts + dev_qtexts
    rng2 = np.random.RandomState(SEED + 1)
    f_idx = rng2.choice(len(query_pool), min(FIT_N, len(query_pool)), replace=False)
    fit_texts = [query_pool[i] for i in sorted(f_idx)]

    # ── 인코딩 or 캐시 로드 ──
    need_model = not (os.path.exists(c_emb_path) and
                      os.path.exists(q_emb_path) and
                      os.path.exists(ft_path))
    embedder = None
    if need_model:
        print(f"\n[인코딩] {model_key} 모델 로드...")
        embedder = HFLLMPoolEmbedder(*MODEL_MAP[model_key])

    def _enc_or_load(path, texts, label):
        if os.path.exists(path):
            print(f"  {label} cache 로드: {path}")
            return np.load(path)
        print(f"  {label} {len(texts):,}개 인코딩...")
        emb = embedder.encode(texts).astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        np.save(path, emb); return emb

    base_corpus = _enc_or_load(c_emb_path, sub_corpus_texts, "corpus")
    base_query  = _enc_or_load(q_emb_path, sub_test_qtexts,  "queries")
    X_fit       = _enc_or_load(ft_path,    fit_texts,         "fit data")
    if not os.path.exists(ft_path.replace('.npy','')+'_done.txt'):
        open(ft_path, 'a').close()  # 빈 마커

    return base_corpus, base_query, X_fit, sub_corpus_ids, sub_test_qids


# ── NCWP Core ─────────────────────────────────────────────
def _orth_penalty(W):
    WT_W = W.T @ W
    return ((WT_W - torch.eye(WT_W.shape[0], device=W.device))**2).mean()

def _cov_penalty(Z):
    B = Z.shape[0]
    if B <= 1: return torch.tensor(0., device=Z.device)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / (B - 1)
    return ((Cov - torch.eye(Cov.shape[0], device=Z.device))**2).mean()

def _zca_shrink(X, shrink=0.08, eps=1e-6):
    orig = X.dtype; X = X.double()
    mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=eps))) @ evec.T
    return mu.squeeze(0).to(orig), S.to(orig)

@torch.no_grad()
def _topk_knn_with_sims(X, k, chunk=2048):
    """top-k 이웃 인덱스와 코사인 유사도를 함께 반환."""
    N = X.shape[0]
    idx = torch.empty((N, k), dtype=torch.long,  device="cpu")
    sim = torch.empty((N, k), dtype=torch.float, device="cpu")
    for s in range(0, N, chunk):
        e = min(N, s+chunk); s_ = X[s:e] @ X.T
        s_[:, torch.arange(s, e, device=X.device)] = -1e9
        topk = torch.topk(s_, k=k, dim=1)
        idx[s:e] = topk.indices.cpu()
        sim[s:e] = topk.values.cpu()
    return idx, sim

def train_ncwp(X_np, rank, k=10, temperature=0.12, lc=1.0, lo=1.0, lk=1.0,
               cosine_tau=-2.0, epochs=20, lr=8e-3, shrink=0.08):
    """
    X_np: (N, D) float32 → returns (W, mu_in, mu_out, std_out)

    cosine_tau : positive neighbor의 최소 코사인 유사도 (whitened 공간 기준).
                 -2.0 = 필터 없음 (기본값, 모든 k-NN 이웃 허용).
                 0.0~0.9 = τ 이상인 이웃만 positive로 허용.
                 유효 이웃 0개면 모든 k-NN으로 fallback.
    """
    X = torch.from_numpy(X_np).to(DEVICE); N, D = X.shape
    mu, S = _zca_shrink(X, shrink)
    Xw = F.normalize((X - mu) @ S, dim=1)
    k_ = max(1, min(k, N-1))
    knn_idx, knn_sim = _topk_knn_with_sims(Xw, k_)  # (N, k), (N, k) on CPU
    r = min(rank, D)
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr)
    B = 64
    tau_tensor = torch.tensor(cosine_tau, dtype=torch.float)

    for _ in range(epochs):
        batch_idx = torch.randperm(N)[:B]
        # cosine_tau 필터 적용하여 valid positive 샘플링
        pos_list = []
        for anc in batch_idx.tolist():
            valid_mask = knn_sim[anc] >= tau_tensor
            valid_nb = knn_idx[anc][valid_mask]
            if len(valid_nb) == 0:
                valid_nb = knn_idx[anc]          # fallback: 모든 k-NN 사용
            pos_list.append(valid_nb[torch.randint(0, len(valid_nb), (1,))].item())
        pos = torch.tensor(pos_list, device=DEVICE)

        Z = F.normalize(torch.cat([Xw[batch_idx.to(DEVICE)], Xw[pos]], 0) @ W, dim=1)
        sim = (Z @ Z.T) / temperature; sim.fill_diagonal_(-1e9)
        loss = (lk * (-(sim[:B].gather(1, torch.arange(B, 2*B, device=DEVICE).view(-1,1)).squeeze(1)
                        - torch.logsumexp(sim[:B], 1)).mean())
                + lc * _cov_penalty(Z) + lo * _orth_penalty(W))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        SW = S @ W; Y = (X - mu) @ SW
    return SW.cpu().numpy(), mu.cpu().numpy(), Y.mean(0).cpu().numpy(), torch.clamp(Y.std(0), min=1e-6).cpu().numpy()


# ── Projection & Eval ─────────────────────────────────────
def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def proj_ncwp(X, W, mu_in, mu_out, std_out):
    return _l2(((X - mu_in) @ W - mu_out) / std_out)

def evaluate(corpus_embs, corpus_ids, query_embs, query_ids, qrels,
             k_values=(10, 100), chunk=500):
    max_k = max(k_values)
    ndcg_at = {k: [] for k in k_values}
    recall_at= {k: [] for k in k_values}
    ap_list  = []
    for qi in range(0, len(query_ids), chunk):
        qe = min(len(query_ids), qi+chunk)
        sim = query_embs[qi:qe] @ corpus_embs.T
        if sim.shape[1] <= max_k:
            top_idx = np.argsort(-sim, axis=1)
        else:
            part  = np.argpartition(-sim, max_k, axis=1)[:, :max_k]
            order = np.argsort(-sim[np.arange(len(part))[:,None], part], axis=1)
            top_idx = part[np.arange(len(part))[:,None], order]
        for ci, q in enumerate(range(qi, qe)):
            qid = query_ids[q]
            if qid not in qrels: continue
            rel = qrels[qid]; ranked = [corpus_ids[i] for i in top_idx[ci]]
            ideal = sorted(rel.values(), reverse=True)
            for k in k_values:
                top = ranked[:k]
                dcg  = sum((2**rel.get(d,0)-1)/math.log2(r+2) for r,d in enumerate(top))
                idcg = sum((2**rv-1)/math.log2(r+2) for r,rv in enumerate(ideal[:k]))
                ndcg_at[k].append(dcg/idcg if idcg>0 else 0.)
                relset = {d for d,s in rel.items() if s>0}
                recall_at[k].append(len(set(top)&relset)/len(relset) if relset else 0.)
            total_rel = sum(1 for s in rel.values() if s>0)
            nr=0; ap=0.
            for r,d in enumerate(ranked,1):
                if rel.get(d,0)>0: nr+=1; ap+=nr/r
            ap_list.append(ap/total_rel if total_rel>0 else 0.)
    m = {f"ndcg@{k}": np.mean(ndcg_at[k])*100 for k in k_values}
    m.update({f"recall@{k}": np.mean(recall_at[k])*100 for k in k_values})
    m["map"] = np.mean(ap_list)*100
    return m


# ── PCA-White (reference) ─────────────────────────────────
def pca_white_proj(X_fit, base_dim, target_k, shrink=0.08, eps=1e-6):
    X = torch.from_numpy(X_fit).to(DEVICE).double()
    N, D = X.shape; mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(N-1,1); tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev,[0]); evec = torch.flip(evec,[1])
    k = min(target_k, D)
    W = evec[:,:k] @ torch.diag(1./torch.sqrt(torch.clamp(ev[:k], min=eps)))
    return mu.squeeze(0).cpu().float().numpy(), W.cpu().float().numpy()


# ── Main ──────────────────────────────────────────────────
def run_sensitivity(model_key):
    out_dir = OUT_ROOT
    os.makedirs(out_dir, exist_ok=True)

    # 1. 데이터 로드
    (corpus_ids, corpus_texts,
     test_qids, test_qtexts,
     qrels_test, dev_qtexts) = load_quora_data()

    # 2. 임베딩 (캐시 우선)
    print(f"\n[임베딩] {model_key} 캐시 확인...")
    (base_corpus, base_query,
     X_fit, sub_corpus_ids, sub_test_qids) = get_embeddings(
        model_key, corpus_ids, corpus_texts,
        test_qids, test_qtexts, qrels_test, dev_qtexts)

    # sub_test_qids에 맞게 qrels 필터
    qrels_sub = {qid: qrels_test[qid] for qid in sub_test_qids if qid in qrels_test}

    # 진단: relevant 문서가 corpus subset에 몇 개나 있는지
    sub_corpus_set = set(sub_corpus_ids)
    found = sum(len(set(v.keys()) & sub_corpus_set) for v in qrels_sub.values())
    total_rel = sum(len(v) for v in qrels_sub.values())
    print(f"  relevant 커버리지: {found}/{total_rel} ({found/max(total_rel,1)*100:.1f}%)")

    base_dim = base_corpus.shape[1]
    print(f"  base_dim={base_dim}  corpus={base_corpus.shape}  query={base_query.shape}  fit={X_fit.shape}")

    # 3. Base 성능 (dim=full)
    base_m = evaluate(base_corpus, sub_corpus_ids, base_query, sub_test_qids, qrels_sub)
    print(f"\n[Base] NDCG@10={base_m['ndcg@10']:.2f}  Recall@100={base_m['recall@100']:.2f}")

    # 4. PCA-White reference (각 target dim)
    pca_ref = {}
    for dim in TARGET_DIMS:
        mu_pca, W_pca = pca_white_proj(X_fit, base_dim, dim)
        c = _l2((base_corpus - mu_pca) @ W_pca)
        q = _l2((base_query  - mu_pca) @ W_pca)
        m = evaluate(c, sub_corpus_ids, q, sub_test_qids, qrels_sub)
        pca_ref[dim] = m
        print(f"[PCA-White] dim={dim}  NDCG@10={m['ndcg@10']:.2f}")

    # 5. OAT Sweep
    records = []
    total = sum(len(v) for v in SWEEPS.values()) * len(TARGET_DIMS)
    done  = 0

    for sweep_name, values in SWEEPS.items():
        hp_key = SWEEP_KEY_MAP[sweep_name]
        print(f"\n{'─'*55}")
        print(f"  Sweep: {sweep_name}  values={values}")
        print(f"{'─'*55}")

        for val in values:
            # 현재 HP 구성
            hp = dict(DEFAULT_HP)
            hp[hp_key] = val

            for dim in TARGET_DIMS:
                _set_seed(SEED)
                W, mu_in, mu_out, std_out = train_ncwp(
                    X_fit, rank=dim,
                    k=hp["k"], temperature=hp["temperature"],
                    lc=hp["lc"], lo=hp["lo"], lk=hp["lk"],
                    cosine_tau=hp.get("cosine_tau", -2.0))

                c = proj_ncwp(base_corpus, W, mu_in, mu_out, std_out)
                q = proj_ncwp(base_query,  W, mu_in, mu_out, std_out)
                m = evaluate(c, sub_corpus_ids, q, sub_test_qids, qrels_sub)

                rec = {
                    "sweep_param": sweep_name,
                    "param_value": val,
                    "dim": dim,
                    **{f"hp_{k}": v for k, v in hp.items()},
                    **m,
                }
                records.append(rec)
                done += 1
                print(f"  [{done:>3}/{total}] {sweep_name}={val:<6}  dim={dim:<4} "
                      f"NDCG@10={m['ndcg@10']:.2f}  Recall@100={m['recall@100']:.2f}")

    # 6. 저장
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"{model_key}_sensitivity_results_v2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    # 7. 플롯
    pdf_path = os.path.join(out_dir, f"{model_key}_sensitivity_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_sensitivity(df, model_key, base_m, pca_ref, out_dir, pdf=pdf)
    print(f"PDF 저장: {pdf_path}")
    return df


# ── Plotting ─────────────────────────────────────────────
def plot_sensitivity(df, model_key, base_m, pca_ref, out_dir,
                     metric="ndcg@10", pdf=None):
    sweep_names = list(SWEEPS.keys())
    n = len(sweep_names)
    colors = {32: "#E91E63", 64: "#2196F3"}
    marks  = {32: "o",       64: "s"}

    fig, axes = plt.subplots(1, n, figsize=(3.8*n, 5), sharey=False)
    sns.set_theme(style="whitegrid")

    for ax, sname in zip(axes, sweep_names):
        sub = df[df["sweep_param"] == sname]

        for dim in TARGET_DIMS:
            d = sub[sub["dim"] == dim].sort_values("param_value")
            ax.plot(d["param_value"], d[metric],
                    color=colors[dim], marker=marks[dim],
                    label=f"NCWP dim={dim}", lw=2, markersize=7, zorder=3)

        # 기준선
        ax.axhline(base_m[metric], ls="--", color="black", lw=1.2,
                   label=f"Base ({base_m[metric]:.1f})", alpha=0.7)
        for dim, ref_m in pca_ref.items():
            ax.axhline(ref_m[metric], ls=":", color=colors[dim], lw=1.2,
                       label=f"PCA-W dim={dim} ({ref_m[metric]:.1f})", alpha=0.6)

        # x축 스케일 설정
        vals = SWEEPS[sname]
        if sname == "cosine_tau (τ)":
            # linear scale; -2.0은 "no filter" 레이블로 표기
            ax.set_xticks(vals)
            ax.set_xticklabels(["no\nfilter" if v == -2.0 else str(v) for v in vals],
                               fontsize=8)
        elif min(vals) > 0 and max(vals) / min(vals) >= 10:
            ax.set_xscale("log")
            ax.set_xticks(vals)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            ax.set_xticks(vals)

        ax.set_xlabel(sname, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric.upper() if sname == sweep_names[0] else "")
        ax.set_title(sname, fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, which="both", ls="--", alpha=0.4)

    # 공통 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"Quora — {model_key} — Hyperparameter Sensitivity\n"
                 f"(evalQ={EVAL_Q}, evalC={EVAL_C}, fit_N={FIT_N})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(bbox_inches="tight")
    else:
        fname = os.path.join(out_dir, f"{model_key}_sensitivity_{metric.replace('@','')}_v2.pdf")
        plt.savefig(fname, bbox_inches="tight")
        print(f"그래프 저장: {fname}")
    plt.close()

    # 추가: 2개 dim × 5개 sweep 요약 표 출력
    print(f"\n{'='*60}")
    print(f"  Sensitivity Summary ({metric.upper()})  —  {model_key}")
    print(f"{'='*60}")
    for sname in sweep_names:
        sub = df[df["sweep_param"] == sname]
        print(f"\n  {sname}:")
        for dim in TARGET_DIMS:
            d = sub[sub["dim"] == dim].sort_values("param_value")
            vals_m = d[metric].values
            print(f"    dim={dim:<4}  min={vals_m.min():.2f}  max={vals_m.max():.2f}"
                  f"  range={vals_m.max()-vals_m.min():.2f}  "
                  f"({d['param_value'].values[vals_m.argmin()]}→{d['param_value'].values[vals_m.argmax()]})")


# ── Entry ─────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    args = p.parse_args()
    run_sensitivity(args.model)
