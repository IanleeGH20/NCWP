"""
run_scifact_experiment.py

SciFact retrieval dataset으로 NCWP 실험 수행.
run_ncwp_fewshot_extended.py 와 동일한 NCWP 학습 로직을 사용하되,
평가는 STS Spearman 대신 NDCG@10 / Recall@100 / MAP 으로 수행.

컨테이너 경로 기준:
  - nanoGPT  : /workspace/RAG/code/Make_embedding/nanoGPT
  - 결과 저장 : /workspace/NCWP/scifact_results

Dataset (mteb/scifact):
  corpus  : 5,183 documents (title + text)
  queries : 1,109 queries
  qrels   : train 919 rows / test 339 rows  (query-id, corpus-id, score)

Fit 모드 (--fit_mode):
  corpus  [방법 A] : corpus 문서만으로 비지도 kNN 학습 (기본값)
  mixed   [방법 C] : train query + corpus 혼합으로 비지도 kNN 학습

평가: test qrels 기준 retrieval 성능 (NDCG@10, Recall@100, MAP)

실행 예시 (컨테이너 내부):
  cd /workspace/NCWP
  python run_scifact_experiment.py --model llama-8b --fit_mode corpus
  python run_scifact_experiment.py --model llama-8b --fit_mode mixed
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# NVRTC Error Fix
# ============================================================
for _nvrtc in [
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib",
]:
    if os.path.exists(_nvrtc):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _nvrtc not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_nvrtc}:{_ld}"
        break

# ============================================================
# Configuration
# ============================================================
class SciFACTConfig:
    seed          = 42
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    k_neighbors   = 10

    lambda_cov      = 1.0
    lambda_orth     = 1.0
    lambda_contrast = 1.0

    epochs      = 20
    lr          = 8e-3
    temperature = 0.12
    shrink      = 0.08

    # HuggingFace 모델 캐시 (기존 nanoGPT 캐시 재사용)
    hf_cache_dir = "/workspace/RAG/code/Make_embedding/nanoGPT"
    # 결과 저장 루트 (fit_mode 별로 하위 폴더가 생성됨)
    output_root  = "/workspace/NCWP/scifact_results"

cfg = SciFACTConfig()
os.makedirs(cfg.output_root, exist_ok=True)
os.environ["HF_HOME"] = cfg.hf_cache_dir


# ============================================================
# Reproducibility
# ============================================================
def _set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

_set_seed(cfg.seed)


# ============================================================
# NCWP Core Helpers  (run_ncwp_fewshot_extended.py 와 동일)
# ============================================================
def _orth_penalty(W: torch.Tensor) -> torch.Tensor:
    WT_W = W.T @ W
    I = torch.eye(WT_W.shape[0], device=W.device, dtype=W.dtype)
    return torch.mean((WT_W - I) ** 2)

def _cov_penalty(Z: torch.Tensor) -> torch.Tensor:
    B = Z.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / (B - 1)
    I = torch.eye(Cov.shape[0], device=Z.device, dtype=Z.dtype)
    return torch.mean((Cov - I) ** 2)

def _zca_shrink(X: torch.Tensor, shrink: float = 0.10, eps: float = 1e-6):
    orig_dtype = X.dtype
    X = X.double()
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    N = X.shape[0]
    Cov = (Xc.T @ Xc) / max(N - 1, 1)
    D = Cov.shape[0]
    trace = torch.trace(Cov)
    mean_eig = trace / D
    Cov_shrunk = (1.0 - shrink) * Cov + shrink * mean_eig * torch.eye(D, device=X.device, dtype=X.dtype)
    evals, evecs = torch.linalg.eigh(Cov_shrunk)
    evals = torch.clamp(evals, min=eps)
    S = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
    return mu.squeeze(0).to(orig_dtype), S.to(orig_dtype)

@torch.no_grad()
def _topk_cosine_knn(X: torch.Tensor, k: int, chunk: int = 2048):
    N = X.shape[0]
    idx_all = torch.empty((N, k), dtype=torch.long, device="cpu")
    X_T = X.T
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Q = X[s:e]
        sims = Q @ X_T
        row_idx = torch.arange(s, e, device=X.device)
        sims[:, row_idx] = -1e9
        topk = torch.topk(sims, k=k, dim=1).indices.detach().cpu()
        idx_all[s:e] = topk
    return idx_all

def fit_lpp_projector(X: np.ndarray, k: int, n_neighbors: int = 10, reg: float = 1e-3):
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    Xn = Xc / norms
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=min(n_neighbors, N - 1) - 1, axis=1)[:, :n_neighbors]
    Wg = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), idx.shape[1])
    cols = idx.reshape(-1)
    Wg[rows, cols] = 1.0
    Wg = np.maximum(Wg, Wg.T)
    d = Wg.sum(axis=1)
    Dg = np.diag(d)
    Lg = Dg - Wg
    XtDX = Xc.T @ Dg @ Xc + reg * np.eye(D, dtype=np.float32)
    XtLX = Xc.T @ Lg @ Xc
    try:
        A = np.linalg.solve(XtDX, XtLX)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(XtDX) @ XtLX
    w, V = np.linalg.eigh(A)
    V = V[:, np.argsort(w)]
    r = min(k, D)
    return V[:, :r].astype(np.float32), mu.squeeze(0).astype(np.float32)

def fit_random_projector(dim_in: int, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    M = rng.normal(size=(dim_in, k)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return Q[:, :k].astype(np.float32)

def compute_pca_whitening_matrix(X_np, target_dim, shrink=0.08, eps=1e-6):
    if np.isnan(X_np).any():
        X_np = np.nan_to_num(X_np)
    X = torch.from_numpy(X_np).to(cfg.device).double()
    N, D = X.shape
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    Cov = (Xc.T @ Xc) / max(N - 1, 1)
    trace = torch.trace(Cov)
    mean_eig = trace / D
    Cov_shrunk = (1.0 - shrink) * Cov + shrink * mean_eig * torch.eye(D, device=X.device, dtype=X.dtype)
    evals, evecs = torch.linalg.eigh(Cov_shrunk)
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    scales = evals.cpu().numpy()
    comps = evecs.cpu().numpy()
    k_dim = min(target_dim, D)
    evals_k = torch.clamp(evals[:k_dim], min=eps)
    evecs_k = evecs[:, :k_dim]
    W = evecs_k @ torch.diag(1.0 / torch.sqrt(evals_k))
    return (mu.squeeze(0).cpu().numpy().astype(np.float32),
            W.cpu().numpy().astype(np.float32),
            comps.astype(np.float32),
            scales.astype(np.float32))


# ============================================================
# NCWP Trainer  (run_ncwp_fewshot_extended.py 와 동일)
# ============================================================
class NewProposalTrainer:
    def __init__(self, rank: int, k_neighbors: int = 40, **kwargs):
        self.rank            = rank
        self.k_neighbors     = k_neighbors
        self.device          = cfg.device
        self.lr              = kwargs.get("lr", 8e-3)
        self.epochs          = kwargs.get("max_epochs", 20)
        self.shrink          = kwargs.get("shrink", 0.08)
        self.temperature     = kwargs.get("temperature", 0.12)
        self.lambda_cov      = kwargs.get("lambda_cov", 1.0)
        self.lambda_orth     = kwargs.get("lambda_orth", 1.0)
        self.lambda_contrast = kwargs.get("lambda_contrast", 1.0)

    def _contrastive_loss(self, logits_row, pos_index):
        pos_logits = logits_row.gather(1, pos_index.view(-1, 1)).squeeze(1)
        denom = torch.logsumexp(logits_row, dim=1)
        return -(pos_logits - denom).mean()

    def fit(self, X_np):
        X = torch.from_numpy(X_np).to(self.device)
        N, D = X.shape
        mu_in_t, S = _zca_shrink(X, shrink=self.shrink)
        Xw = F.normalize((X - mu_in_t) @ S, dim=1)
        k = max(1, min(self.k_neighbors, max(1, N - 1)))
        with torch.no_grad():
            knn_idx = _topk_cosine_knn(Xw, k=k)
        r = min(self.rank, D)
        W = torch.nn.Parameter(
            torch.randn(D, r, device=self.device) * (1.0 / math.sqrt(D))
        )
        opt = torch.optim.AdamW([W], lr=self.lr)
        batch_size = 64
        for _ in range(self.epochs):
            idx = torch.randperm(N)[:batch_size]
            pos_idx = knn_idx[idx.cpu(), torch.randint(0, k, (batch_size,))].to(self.device)
            Xa = Xw[idx.to(self.device)]
            Xb = Xw[pos_idx]
            Z = F.normalize(torch.cat([Xa, Xb], dim=0) @ W, dim=1)
            sim = (Z @ Z.T) / self.temperature
            sim.fill_diagonal_(-1e9)
            B = batch_size
            loss = (
                self.lambda_contrast * self._contrastive_loss(
                    sim[:B], torch.arange(B, 2 * B, device=self.device)
                )
                + self.lambda_cov  * _cov_penalty(Z)
                + self.lambda_orth * _orth_penalty(W)
            )
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            S_W = S @ W
            Y = (X - mu_in_t) @ S_W
            mu_out  = Y.mean(0)
            std_out = torch.clamp(Y.std(0), min=1e-6)
        return (S_W.cpu().numpy(), mu_in_t.cpu().numpy(),
                mu_out.cpu().numpy(), std_out.cpu().numpy())


# ============================================================
# Model Wrappers  (run_ncwp_fewshot_extended.py 와 동일)
# ============================================================
class WrappedModel:
    def __init__(self, base): self.base = base
    def encode(self, sents, **kw): return self.base.encode(sents, **kw)

class WrappedLinear(WrappedModel):
    def __init__(self, base, W, mu_in):
        super().__init__(base); self.W = W; self.mu_in = mu_in
    def encode(self, sents, **kw):
        X = self.base.encode(sents, **kw).astype(np.float32)
        Z = (X - self.mu_in) @ self.W
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

class WrappedSoftWhitening(WrappedModel):
    def __init__(self, base, mu, comps, scales, k, alpha=0.1):
        super().__init__(base)
        self.mu = mu; self.W = comps[:, :k]; self.s = scales[:k]; self.alpha = alpha
    def encode(self, sents, **kw):
        X = self.base.encode(sents, **kw).astype(np.float32)
        Z = (X - self.mu) @ self.W
        s_mean = self.s.mean() if self.s.size > 0 else 1.0
        denom = np.sqrt((1.0 - self.alpha) * self.s + self.alpha * s_mean + 1e-8)
        Z = Z / denom
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

class WrappedNCWP(WrappedModel):
    def __init__(self, base, W, mu_in, mu_out, std_out):
        super().__init__(base)
        self.W = W; self.mu_in = mu_in; self.mu_out = mu_out; self.std_out = std_out
    def encode(self, sents, **kw):
        X = self.base.encode(sents, **kw).astype(np.float32)
        Z = ((X - self.mu_in) @ self.W - self.mu_out) / self.std_out
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

class HFLLMPoolEmbedder:
    def __init__(self, model_name, device="cuda", use_auto_device_map=True):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "attn_implementation": "eager",
        }
        n_gpus = torch.cuda.device_count()
        print(f"[Info] Found {n_gpus} GPUs.")
        if n_gpus > 1:
            print(f"[Info] DataParallel on {n_gpus} GPUs.")
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(device)
            self.is_parallel = True
        else:
            if use_auto_device_map:
                model_kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            if not use_auto_device_map:
                self.model.to(device)
            self.is_parallel = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences, batch_size=256, **kwargs):
        embs = []
        target_device = (self.device if self.is_parallel
                         else getattr(self.model, "device", self.device))
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(target_device)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            x = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embs.append(F.normalize(x, p=2, dim=1).cpu().float().numpy())
        return np.concatenate(embs, axis=0) if embs else np.array([])


# ============================================================
# SciFact Data Loading
# ============================================================
def load_scifact():
    """
    Returns:
        corpus_ids        : list[str]
        corpus_texts      : list[str]  (title + ' ' + text)
        test_qids         : list[str]
        test_qtexts       : list[str]
        qrels_test        : dict  { query_id: {corpus_id: score} }
        train_qtexts      : list[str]  (train split 쿼리 텍스트, 방법 C용)
    """
    print("Loading SciFact corpus  (mteb/scifact corpus)...")
    corpus_ds    = load_dataset("mteb/scifact", "corpus", split="corpus")
    corpus_ids   = [row["_id"] for row in corpus_ds]
    corpus_texts = [(row["title"] + " " + row["text"]).strip() for row in corpus_ds]

    print("Loading SciFact queries (mteb/scifact queries)...")
    queries_ds   = load_dataset("mteb/scifact", "queries", split="queries")
    qid2txt      = {row["_id"]: row["text"] for row in queries_ds}

    print("Loading SciFact qrels   (mteb/scifact train+test)...")
    qrels_ds     = load_dataset("mteb/scifact")   # splits: train / test

    def _build(split_ds):
        d = defaultdict(dict)
        for row in split_ds:
            d[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
        return dict(d)

    qrels_train  = _build(qrels_ds["train"])
    qrels_test   = _build(qrels_ds["test"])

    # test queries (평가용)
    test_qids    = [qid for qid in sorted(qrels_test.keys()) if qid in qid2txt]
    test_qtexts  = [qid2txt[qid] for qid in test_qids]

    # train queries (방법 C fit용)
    train_qids   = [qid for qid in sorted(qrels_train.keys()) if qid in qid2txt]
    train_qtexts = [qid2txt[qid] for qid in train_qids]

    print(f"  corpus         : {len(corpus_ids):,} docs")
    print(f"  queries (test) : {len(test_qids)}")
    print(f"  queries (train): {len(train_qtexts)}  ← 방법 C fit용")
    print(f"  qrels   (test) : {sum(len(v) for v in qrels_test.values())} pairs")
    return corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test, train_qtexts


# ============================================================
# Retrieval Evaluation  (NDCG@k, Recall@k, MAP)
# ============================================================
def evaluate_retrieval(model, corpus_texts, corpus_ids, query_texts, query_ids,
                       qrels, k_values=(10, 100), batch_size=256,
                       return_anisotropy=False, return_self_sim=False):
    """
    Primary metric : NDCG@10  (STS 실험의 Spearman 역할)
    """
    print(f"    Encoding {len(corpus_texts):,} corpus docs...", flush=True)
    corpus_embs = model.encode(corpus_texts, batch_size=batch_size)
    corpus_embs = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-12)

    print(f"    Encoding {len(query_texts)} queries...", flush=True)
    query_embs  = model.encode(query_texts, batch_size=batch_size)
    query_embs  = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12)

    ndcg_at   = {k: [] for k in k_values}
    recall_at = {k: [] for k in k_values}
    ap_list   = []
    max_k     = max(k_values)

    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel        = qrels[qid]
        sims       = corpus_embs @ query_embs[qi]
        ranked_ids = [corpus_ids[i] for i in np.argsort(-sims)[:max_k]]
        ideal_rels = sorted(rel.values(), reverse=True)

        for k in k_values:
            top_ids = ranked_ids[:k]
            dcg  = sum((2 ** rel.get(d, 0) - 1) / math.log2(r + 2)
                       for r, d in enumerate(top_ids))
            idcg = sum((2 ** rv - 1) / math.log2(r + 2)
                       for r, rv in enumerate(ideal_rels[:k]))
            ndcg_at[k].append(dcg / idcg if idcg > 0 else 0.0)

            relevant = {d for d, s in rel.items() if s > 0}
            recall_at[k].append(len(set(top_ids) & relevant) / len(relevant)
                                 if relevant else 0.0)

        total_rel = sum(1 for s in rel.values() if s > 0)
        num_rel = 0; ap = 0.0
        for r, d in enumerate(ranked_ids, 1):
            if rel.get(d, 0) > 0:
                num_rel += 1; ap += num_rel / r
        ap_list.append(ap / total_rel if total_rel > 0 else 0.0)

    metrics = {f"ndcg@{k}": np.mean(ndcg_at[k]) * 100   for k in k_values}
    metrics.update({f"recall@{k}": np.mean(recall_at[k]) * 100 for k in k_values})
    metrics["map"] = np.mean(ap_list) * 100

    extras = []
    X = corpus_embs
    if return_anisotropy:
        mu = X.mean(axis=0, keepdims=True)
        mu_hat = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)
        extras.append(float((X @ mu_hat.T).mean()))
    if return_self_sim:
        N = X.shape[0]
        if N < 10000:
            G = X @ X.T
            extras.append(float((G.sum() - np.trace(G)) / (N * (N - 1))))
        else:
            idx1 = np.random.randint(0, N, 200000)
            idx2 = np.random.randint(0, N, 200000)
            mask = idx1 != idx2
            extras.append(float((X[idx1[mask]] * X[idx2[mask]]).sum(axis=1).mean()))

    return (metrics, extras) if extras else metrics


# ============================================================
# Plotting
# ============================================================
def plot_results(results_list, filename_suffix, primary="ndcg@10", pdf=None):
    sns.set_theme(style="whitegrid")
    MARKERS = {
        "Base": "o", "PCA-White": "s", "Soft-White": "D",
        "LPP": "x", "Random": ".", "NCWP": "^",
        "NCWP_Default": "^", "NCWP_HighCont": "v",
        "NCWP_HighCov": "<", "NCWP_HighOrth": ">",
        "NCWP_k5": "1", "NCWP_k20": "p", "NCWP_k40": "*",
    }
    methods = defaultdict(lambda: {"dims": [], "scores": [], "ani": []})
    for r in results_list:
        m = r["method"]
        methods[m]["dims"].append(r["dim"])
        methods[m]["scores"].append(r[primary])
        if "anisotropy" in r:
            methods[m]["ani"].append(r["anisotropy"])

    # Primary metric plot
    fig1 = plt.figure(figsize=(12, 7))
    for m, data in methods.items():
        zipped = sorted(zip(data["dims"], data["scores"]))
        dims, scores = zip(*zipped)
        if m == "Base":
            plt.axhline(y=scores[0], linestyle="--",
                        label=f"Base ({scores[0]:.2f})", color="black", alpha=0.7)
        else:
            plt.plot(dims, scores, marker=MARKERS.get(m, "o"), label=m, linewidth=1.5)
    plt.xscale("log", base=2)
    plt.title(f"SciFact {primary.upper()} vs Dimension ({filename_suffix})")
    plt.xlabel("Dimension (log scale)")
    plt.ylabel(primary.upper())
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig1, bbox_inches="tight")
    else:
        plt.savefig(os.path.join(cfg.output_dir, f"{filename_suffix}.pdf"), bbox_inches="tight")
    plt.close()

    # Anisotropy plot
    if any(len(d["ani"]) > 0 for d in methods.values()):
        fig2 = plt.figure(figsize=(12, 7))
        for m, data in methods.items():
            if not data["ani"]: continue
            zipped = sorted(zip(data["dims"], data["ani"]))
            dims, anis = zip(*zipped)
            if m == "Base":
                plt.axhline(y=anis[0], linestyle="--",
                            label=f"Base ({anis[0]:.4f})", color="black", alpha=0.7)
            else:
                plt.plot(dims, anis, marker=MARKERS.get(m, "o"), label=m, linewidth=1.5)
        plt.xscale("log", base=2)
        plt.title(f"SciFact Anisotropy vs Dimension ({filename_suffix})")
        plt.xlabel("Dimension (log scale)")
        plt.ylabel("Avg Cosine Similarity (Anisotropy)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        if pdf is not None:
            pdf.savefig(fig2, bbox_inches="tight")
        else:
            plt.savefig(os.path.join(cfg.output_dir, f"{filename_suffix}_anisotropy.pdf"), bbox_inches="tight")
        plt.close()

    print(f"  Plots saved → {cfg.output_dir}/{filename_suffix}*.pdf")


# ============================================================
# Main Experiment
# ============================================================
def run_experiment(model_key, fit_mode="corpus"):
    """
    fit_mode:
      'corpus' [방법 A] : corpus 문서만으로 비지도 kNN 학습
      'mixed'  [방법 C] : train query + corpus 혼합으로 비지도 kNN 학습
    """
    assert fit_mode in ("corpus", "mixed"), f"Unknown fit_mode: {fit_mode}"

    # fit_mode 별 결과 폴더 분리
    cfg.output_dir = os.path.join(cfg.output_root, f"fit_{fit_mode}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" SciFact Experiment  |  Model: {model_key}  |  fit_mode: {fit_mode}")
    print(f" Results → {cfg.output_dir}")
    print(f"{'='*60}")

    # 1. Data
    corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test, train_qtexts = load_scifact()

    # Fit 데이터 구성
    if fit_mode == "corpus":
        fit_sentences = corpus_texts
        print(f"\n[방법 A] Fit: corpus {len(fit_sentences):,}개 문서")
    else:  # mixed
        # 중복 제거 후 혼합 (corpus + train queries)
        fit_sentences = corpus_texts + train_qtexts
        print(f"\n[방법 C] Fit: corpus {len(corpus_texts):,}개 + train queries {len(train_qtexts)}개"
              f" = {len(fit_sentences):,}개")

    # 2. Embedder
    print(f"\nLoading model: {model_key}...")
    MODEL_MAP = {
        "nano_gpt": ("gpt2",                         False),
        "qwen-4b":  ("Qwen/Qwen1.5-4B",              True),
        "qwen-8b":  ("Qwen/Qwen2-7B",                True),
        "llama-8b": ("meta-llama/Meta-Llama-3.1-8B", True),
    }
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_key}")
    model_name, use_auto = MODEL_MAP[model_key]
    embedder = HFLLMPoolEmbedder(model_name, cfg.device, use_auto_device_map=use_auto)

    # 3. Embed fit data
    print(f"\nEmbedding fit data ({len(fit_sentences):,} sentences)...")
    X_fit    = embedder.encode(fit_sentences, batch_size=256)
    base_dim = X_fit.shape[1]
    print(f"  shape: {X_fit.shape}")

    # Dims: base_dim//2 → 4 (log2 steps)
    dims = []
    curr = base_dim // 2
    while curr >= 4:
        dims.append(curr); curr //= 2
    print(f"Target dims: {dims}")

    results = []

    # 4. Base
    print("\n[Base]")
    ret = evaluate_retrieval(embedder, corpus_texts, corpus_ids,
                             test_qtexts, test_qids, qrels_test,
                             return_anisotropy=True, return_self_sim=True)
    base_metrics, base_extras = ret
    results.append({"dim": base_dim, "method": "Base",
                    **base_metrics, "anisotropy": base_extras[0], "self_sim": base_extras[1]})
    print(f"  NDCG@10={base_metrics['ndcg@10']:.2f}  "
          f"Recall@100={base_metrics['recall@100']:.2f}  "
          f"MAP={base_metrics['map']:.2f}  "
          f"Anisotropy={base_extras[0]:.4f}")

    # 5. PCA whitening (pre-compute once)
    print("\nComputing PCA whitening matrix...")
    mu_pca, W_pca_full, comps_full, scales_full = compute_pca_whitening_matrix(X_fit, base_dim)

    # 6. Standard methods
    for k in dims:
        print(f"\n[dim={k}]")

        for method_name, model_obj in [
            ("PCA-White",  WrappedLinear(embedder, W=W_pca_full[:, :k], mu_in=mu_pca)),
            ("Soft-White", WrappedSoftWhitening(embedder, mu_pca, comps_full, scales_full, k)),
            ("Random",     WrappedLinear(embedder, W=fit_random_projector(base_dim, k, cfg.seed),
                                         mu_in=X_fit.mean(0))),
        ]:
            m, ex = evaluate_retrieval(model_obj, corpus_texts, corpus_ids,
                                       test_qtexts, test_qids, qrels_test,
                                       return_anisotropy=True, return_self_sim=True)
            results.append({"dim": k, "method": method_name,
                             **m, "anisotropy": ex[0], "self_sim": ex[1]})
            print(f"  {method_name:<12} NDCG@10={m['ndcg@10']:.2f}")

        W_lpp, mu_lpp = fit_lpp_projector(X_fit, k=k)
        m, ex = evaluate_retrieval(WrappedLinear(embedder, W=W_lpp, mu_in=mu_lpp),
                                   corpus_texts, corpus_ids,
                                   test_qtexts, test_qids, qrels_test,
                                   return_anisotropy=True, return_self_sim=True)
        results.append({"dim": k, "method": "LPP",
                         **m, "anisotropy": ex[0], "self_sim": ex[1]})
        print(f"  {'LPP':<12} NDCG@10={m['ndcg@10']:.2f}")

    # 7. NCWP Ablation
    ncwp_configs = [
        {"name": "NCWP_Default",  "k": 10, "cov": 1.0, "orth": 1.0, "cont": 1.0},
        {"name": "NCWP_HighCont", "k": 10, "cov": 1.0, "orth": 1.0, "cont": 5.0},
        {"name": "NCWP_HighCov",  "k": 10, "cov": 5.0, "orth": 1.0, "cont": 1.0},
        {"name": "NCWP_HighOrth", "k": 10, "cov": 1.0, "orth": 5.0, "cont": 1.0},
        {"name": "NCWP_k5",       "k":  5, "cov": 1.0, "orth": 1.0, "cont": 1.0},
        {"name": "NCWP_k20",      "k": 20, "cov": 1.0, "orth": 1.0, "cont": 1.0},
        {"name": "NCWP_k40",      "k": 40, "cov": 1.0, "orth": 1.0, "cont": 1.0},
    ]

    results_main     = list(results)
    results_ablation = [r for r in results if r["method"] == "Base"]

    for conf in ncwp_configs:
        name = conf["name"]
        print(f"\n[NCWP: {name}]")
        for k in dims:
            wdir  = os.path.join(cfg.output_dir, "weights", model_key, name)
            wpath = os.path.join(wdir, f"dim_{k}.npz")

            if os.path.exists(wpath):
                d = np.load(wpath)
            else:
                print(f"  dim={k}: training...", end=" ", flush=True)
                trainer = NewProposalTrainer(
                    rank=k, k_neighbors=conf["k"],
                    lambda_cov=conf["cov"], lambda_orth=conf["orth"],
                    lambda_contrast=conf["cont"], max_epochs=cfg.epochs,
                )
                W, mu_in, mu_out, std_out = trainer.fit(X_fit)
                os.makedirs(wdir, exist_ok=True)
                np.savez(wpath, W=W, mu_in=mu_in, mu_out=mu_out, std_out=std_out)
                print("done")
                d = {"W": W, "mu_in": mu_in, "mu_out": mu_out, "std_out": std_out}

            ncwp_m = WrappedNCWP(embedder, d["W"], d["mu_in"], d["mu_out"], d["std_out"])
            m, ex  = evaluate_retrieval(ncwp_m, corpus_texts, corpus_ids,
                                        test_qtexts, test_qids, qrels_test,
                                        return_anisotropy=True, return_self_sim=True)
            row = {"dim": k, "method": name, **m,
                   "anisotropy": ex[0], "self_sim": ex[1]}
            results_ablation.append(row)
            if name == "NCWP_Default":
                results_main.append({**row, "method": "NCWP"})
            print(f"  dim={k}  NDCG@10={m['ndcg@10']:.2f}  MAP={m['map']:.2f}")

    # 8. Save & Plot
    print("\nSaving results...")
    pd.DataFrame(results_main).to_csv(
        os.path.join(cfg.output_dir, f"{model_key}_results_main.csv"), index=False)
    pdf_path = os.path.join(cfg.output_dir, f"{model_key}_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_results(results_main, f"{model_key}_main", pdf=pdf)

        pd.DataFrame(results_ablation).to_csv(
            os.path.join(cfg.output_dir, f"{model_key}_results_ablation.csv"), index=False)
        plot_results(results_ablation, f"{model_key}_ablation", pdf=pdf)
    print(f"PDF 저장: {pdf_path}")

    print(f"\n{'='*60}")
    print(f" Done!  Results → {cfg.output_dir}/")
    print(f"{'='*60}")
    cols = ["dim", "method", "ndcg@10", "recall@100", "map"]
    print(pd.DataFrame(results_main)[cols].to_string(index=False))


# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SciFact NCWP Experiment")
    parser.add_argument("--model", required=True,
                        choices=["nano_gpt", "qwen-4b", "qwen-8b", "llama-8b"],
                        help="Base LLM model key")
    parser.add_argument("--fit_mode", default="corpus",
                        choices=["corpus", "mixed"],
                        help=(
                            "corpus [방법 A]: corpus 문서만으로 비지도 kNN 학습 | "
                            "mixed  [방법 C]: train query + corpus 혼합 학습"
                        ))
    args = parser.parse_args()
    run_experiment(args.model, fit_mode=args.fit_mode)
