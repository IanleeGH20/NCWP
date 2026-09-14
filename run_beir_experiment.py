"""
run_beir_experiment.py  ── v4 (warmup_steps=200, cosine_tau=0.1, seed42 v3 재실행)

BEIR 데이터셋으로 NCWP 실험 수행.

★ v2 핵심 최적화 ★
  corpus/query 인코딩은 각각 딱 1번만 수행, 이후 projection만 적용.

지원 데이터셋:
  scifact  : corpus 5,183   / test  300  / no dev
  quora    : corpus 522,931 / test 10,000 / dev
  scidocs  : corpus 25,657  / test 1,000  / no dev
  nfcorpus : corpus 3,633   / test  323  / dev
  fiqa     : corpus 57,638  / test  648  / dev(train)

지원 모델:
  LLM     : nano_gpt, qwen-4b, qwen-8b, llama-8b
  Encoder : e5-base (E5-base, mean pool + prefix)
            bge-base (BGE-base-en-v1.5, CLS pool)

BM25 baseline: --no_bm25 플래그로 비활성화 가능 (대용량 corpus에서 느릴 수 있음)

fit_mode:
  corpus_sample  [방법 A] : corpus 랜덤 N개 샘플로 비지도 kNN 학습
  dev_queries    [방법 C] : dev split 쿼리로 학습
  mixed          [방법 A+C]: corpus_sample + dev_queries 혼합

실행 예시:
  python run_beir_experiment.py --dataset quora --model llama-8b --fit_mode corpus_sample
  python run_beir_experiment.py --dataset scidocs --model e5-base --fit_mode corpus_sample
  python run_beir_experiment.py --dataset nfcorpus --model bge-base --fit_mode dev_queries
  python run_beir_experiment.py --dataset fiqa --model e5-base --fit_mode dev_queries --no_bm25
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
for _p in [
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib",
]:
    if os.path.exists(_p):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _p not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_p}:{_ld}"
        break


# ============================================================
# Dataset Configs
# ============================================================
DATASET_CONFIGS = {
    "scifact": {
        "mteb_name":          "mteb/scifact",
        "has_dev":            False,
        "dev_split":          "train",   # labeled train as dev proxy
        "test_split":         "test",
        "default_fit_sample": None,
        "eval_chunk":         1000,
    },
    "quora": {
        "mteb_name":          "mteb/quora",
        "has_dev":            True,
        "dev_split":          "dev",
        "test_split":         "test",
        "default_fit_sample": 20000,
        "eval_chunk":         500,
    },
    "scidocs": {
        "mteb_name":          "mteb/scidocs",
        "has_dev":            False,
        "dev_split":          None,      # SCIDOCS: test only
        "test_split":         "test",
        "default_fit_sample": 5000,      # corpus sample for fitting
        "eval_chunk":         500,
    },
    "nfcorpus": {
        "mteb_name":          "mteb/nfcorpus",
        "has_dev":            True,
        "dev_split":          "dev",
        "test_split":         "test",
        "default_fit_sample": None,      # small corpus (~3.6K), use all
        "eval_chunk":         200,
    },
    "fiqa": {
        "mteb_name":          "mteb/fiqa",
        "has_dev":            True,
        "dev_split":          "dev",
        "test_split":         "test",
        "default_fit_sample": 10000,
        "eval_chunk":         500,
    },
}

# query_sample fit_mode 에서 사용하는 N 단계 (sample efficiency 실험)
QUERY_SAMPLE_NS = [3000, 5000, 10000]


# ============================================================
# Configuration
# ============================================================
class BEIRConfig:
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

    hf_cache_dir = "/workspace/RAG/code/Make_embedding/nanoGPT"
    results_root = "/workspace/NCWP"

cfg = BEIRConfig()
os.makedirs(cfg.results_root, exist_ok=True)
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
# NCWP Core Helpers  (완전한 버전 — ncwp_eval/code/projector.py 기준)
# ============================================================
try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    _TQDM_AVAILABLE = False

def _orth_penalty(W):
    WT_W = W.T @ W
    I = torch.eye(WT_W.shape[0], device=W.device, dtype=W.dtype)
    return torch.mean((WT_W - I) ** 2)

def _cov_penalty(Z):
    B = Z.shape[0]
    if B <= 1: return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / (B - 1)
    I = torch.eye(Cov.shape[0], device=Z.device, dtype=Z.dtype)
    return torch.mean((Cov - I) ** 2)

def _zca_shrink(X, shrink=0.10, eps=1e-6):
    mu = X.mean(0, keepdim=True); Xc = X - mu
    N = X.shape[0]
    Cov = (Xc.T @ Xc) / max(N - 1, 1)
    D = Cov.shape[0]; trace = torch.trace(Cov)
    mean_eig = trace / D
    Cov_s = (1.0 - shrink) * Cov + shrink * mean_eig * torch.eye(D, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cov_s)
    ev = torch.clamp(ev, min=eps)
    S = evec @ torch.diag(1.0 / torch.sqrt(ev)) @ evec.T
    return mu.squeeze(0), S

@torch.no_grad()
def _topk_cosine_knn(X, k, chunk=2048):
    N = X.shape[0]; device = X.device
    idx_all = torch.empty((N, k), dtype=torch.long, device="cpu")
    X_T = X.T
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        sims = X[s:e] @ X_T
        sims[:, torch.arange(s, e, device=device)] = -1e9
        idx_all[s:e] = torch.topk(sims, k=k, dim=1).indices.detach().cpu()
    return idx_all

@torch.no_grad()
def _topk_knn_with_sims(X, k, chunk=2048):
    """kNN 인덱스와 코사인 유사도를 함께 반환 (cosine_tau 필터용)."""
    N = X.shape[0]; device = X.device
    idx_all = torch.empty((N, k), dtype=torch.long,  device="cpu")
    sim_all = torch.empty((N, k), dtype=torch.float, device="cpu")
    X_T = X.T
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        sims = X[s:e] @ X_T
        sims[:, torch.arange(s, e, device=device)] = -1e9
        topk = torch.topk(sims, k=k, dim=1)
        idx_all[s:e] = topk.indices.detach().cpu()
        sim_all[s:e] = topk.values.detach().cpu()
    return idx_all, sim_all

def fit_lpp_projector(X, k, n_neighbors=10, reg=1e-3):
    X = np.asarray(X, dtype=np.float32); N, D = X.shape
    mu = X.mean(0, keepdims=True); Xc = X - mu
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=min(n_neighbors, N-1)-1, axis=1)[:, :n_neighbors]
    Wg = np.zeros((N, N), dtype=np.float32)
    Wg[np.repeat(np.arange(N), idx.shape[1]), idx.reshape(-1)] = 1.0
    Wg = np.maximum(Wg, Wg.T)
    d = Wg.sum(1); Dg = np.diag(d); Lg = Dg - Wg
    XtDX = Xc.T @ Dg @ Xc + reg * np.eye(D, dtype=np.float32)
    XtLX = Xc.T @ Lg @ Xc
    try:    A = np.linalg.solve(XtDX, XtLX)
    except: A = np.linalg.pinv(XtDX) @ XtLX
    w, V = np.linalg.eigh(A)
    return V[:, np.argsort(w)][:, :min(k, D)].astype(np.float32), mu.squeeze(0).astype(np.float32)

def fit_random_projector(dim_in, k, seed=42):
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(dim_in, k)).astype(np.float32))
    return Q[:, :k].astype(np.float32)

def compute_pca_whitening_matrix(X_np, target_dim, shrink=0.08, eps=1e-6):
    if np.isnan(X_np).any(): X_np = np.nan_to_num(X_np)
    X = torch.from_numpy(X_np).to(cfg.device).double()
    N, D = X.shape; mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(N - 1, 1)
    trace = torch.trace(Cov)
    Cov_s = (1-shrink)*Cov + shrink*(trace/D)*torch.eye(D, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cov_s)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(target_dim, D)
    W = evec[:, :k] @ torch.diag(1.0 / torch.sqrt(torch.clamp(ev[:k], min=eps)))
    return (mu.squeeze(0).cpu().numpy().astype(np.float32),
            W.cpu().numpy().astype(np.float32),
            evec.cpu().numpy().astype(np.float32),
            ev.cpu().numpy().astype(np.float32))


# ============================================================
# NCWP Trainer  ── 완전한 버전 (ncwp_eval/code/projector.py 기준)
#   Symmetric InfoNCE + Top-K hard negatives + QR retraction
#   Cosine LR/temp schedule + kNN refinement + Early stopping
# ============================================================
class NewProposalTrainer:
    """
    완전한 NCWP 학습기:
      - ZCA+shrink whitening
      - kNN positive mining + Symmetric InfoNCE
      - Top-K hard negative mining
      - Covariance / Orthogonality regularization
      - Cosine LR & temperature schedule (warmup 포함)
      - Periodic QR retraction (Stiefel manifold)
      - kNN refinement round (projected 공간에서 kNN 재구성)
      - Early stopping + Best W tracking
    """
    def __init__(
        self,
        rank,
        temperature=0.12,
        temp_min=None,
        use_cosine_temp=True,
        k_neighbors=40,
        shrink=0.08,
        max_epochs=20,
        lr=8e-3,
        lr_min=None,
        warmup_steps=200,
        use_cosine_lr=True,
        batch_pairs=None,
        steps_per_epoch=None,
        lambda_cov=0.05,
        lambda_orth=0.02,
        cosine_tau=0.1,
        topk_negatives=256,
        retraction_interval=10,
        refine_knn_rounds=1,
        early_stop_patience=6,
        weight_decay=1e-4,
        seed=42,
        device=None,
        verbose=False,
        use_tqdm=False,
    ):
        self.rank            = int(rank)
        self.temperature     = float(temperature)
        self.temp_min        = float(temp_min) if temp_min is not None else None
        self.use_cosine_temp = bool(use_cosine_temp)
        self.k_neighbors     = int(k_neighbors)
        self.shrink          = float(shrink)
        self.max_epochs      = int(max_epochs)
        self.lr              = float(lr)
        self.lr_min          = float(lr_min) if lr_min is not None else None
        self.warmup_steps    = int(warmup_steps)
        self.use_cosine_lr   = bool(use_cosine_lr)
        self.batch_pairs     = batch_pairs
        self.steps_per_epoch = steps_per_epoch
        self.lambda_cov      = float(lambda_cov)
        self.lambda_orth     = float(lambda_orth)
        self.cosine_tau      = float(cosine_tau)
        self.topk_negatives  = int(topk_negatives) if topk_negatives is not None else None
        self.retraction_interval  = int(retraction_interval)
        self.refine_knn_rounds    = int(refine_knn_rounds)
        self.early_stop_patience  = int(early_stop_patience)
        self.weight_decay    = float(weight_decay)
        self.verbose         = bool(verbose)
        self.use_tqdm        = bool(use_tqdm) and _TQDM_AVAILABLE
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import random
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def _infer_batch_pairs(self, N):
        if self.batch_pairs is not None:
            return int(self.batch_pairs)
        return int(max(64, min(1024, max(64, N // 64))))

    def _infer_steps(self, N):
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return int(min(160, max(100, math.ceil(N / max(1, 2 * self._infer_batch_pairs(N))))))

    def _contrastive_loss(self, logits_row, pos_index, topk=None):
        B = logits_row.shape[0]
        pos_logits = logits_row.gather(1, pos_index.view(-1, 1)).squeeze(1)
        if (topk is None) or (topk >= logits_row.shape[1] - 1):
            denom = torch.logsumexp(logits_row, dim=1)
        else:
            mask = torch.ones_like(logits_row, dtype=torch.bool)
            mask[torch.arange(B, device=logits_row.device), pos_index] = False
            negs = logits_row.masked_select(mask).view(B, -1)
            kk = min(topk, negs.shape[1])
            vals, _ = torch.topk(negs, k=kk, dim=1)
            denom = torch.logsumexp(
                torch.cat([vals, pos_logits.unsqueeze(1)], dim=1), dim=1)
        return -(pos_logits - denom).mean()

    def fit(self, X_np):
        N, D = X_np.shape
        r = min(self.rank, D)
        X = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(self.device)

        # 1) ZCA+shrink whitening
        mu_in_t, S = _zca_shrink(X, shrink=self.shrink)
        Xw = F.normalize((X - mu_in_t) @ S, dim=1)

        # 2) Initial kNN (유사도 포함 — cosine_tau 필터용)
        k = max(1, min(self.k_neighbors, max(1, N - 1)))
        with torch.no_grad():
            knn_idx, knn_sim = _topk_knn_with_sims(Xw, k=k)

        # 3) W 초기화
        W = torch.nn.Parameter(
            torch.randn(D, r, device=self.device) * (1.0 / math.sqrt(D)))
        opt = torch.optim.AdamW([W], lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self._infer_steps(N)
        lr_min   = self.lr_min   if self.lr_min   is not None else 0.2 * self.lr
        temp_min = self.temp_min if self.temp_min is not None else 0.5 * self.temperature
        global_step = 0
        best_W   = None
        best_loss = float("inf")
        no_imp   = 0

        def _one_round(knn_idx_local, knn_sim_local, tag="init"):
            nonlocal global_step, best_W, best_loss, no_imp
            total_steps_round = self._infer_steps(N)
            for epoch in range(1, self.max_epochs + 1):
                epoch_loss = 0.0
                for _ in range(total_steps_round):
                    # LR schedule
                    if self.use_cosine_lr:
                        T_total = self.max_epochs * total_steps_round
                        t = max(0.0, global_step - self.warmup_steps) / max(1, T_total - self.warmup_steps)
                        if global_step < self.warmup_steps:
                            cur_lr = lr_min + (self.lr - lr_min) * (global_step / max(1, self.warmup_steps))
                        else:
                            cur_lr = lr_min + 0.5 * (self.lr - lr_min) * (1 + math.cos(math.pi * t))
                        for pg in opt.param_groups:
                            pg["lr"] = cur_lr
                    # Temperature schedule
                    cur_temp = self.temperature
                    if self.use_cosine_temp:
                        t2 = global_step / max(1, self.max_epochs * total_steps_round - 1)
                        cur_temp = temp_min + 0.5 * (self.temperature - temp_min) * (1 + math.cos(math.pi * t2))

                    # Positive pair sampling (cosine_tau 필터 적용)
                    with torch.no_grad():
                        Bp = self._infer_batch_pairs(N)
                        a_cpu = torch.randint(0, N, (Bp,), dtype=torch.long)
                        b_list = []
                        tau_t = torch.tensor(self.cosine_tau, dtype=torch.float)
                        for anc in a_cpu.tolist():
                            valid_mask = knn_sim_local[anc] >= tau_t
                            valid_nb   = knn_idx_local[anc][valid_mask]
                            if len(valid_nb) == 0:
                                valid_nb = knn_idx_local[anc]   # fallback: τ 필터 무시
                            b_list.append(valid_nb[torch.randint(0, len(valid_nb), (1,))].item())
                        b_cpu = torch.tensor(b_list, dtype=torch.long)
                        a_idx = a_cpu.to(self.device, non_blocking=True)
                        b_idx = b_cpu.to(self.device, non_blocking=True)
                        Xbatch = torch.cat([Xw[a_idx], Xw[b_idx]], dim=0)  # (2B, D)

                    # Forward
                    Z = F.normalize(Xbatch @ W, dim=1)
                    sim = (Z @ Z.T) / cur_temp
                    B2 = Z.shape[0]; B = B2 // 2
                    eye_mask = torch.eye(B2, device=self.device, dtype=torch.bool)
                    sim = sim.masked_fill(eye_mask, -1e9)
                    idx1 = torch.arange(B, device=self.device)
                    idx2 = idx1 + B

                    # Symmetric InfoNCE + regularization
                    loss_i = self._contrastive_loss(sim[idx1], idx2, topk=self.topk_negatives)
                    loss_j = self._contrastive_loss(sim[idx2], idx1, topk=self.topk_negatives)
                    loss = (0.5 * (loss_i + loss_j)
                            + self.lambda_cov  * _cov_penalty(Z)
                            + self.lambda_orth * _orth_penalty(W))

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    # QR retraction (Stiefel manifold)
                    if self.retraction_interval > 0 and (global_step + 1) % self.retraction_interval == 0:
                        with torch.no_grad():
                            Q, _ = torch.linalg.qr(W.data)
                            W.data = Q[:, :r]

                    epoch_loss += float(loss.item())
                    global_step += 1

                epoch_loss /= max(1, total_steps_round)
                if epoch_loss + 1e-6 < best_loss:
                    best_loss = epoch_loss; no_imp = 0
                    best_W = W.detach().clone()
                else:
                    no_imp += 1
                    if no_imp >= self.early_stop_patience:
                        break

        # 초기 학습
        _one_round(knn_idx, knn_sim, "init")

        # kNN refinement: projected 공간에서 kNN 재구성 후 재학습
        for rr in range(self.refine_knn_rounds):
            with torch.no_grad():
                Zfull = F.normalize(Xw @ best_W, dim=1)
                knn_idx, knn_sim = _topk_knn_with_sims(Zfull, k=k)
                W.data.copy_(best_W)
            _one_round(knn_idx, knn_sim, f"ref{rr+1}")

        with torch.no_grad():
            SW = S @ best_W
            Y  = (X - mu_in_t) @ SW
        return (SW.detach().cpu().numpy(),
                mu_in_t.detach().cpu().numpy(),
                Y.mean(0).detach().cpu().numpy(),
                torch.clamp(Y.std(0), min=1e-6).detach().cpu().numpy())


# ============================================================
# Model Configs  (LLM + Encoder 통합)
# ============================================================
MODEL_CONFIGS = {
    "nano_gpt": {
        "hf_name":       "gpt2",
        "use_auto":      False,
        "embedder_type": "llm",
    },
    "qwen-4b": {
        "hf_name":       "Qwen/Qwen1.5-4B",
        "use_auto":      True,
        "embedder_type": "llm",
    },
    "qwen-8b": {
        "hf_name":       "Qwen/Qwen2-7B",
        "use_auto":      True,
        "embedder_type": "llm",
    },
    "llama-8b": {
        "hf_name":       "meta-llama/Meta-Llama-3.1-8B",
        "use_auto":      True,
        "embedder_type": "llm",
    },
    "e5-base": {
        "hf_name":       "intfloat/e5-base",
        "use_auto":      False,
        "embedder_type": "biencoder",
        "pool_mode":     "mean",
        "query_prefix":  "query: ",
        "doc_prefix":    "passage: ",
    },
    "bge-base": {
        "hf_name":       "BAAI/bge-base-en-v1.5",
        "use_auto":      False,
        "embedder_type": "biencoder",
        "pool_mode":     "cls",
        "query_prefix":  "",
        "doc_prefix":    "",
    },
}


# ============================================================
# LLM Embedder  (mean pool, last hidden state)
# ============================================================
class HFLLMPoolEmbedder:
    def __init__(self, model_name, device="cuda", use_auto_device_map=True):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        kw = {"trust_remote_code": True,
              "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
              "attn_implementation": "eager"}
        n = torch.cuda.device_count(); print(f"[Info] {n} GPUs.")
        if n > 1:
            self.model = torch.nn.DataParallel(
                AutoModel.from_pretrained(model_name, **kw)).cuda()
            self.is_parallel = True
        else:
            if use_auto_device_map: kw["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(model_name, **kw)
            if not use_auto_device_map: self.model.to(device)
            self.is_parallel = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences, batch_size=256, **kwargs):
        embs = []
        dev = self.device if self.is_parallel else getattr(self.model, "device", self.device)
        for i in range(0, len(sentences), batch_size):
            enc = self.tok(sentences[i:i+batch_size], padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(dev)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            x = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embs.append(F.normalize(x, p=2, dim=1).cpu().float().numpy())
        return np.concatenate(embs) if embs else np.array([])


# ============================================================
# Encoder Embedder  (BERT-style, mean/CLS pool + prefix)
# ============================================================
class HFBiEncoderEmbedder:
    """E5 / BGE 등 인코더 모델용. mean pooling 또는 CLS 토큰 지원."""

    def __init__(self, model_name, pool_mode="mean",
                 query_prefix="", doc_prefix="", device="cuda"):
        self.device      = device
        self.pool_mode   = pool_mode
        self.query_prefix = query_prefix
        self.doc_prefix   = doc_prefix

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        print(f"[BiEncoder] {model_name}  pool={pool_mode}  device={device}")

    @torch.no_grad()
    def encode(self, sentences, batch_size=256, prefix="", **kwargs):
        if prefix:
            sentences = [prefix + s for s in sentences]
        embs = []
        for i in range(0, len(sentences), batch_size):
            enc = self.tok(sentences[i:i+batch_size], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            if self.pool_mode == "cls":
                x = h[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                x = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embs.append(F.normalize(x, p=2, dim=1).cpu().float().numpy())
        return np.concatenate(embs) if embs else np.array([])


# ============================================================
# BM25 Baseline
# ============================================================
BM25_CORPUS_LIMIT = 100_000   # 이 이상이면 BM25 자동 스킵 (순수 Python 한계)


def evaluate_bm25(corpus_texts, corpus_ids, query_texts, query_ids, qrels,
                  k_values=(10, 100)):
    """BM25Okapi로 리트리벌 평가. rank_bm25 패키지 사용.
    corpus > BM25_CORPUS_LIMIT 이면 자동 스킵 (너무 느림).
    """
    if len(corpus_texts) > BM25_CORPUS_LIMIT:
        print(f"  [BM25] corpus {len(corpus_texts):,} > {BM25_CORPUS_LIMIT:,} → 자동 스킵 "
              f"(순수 Python 한계). --no_bm25 불필요.")
        return None

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  [BM25] rank_bm25 미설치 — pip install rank-bm25")
        return None

    print(f"  BM25: corpus {len(corpus_texts):,}개 인덱싱...", flush=True)
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    ndcg_at   = {k: [] for k in k_values}
    recall_at = {k: [] for k in k_values}
    ap_list   = []
    max_k = max(k_values)

    print(f"  BM25: {len(query_ids):,}개 쿼리 평가...", flush=True)
    for qid, qtext in zip(query_ids, query_texts):
        if qid not in qrels:
            continue
        tok_q  = qtext.lower().split()
        scores = bm25.get_scores(tok_q)
        top_idx = np.argsort(-scores)[:max_k]
        ranked  = [corpus_ids[i] for i in top_idx]
        rel     = qrels[qid]
        ideal   = sorted(rel.values(), reverse=True)

        for k in k_values:
            top = ranked[:k]
            dcg  = sum((2**rel.get(d, 0)-1)/math.log2(r+2) for r, d in enumerate(top))
            idcg = sum((2**rv-1)/math.log2(r+2) for r, rv in enumerate(ideal[:k]))
            ndcg_at[k].append(dcg/idcg if idcg > 0 else 0.0)
            relevant = {d for d, s in rel.items() if s > 0}
            recall_at[k].append(len(set(top) & relevant)/len(relevant) if relevant else 0.0)

        total_rel = sum(1 for s in rel.values() if s > 0)
        nr = 0; ap = 0.0
        for r, d in enumerate(ranked, 1):
            if rel.get(d, 0) > 0:
                nr += 1; ap += nr / r
        ap_list.append(ap / total_rel if total_rel > 0 else 0.0)

    metrics = {f"ndcg@{k}": np.mean(ndcg_at[k]) * 100 for k in k_values}
    metrics.update({f"recall@{k}": np.mean(recall_at[k]) * 100 for k in k_values})
    metrics["map"] = np.mean(ap_list) * 100
    return metrics


# ============================================================
# Projection helpers  ── base embedding → projected embedding
# (모두 numpy 연산, LLM 재호출 없음)
# ============================================================
def _l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _embedding_cache_path(dataset_name, model_key, cache_name):
    cache_dir = os.path.join(
        cfg.results_root,
        f"{dataset_name}_results",
        "embedding_cache",
        model_key,
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_name}.npy")


def _load_or_compute_embeddings(cache_path, step_message, done_label,
                                sentences, embedder, batch_size=256):
    if os.path.exists(cache_path):
        print(f"\n[캐시] {step_message} 로드: {cache_path}", flush=True)
        X = np.load(cache_path).astype(np.float32, copy=False)
    else:
        print(f"\n[인코딩] {step_message}...", flush=True)
        X = embedder.encode(sentences, batch_size=batch_size).astype(np.float32)
        X = _l2norm(X)
        np.save(cache_path, X)
        print(f"  캐시 저장: {cache_path}")
    print(f"  {done_label} 완료: {X.shape}")
    return X

def proj_linear(X_base, W, mu):
    """PCA-White / Random / LPP 공통"""
    return _l2norm((X_base - mu) @ W)

def proj_soft_white(X_base, mu, comps, scales, k, alpha=0.1):
    Z = (X_base - mu) @ comps[:, :k]
    sm = scales[:k].mean() if scales[:k].size > 0 else 1.0
    Z = Z / np.sqrt((1 - alpha) * scales[:k] + alpha * sm + 1e-8)
    return _l2norm(Z)

def proj_ncwp(X_base, W, mu_in, mu_out, std_out):
    Z = ((X_base - mu_in) @ W - mu_out) / std_out
    return _l2norm(Z)


# ============================================================
# Data Loading
# ============================================================
def load_beir_data(dataset_name):
    dcfg = DATASET_CONFIGS[dataset_name]
    mname = dcfg["mteb_name"]

    print(f"Loading {dataset_name} corpus...")
    c_ds = load_dataset(mname, "corpus", split="corpus")
    corpus_ids   = [r["_id"] for r in c_ds]
    corpus_texts = [(r["title"] + " " + r["text"]).strip() for r in c_ds]

    print(f"Loading {dataset_name} queries...")
    q_ds   = load_dataset(mname, "queries", split="queries")
    qid2tx = {r["_id"]: r["text"] for r in q_ds}

    print(f"Loading {dataset_name} qrels...")
    qr_ds = load_dataset(mname)

    def _build(split):
        d = defaultdict(dict)
        for r in qr_ds[split]:
            d[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
        return dict(d)

    qrels_test = _build(dcfg["test_split"])
    if dcfg["dev_split"] is not None and dcfg["dev_split"] in qr_ds:
        qrels_dev = _build(dcfg["dev_split"])
    else:
        qrels_dev = {}

    test_qids   = [q for q in sorted(qrels_test) if q in qid2tx]
    test_qtexts = [qid2tx[q] for q in test_qids]

    dev_qids    = [q for q in sorted(qrels_dev) if q in qid2tx]
    dev_qtexts  = [qid2tx[q] for q in dev_qids]

    print(f"  corpus   : {len(corpus_ids):,}")
    print(f"  test     : {len(test_qids):,} queries / {sum(len(v) for v in qrels_test.values()):,} pairs")
    print(f"  dev(fit) : {len(dev_qtexts):,} queries")
    return corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test, dev_qtexts


def build_eval_subset(corpus_ids, corpus_texts,
                      test_qids, test_qtexts, qrels_test,
                      query_limit=None, corpus_limit=None, seed=42):
    """빠른 실험용 평가 subset을 만든다. 정답 문서는 항상 포함한다."""
    if query_limit is None and corpus_limit is None:
        return corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test

    rng = np.random.RandomState(seed)

    if query_limit is not None and query_limit < len(test_qids):
        q_idx = np.sort(rng.choice(len(test_qids), query_limit, replace=False))
        eval_qids = [test_qids[i] for i in q_idx]
        eval_qtexts = [test_qtexts[i] for i in q_idx]
    else:
        eval_qids = list(test_qids)
        eval_qtexts = list(test_qtexts)

    eval_qrels = {qid: dict(qrels_test[qid]) for qid in eval_qids if qid in qrels_test}

    if corpus_limit is None or corpus_limit >= len(corpus_ids):
        eval_corpus_ids = list(corpus_ids)
        eval_corpus_texts = list(corpus_texts)
        print(f"\n[Fast eval] query {len(eval_qids):,}개 / corpus 전체 {len(eval_corpus_ids):,}개 사용")
        return eval_corpus_ids, eval_corpus_texts, eval_qids, eval_qtexts, eval_qrels

    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    positive_ids = {
        cid
        for rel in eval_qrels.values()
        for cid, score in rel.items()
        if score > 0 and cid in corpus_id_to_idx
    }

    if len(positive_ids) >= corpus_limit:
        selected_ids = positive_ids
        print(f"\n[Fast eval] 경고: 정답 문서만 {len(positive_ids):,}개라 corpus_limit={corpus_limit:,}를 초과합니다.")
    else:
        remaining_needed = corpus_limit - len(positive_ids)
        remaining_ids = [cid for cid in corpus_ids if cid not in positive_ids]
        if remaining_needed >= len(remaining_ids):
            sampled_negative_ids = remaining_ids
        else:
            neg_idx = rng.choice(len(remaining_ids), remaining_needed, replace=False)
            sampled_negative_ids = [remaining_ids[i] for i in neg_idx]
        selected_ids = positive_ids | set(sampled_negative_ids)

    selected_indices = [i for i, cid in enumerate(corpus_ids) if cid in selected_ids]
    eval_corpus_ids = [corpus_ids[i] for i in selected_indices]
    eval_corpus_texts = [corpus_texts[i] for i in selected_indices]
    eval_qrels = {
        qid: {cid: score for cid, score in rel.items() if cid in selected_ids}
        for qid, rel in eval_qrels.items()
    }

    print(
        f"\n[Fast eval] query {len(eval_qids):,}개 / corpus {len(eval_corpus_ids):,}개 "
        f"(정답 문서 {len(positive_ids):,}개 포함)"
    )
    return eval_corpus_ids, eval_corpus_texts, eval_qids, eval_qtexts, eval_qrels


# ============================================================
# Retrieval Evaluation  ── pre-computed embeddings version
# ============================================================
def evaluate_retrieval_precomputed(corpus_embs, corpus_ids,
                                   query_embs, query_ids, qrels,
                                   k_values=(10, 100), eval_chunk=500,
                                   return_anisotropy=False, return_self_sim=False):
    """
    corpus_embs, query_embs : 이미 projected + l2-normalized (N, dim) float32 numpy
    LLM 재인코딩 없이 행렬곱 평가만 수행.
    """
    max_k = max(k_values)
    ndcg_at   = {k: [] for k in k_values}
    recall_at = {k: [] for k in k_values}
    ap_list   = []

    print(f"    Evaluating (chunk={eval_chunk}, corpus={len(corpus_ids):,})...", flush=True)
    for qi_start in range(0, len(query_ids), eval_chunk):
        qi_end  = min(len(query_ids), qi_start + eval_chunk)
        q_chunk = query_embs[qi_start:qi_end]
        sim_mat = q_chunk @ corpus_embs.T

        if sim_mat.shape[1] <= max_k:
            top_idx = np.argsort(-sim_mat, axis=1)
        else:
            part  = np.argpartition(-sim_mat, kth=max_k, axis=1)[:, :max_k]
            order = np.argsort(-sim_mat[np.arange(len(part))[:, None], part], axis=1)
            top_idx = part[np.arange(len(part))[:, None], order]

        for ci, qi in enumerate(range(qi_start, qi_end)):
            qid = query_ids[qi]
            if qid not in qrels: continue
            rel = qrels[qid]
            ranked = [corpus_ids[i] for i in top_idx[ci]]
            ideal  = sorted(rel.values(), reverse=True)

            for k in k_values:
                top = ranked[:k]
                dcg  = sum((2**rel.get(d, 0)-1)/math.log2(r+2) for r, d in enumerate(top))
                idcg = sum((2**rv-1)/math.log2(r+2) for r, rv in enumerate(ideal[:k]))
                ndcg_at[k].append(dcg/idcg if idcg > 0 else 0.0)
                relevant = {d for d, s in rel.items() if s > 0}
                recall_at[k].append(len(set(top) & relevant)/len(relevant) if relevant else 0.0)

            total_rel = sum(1 for s in rel.values() if s > 0)
            nr = 0; ap = 0.0
            for r, d in enumerate(ranked, 1):
                if rel.get(d, 0) > 0: nr += 1; ap += nr / r
            ap_list.append(ap / total_rel if total_rel > 0 else 0.0)

    metrics = {f"ndcg@{k}": np.mean(ndcg_at[k])*100 for k in k_values}
    metrics.update({f"recall@{k}": np.mean(recall_at[k])*100 for k in k_values})
    metrics["map"] = np.mean(ap_list)*100

    extras = []
    if return_anisotropy:
        mu = corpus_embs.mean(0, keepdims=True)
        mu_hat = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)
        extras.append(float((corpus_embs @ mu_hat.T).mean()))
    if return_self_sim:
        N = corpus_embs.shape[0]
        if N < 10000:
            G = corpus_embs @ corpus_embs.T
            extras.append(float((G.sum() - np.trace(G)) / (N * (N-1))))
        else:
            idx1 = np.random.randint(0, N, 200000); idx2 = np.random.randint(0, N, 200000)
            mask = idx1 != idx2
            extras.append(float((corpus_embs[idx1[mask]] * corpus_embs[idx2[mask]]).sum(1).mean()))

    return (metrics, extras) if extras else metrics


# ============================================================
# Plotting
# ============================================================
def plot_results(results_list, filename_suffix, out_dir, primary="ndcg@10", pdf=None):
    sns.set_theme(style="whitegrid")
    MARK = {"Base":"o","PCA-White":"s","Soft-White":"D","LPP":"x","Random":".",
            "NCWP":"^","NCWP_Default":"^","NCWP_HighCont":"v","NCWP_HighCov":"<",
            "NCWP_HighOrth":">","NCWP_k5":"1","NCWP_k20":"p","NCWP_k40":"*"}
    methods = defaultdict(lambda: {"dims":[],"scores":[],"ani":[]})
    for r in results_list:
        m = r["method"]
        methods[m]["dims"].append(r["dim"])
        methods[m]["scores"].append(r[primary])
        if "anisotropy" in r: methods[m]["ani"].append(r["anisotropy"])

    fig1 = plt.figure(figsize=(12,7))
    for m, d in methods.items():
        zp = sorted(zip(d["dims"], d["scores"]))
        dims, scores = zip(*zp)
        if m == "Base":
            plt.axhline(y=scores[0], ls="--", color="black", alpha=0.7, label=f"Base ({scores[0]:.2f})")
        else:
            plt.plot(dims, scores, marker=MARK.get(m,"o"), label=m, lw=1.5)
    plt.xscale("log", base=2)
    plt.title(f"{filename_suffix} — {primary.upper()} vs Dimension")
    plt.xlabel("Dimension (log scale)"); plt.ylabel(primary.upper())
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.2); plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig1, bbox_inches="tight")
    else:
        plt.savefig(os.path.join(out_dir, f"{filename_suffix}.pdf"), bbox_inches="tight")
    plt.close()

    if any(len(d["ani"]) > 0 for d in methods.values()):
        fig2 = plt.figure(figsize=(12,7))
        for m, d in methods.items():
            if not d["ani"]: continue
            zp = sorted(zip(d["dims"], d["ani"]))
            dims, anis = zip(*zp)
            if m == "Base":
                plt.axhline(y=anis[0], ls="--", color="black", alpha=0.7, label=f"Base ({anis[0]:.4f})")
            else:
                plt.plot(dims, anis, marker=MARK.get(m,"o"), label=m, lw=1.5)
        plt.xscale("log", base=2)
        plt.title(f"{filename_suffix} — Anisotropy vs Dimension")
        plt.xlabel("Dimension (log scale)"); plt.ylabel("Anisotropy")
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        plt.grid(True, which="both", ls="-", alpha=0.2); plt.tight_layout()
        if pdf is not None:
            pdf.savefig(fig2, bbox_inches="tight")
        else:
            plt.savefig(os.path.join(out_dir, f"{filename_suffix}_anisotropy.pdf"), bbox_inches="tight")
        plt.close()
    print(f"  Plots → {out_dir}/{filename_suffix}*.pdf")


# ============================================================
# Main Experiment  ── embed-once 최적화 버전
# ============================================================
def run_experiment(dataset_name, model_key, fit_mode, fit_sample=None,
                   eval_query_limit=None, eval_corpus_limit=None,
                   run_bm25=True, ncwp_only=False, seed=42):
    """
    fit_mode:
      'corpus_sample' : corpus 랜덤 fit_sample 개 (방법 A)
      'dev_queries'   : dev split 쿼리 (방법 C)
      'mixed'         : corpus_sample + dev_queries (방법 A+C)
      'query_sample'  : test queries 중 fit_sample 개 샘플 (방법 D, sample efficiency)
                        fit_sample=None 이면 QUERY_SAMPLE_NS 전체를 순서대로 실험
    """
    dcfg = DATASET_CONFIGS[dataset_name]
    valid_modes = ("corpus_sample", "dev_queries", "mixed", "query_sample")
    assert fit_mode in valid_modes, f"fit_mode must be one of {valid_modes}"

    # query_sample 모드: fit_sample=None이면 QUERY_SAMPLE_NS 전체 루프
    if fit_mode == "query_sample" and fit_sample is None:
        print(f"\n[query_sample] fit_sample 미지정 → N={QUERY_SAMPLE_NS} 전체 순서 실행")
        for n in QUERY_SAMPLE_NS:
            run_experiment(
                dataset_name, model_key, fit_mode, fit_sample=n,
                eval_query_limit=eval_query_limit,
                eval_corpus_limit=eval_corpus_limit,
                run_bm25=run_bm25,
            )
        return

    if fit_sample is None:
        fit_sample = dcfg["default_fit_sample"]

    eval_suffix = ""
    if eval_query_limit is not None:
        eval_suffix += f"_evalQ{eval_query_limit}"
    if eval_corpus_limit is not None:
        eval_suffix += f"_evalC{eval_corpus_limit}"

    # query_sample 모드는 N별 하위 폴더 없이 같은 폴더에 N을 파일명에 포함
    out_dir = os.path.join(cfg.results_root, f"{dataset_name}_results", f"fit_{fit_mode}")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(cfg.results_root, f"{dataset_name}_results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    # query_sample 모드에서는 tag에 _N{n}을 붙여 N별로 구분
    _qs_n_suffix = f"_N{fit_sample}" if fit_mode == "query_sample" else ""

    print(f"\n{'='*65}")
    print(f"  Dataset: {dataset_name}  |  Model: {model_key}  |  fit_mode: {fit_mode}")
    if fit_sample: print(f"  fit_sample: {fit_sample:,}")
    if eval_query_limit is not None: print(f"  eval_query_limit: {eval_query_limit:,}")
    if eval_corpus_limit is not None: print(f"  eval_corpus_limit: {eval_corpus_limit:,}")
    print(f"  Results → {out_dir}")
    print(f"{'='*65}")

    # ── 1. 데이터 로드 ──────────────────────────────────────────
    (corpus_ids, corpus_texts,
     test_qids, test_qtexts,
     qrels_test, dev_qtexts) = load_beir_data(dataset_name)
    eval_chunk = dcfg["eval_chunk"]

    # ── 2. fit 문장 구성 ────────────────────────────────────────
    if fit_mode == "corpus_sample":
        if fit_sample and fit_sample < len(corpus_texts):
            rng = np.random.RandomState(cfg.seed)
            idx = rng.choice(len(corpus_texts), fit_sample, replace=False)
            fit_sentences = [corpus_texts[i] for i in idx]
        else:
            fit_sentences = corpus_texts
        print(f"\n[방법 A] Fit: corpus 샘플 {len(fit_sentences):,}개")

    elif fit_mode == "dev_queries":
        fit_sentences = dev_qtexts
        print(f"\n[방법 C] Fit: dev queries {len(fit_sentences):,}개")

    elif fit_mode == "query_sample":
        # test queries를 pool로 삼아 fit_sample개 샘플
        # pool이 부족하면 dev_queries도 합산하여 보충
        query_pool = test_qtexts + dev_qtexts
        n = min(fit_sample, len(query_pool))
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(len(query_pool), n, replace=False)
        fit_sentences = [query_pool[i] for i in sorted(idx)]
        print(f"\n[방법 D] Fit: query 샘플 {len(fit_sentences):,}개"
              f"  (pool={len(query_pool):,}: test {len(test_qtexts):,} + dev {len(dev_qtexts):,})")

    else:  # mixed
        if fit_sample and fit_sample < len(corpus_texts):
            rng = np.random.RandomState(cfg.seed)
            idx = rng.choice(len(corpus_texts), fit_sample, replace=False)
            corpus_sample = [corpus_texts[i] for i in idx]
        else:
            corpus_sample = corpus_texts
        fit_sentences = corpus_sample + dev_qtexts
        print(f"\n[방법 A+C] Fit: corpus {len(corpus_sample):,} + dev {len(dev_qtexts):,} = {len(fit_sentences):,}개")

    (eval_corpus_ids, eval_corpus_texts,
     eval_test_qids, eval_test_qtexts,
     eval_qrels_test) = build_eval_subset(
        corpus_ids, corpus_texts,
        test_qids, test_qtexts, qrels_test,
        query_limit=eval_query_limit,
        corpus_limit=eval_corpus_limit,
        seed=cfg.seed,
    )

    # ── 3. 모델 로드 ─────────────────────────────────────────────
    print(f"\nLoading model: {model_key}...")
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}  choices={list(MODEL_CONFIGS)}")
    mcfg = MODEL_CONFIGS[model_key]
    if mcfg["embedder_type"] == "biencoder":
        embedder = HFBiEncoderEmbedder(
            mcfg["hf_name"],
            pool_mode=mcfg["pool_mode"],
            query_prefix=mcfg.get("query_prefix", ""),
            doc_prefix=mcfg.get("doc_prefix", ""),
            device=cfg.device,
        )
    else:
        embedder = HFLLMPoolEmbedder(
            mcfg["hf_name"],
            cfg.device,
            use_auto_device_map=mcfg["use_auto"],
        )

    # 바이인코더의 경우 prefix를 미리 텍스트에 적용
    _doc_prefix   = getattr(embedder, "doc_prefix",   "")
    _query_prefix = getattr(embedder, "query_prefix", "")

    # ── 4. Base 임베딩 인코딩 + 캐시 ─────────────────────────────
    fit_cache_name = f"fit_{fit_mode}"
    if fit_mode in {"corpus_sample", "mixed", "query_sample"} and fit_sample is not None:
        fit_cache_name += f"_N{fit_sample}"

    # 바이인코더 prefix 적용 (캐시와 분리를 위해 model_key에 이미 포함됨)
    corpus_texts_enc = (
        [_doc_prefix + t for t in eval_corpus_texts] if _doc_prefix else eval_corpus_texts
    )
    query_texts_enc = (
        [_query_prefix + t for t in eval_test_qtexts] if _query_prefix else eval_test_qtexts
    )
    # fit sentences: 내용에 따라 prefix 선택
    # corpus_sample/mixed → doc_prefix, dev_queries/query_sample → query_prefix
    _fit_prefix = _doc_prefix if fit_mode in {"corpus_sample", "mixed"} else _query_prefix
    fit_sentences_enc = (
        [_fit_prefix + t for t in fit_sentences] if _fit_prefix else fit_sentences
    )

    base_corpus_embs = _load_or_compute_embeddings(
        _embedding_cache_path(dataset_name, model_key, f"corpus_base{eval_suffix}"),
        f"corpus {len(corpus_texts_enc):,}개 (1회 인코딩 후 전 방법 재사용)",
        "corpus",
        corpus_texts_enc,
        embedder,
        batch_size=256,
    )

    base_query_embs = _load_or_compute_embeddings(
        _embedding_cache_path(dataset_name, model_key, f"test_queries_base{eval_suffix}"),
        f"test queries {len(query_texts_enc):,}개",
        "queries",
        query_texts_enc,
        embedder,
        batch_size=256,
    )

    X_fit = _load_or_compute_embeddings(
        _embedding_cache_path(dataset_name, model_key, fit_cache_name),
        f"fit data {len(fit_sentences_enc):,}개",
        "fit",
        fit_sentences_enc,
        embedder,
        batch_size=256,
    )
    base_dim = X_fit.shape[1]

    dims = []
    curr = base_dim // 2
    while curr >= 4: dims.append(curr); curr //= 2
    print(f"Target dims: {dims}")

    seed_suffix = f"_s{seed}" if seed != 42 else ""
    tag = f"{model_key}{_qs_n_suffix}{eval_suffix}{seed_suffix}"

    # ── 체크포인트: 기존 진행분 로드 ──────────────────────────────
    ckpt_path = os.path.join(out_dir, f"{tag}_checkpoint.csv")
    if os.path.exists(ckpt_path):
        ckpt_df = pd.read_csv(ckpt_path)
        results  = ckpt_df.to_dict("records")
        done_set = {(r["method"], int(r["dim"])) for r in results}
        print(f"\n[재개] 체크포인트 로드: {len(results)}개 결과 복원 ({ckpt_path})")
    else:
        results  = []
        done_set = set()

    def _already(method_name, dim):
        return (method_name, int(dim)) in done_set

    def _eval(c_embs, q_embs, method_name, dim):
        """pre-projected embeddings로 평가 후 결과 dict 반환 + 체크포인트 즉시 저장"""
        if _already(method_name, dim):
            cached = next(r for r in results if r["method"] == method_name and int(r["dim"]) == int(dim))
            print(f"  {method_name:<16} dim={dim:>5}  [체크포인트에서 복원]")
            return cached
        m, ex = evaluate_retrieval_precomputed(
            c_embs, eval_corpus_ids, q_embs, eval_test_qids, eval_qrels_test,
            eval_chunk=eval_chunk, return_anisotropy=True, return_self_sim=True)
        row = {"dim": dim, "method": method_name, **m,
               "anisotropy": ex[0], "self_sim": ex[1]}
        print(f"  {method_name:<16} dim={dim:>5}  NDCG@10={m['ndcg@10']:.2f}"
              f"  Recall@100={m['recall@100']:.2f}  MAP={m['map']:.2f}")
        results.append(row)
        done_set.add((method_name, int(dim)))
        # 결과 즉시 체크포인트에 저장 (크래시 대비)
        pd.DataFrame(results).to_csv(ckpt_path, index=False)
        return row

    # ── 5-0. BM25 baseline ───────────────────────────────────────
    if run_bm25 and not ncwp_only and not _already("BM25", base_dim):
        print("\n[BM25 baseline]")
        bm25_m = evaluate_bm25(
            eval_corpus_texts, eval_corpus_ids,
            eval_test_qtexts, eval_test_qids, eval_qrels_test,
        )
        if bm25_m is not None:
            row = {"dim": base_dim, "method": "BM25",
                   "anisotropy": float("nan"), "self_sim": float("nan"), **bm25_m}
            print(f"  {'BM25':<16} dim={base_dim:>5}  "
                  f"NDCG@10={bm25_m['ndcg@10']:.2f}"
                  f"  Recall@100={bm25_m['recall@100']:.2f}"
                  f"  MAP={bm25_m['map']:.2f}")
            results.append(row)
            done_set.add(("BM25", int(base_dim)))
            pd.DataFrame(results).to_csv(ckpt_path, index=False)

    # ── 5. Base ──────────────────────────────────────────────────
    if not ncwp_only:
        print("\n[Base]")
        _eval(base_corpus_embs, base_query_embs, "Base", base_dim)

    # ── 6. PCA 사전 계산 ─────────────────────────────────────────
        print("\nComputing PCA whitening matrix...")
        mu_pca, W_pca, comps, scales = compute_pca_whitening_matrix(X_fit, base_dim)

    # ── 7. 표준 방법들 (dim별 projection만 수행) ──────────────────
        print(f"\n[Standard methods × {len(dims)} dims]")
        mu_rand = X_fit.mean(0)
        for k in dims:
            print(f"\n  --- dim={k} ---")
            c = proj_linear(base_corpus_embs, W_pca[:, :k], mu_pca)
            q = proj_linear(base_query_embs,  W_pca[:, :k], mu_pca)
            _eval(c, q, "PCA-White", k)

            c = proj_soft_white(base_corpus_embs, mu_pca, comps, scales, k)
            q = proj_soft_white(base_query_embs,  mu_pca, comps, scales, k)
            _eval(c, q, "Soft-White", k)

            W_rand = fit_random_projector(base_dim, k, cfg.seed)
            c = proj_linear(base_corpus_embs, W_rand, mu_rand)
            q = proj_linear(base_query_embs,  W_rand, mu_rand)
            _eval(c, q, "Random", k)

            W_lpp, mu_lpp = fit_lpp_projector(X_fit, k=k)
            c = proj_linear(base_corpus_embs, W_lpp, mu_lpp)
            q = proj_linear(base_query_embs,  W_lpp, mu_lpp)
            _eval(c, q, "LPP", k)

    # ── 8. NCWP Ablation ─────────────────────────────────────────
    ncwp_configs = [
        {"name": "NCWP_Default", "k": 10, "cov": 0.05, "orth": 0.02, "tau": 0.0},
    ]

    for conf in ncwp_configs:
        name = conf["name"]
        print(f"\n[NCWP: {name}]")
        for k in dims:
            seed_dir_sfx = f"_s{seed}" if seed != 42 else ""
            n_dir_sfx    = f"_N{fit_sample}" if fit_mode == "query_sample" and fit_sample is not None else ""
            wdir  = os.path.join(out_dir, "weights", f"{model_key}{n_dir_sfx}{seed_dir_sfx}", name)
            wpath = os.path.join(wdir, f"dim_{k}.npz")
            if os.path.exists(wpath):
                d = np.load(wpath)
            else:
                print(f"  dim={k}: training...", end=" ", flush=True)
                tr = NewProposalTrainer(
                    rank=k,
                    k_neighbors=conf["k"],
                    lambda_cov=conf["cov"],
                    lambda_orth=conf["orth"],
                    cosine_tau=conf.get("tau", 0.1),
                    max_epochs=cfg.epochs,
                    device=str(cfg.device),
                    seed=seed,
                )
                W, mu_in, mu_out, std = tr.fit(X_fit)
                os.makedirs(wdir, exist_ok=True)
                np.savez(wpath, W=W, mu_in=mu_in, mu_out=mu_out, std_out=std)
                print("done"); d = {"W": W, "mu_in": mu_in, "mu_out": mu_out, "std_out": std}

            c = proj_ncwp(base_corpus_embs, d["W"], d["mu_in"], d["mu_out"], d["std_out"])
            q = proj_ncwp(base_query_embs,  d["W"], d["mu_in"], d["mu_out"], d["std_out"])
            _eval(c, q, name, k)

    # ── 9. 최종 CSV & 플롯 저장 ──────────────────────────────────
    # 체크포인트에서 main / ablation 분리
    all_results = pd.read_csv(ckpt_path).to_dict("records")

    ncwp_names   = {c["name"] for c in ncwp_configs}
    std_methods  = {"Base", "BM25", "PCA-White", "Soft-White", "Random", "LPP"}
    results_main = []
    results_ablation = []
    for r in all_results:
        m = r["method"]
        if m in std_methods:
            results_main.append(r)
            if m in {"Base", "BM25"}:
                results_ablation.append(r)
        elif m in ncwp_names:
            results_ablation.append(r)
            if m == "NCWP_Default":
                results_main.append({**r, "method": "NCWP"})

    pd.DataFrame(results_main).to_csv(
        os.path.join(out_dir, f"{tag}_results_main.csv"), index=False)
    pd.DataFrame(results_ablation).to_csv(
        os.path.join(out_dir, f"{tag}_results_ablation.csv"), index=False)
    pdf_path = os.path.join(out_dir, f"{tag}_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_results(results_main, f"{tag}_main", out_dir, pdf=pdf)
        plot_results(results_ablation, f"{tag}_ablation", out_dir, pdf=pdf)
    print(f"PDF 저장: {pdf_path}")

    # 성공적으로 완료된 경우 체크포인트 삭제 (정리)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  체크포인트 삭제 완료: {ckpt_path}")

    print(f"\n{'='*65}")
    print(f"  Done! → {out_dir}/")
    cols = ["dim", "method", "ndcg@10", "recall@100", "map"]
    print(pd.DataFrame(results_main)[cols].to_string(index=False))


# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="BEIR NCWP Experiment (v3)")
    p.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()),
                   help="평가 데이터셋: " + " / ".join(DATASET_CONFIGS.keys()))
    p.add_argument("--model",      required=True, choices=list(MODEL_CONFIGS.keys()),
                   help="임베딩 모델: " + " / ".join(MODEL_CONFIGS.keys()))
    p.add_argument("--fit_mode",   default="corpus_sample",
                   choices=["corpus_sample", "dev_queries", "mixed", "query_sample"],
                   help=("corpus_sample [방법A]: corpus 랜덤 샘플 | "
                         "dev_queries [방법C]: dev 쿼리 | "
                         "mixed [방법A+C]: 혼합 | "
                         "query_sample [방법D]: test/dev queries 샘플"))
    p.add_argument("--fit_sample", type=int, default=None,
                   help="corpus_sample/mixed 시 corpus에서 사용할 샘플 수 (기본값: dataset별 설정)")
    p.add_argument("--eval_query_limit", type=int, default=None,
                   help="빠른 실험용 평가 query 개수 제한")
    p.add_argument("--eval_corpus_limit", type=int, default=None,
                   help="빠른 실험용 평가 corpus 개수 제한 (정답 문서는 항상 포함)")
    p.add_argument("--no_bm25", action="store_true",
                   help="BM25 baseline 평가 생략 (대용량 corpus에서 속도 절약)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42). Multi-seed 실험용")
    p.add_argument("--ncwp_only", action="store_true",
                   help="NCWP 학습/평가만 수행 (Base/PCA 등 deterministic 방법 스킵). Multi-seed 실험용")
    args = p.parse_args()

    # seed 덮어쓰기
    cfg.seed = args.seed
    _set_seed(cfg.seed)

    run_experiment(
        args.dataset,
        args.model,
        args.fit_mode,
        args.fit_sample,
        args.eval_query_limit,
        args.eval_corpus_limit,
        run_bm25=not args.no_bm25,
        ncwp_only=args.ncwp_only,
        seed=args.seed,
    )
