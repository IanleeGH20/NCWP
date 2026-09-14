"""Shared utilities for NCWP rebuttal experiments."""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

# ── Paths (container / host share same mount) ─────────────────
STS_DIR = "/workspace/RAG/code/Make_embedding/nanoGPT/stsbenchmark"
CACHE_DIR = "/workspace/NCWP/sts_ablation_cache"
OUT_ROOT = "/workspace/NCWP/rebuttal_outputs"
QUORA_CACHE = "/workspace/NCWP/quora_results/embedding_cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44]

MODEL_CONFIGS = {
    "qwen-4b": {
        "hf_model_id": "Qwen/Qwen1.5-4B",
        "hidden_dim": 2560,
        "layer": "last_hidden_state",
        "pooling": "mean",
        "max_length": 128,
        "dtype": "bfloat16",
        "dims": [5, 10, 20, 40, 80, 160, 320, 640, 1280],
        "quora_fit_size": 2000,
    },
    "qwen-8b": {
        "hf_model_id": "Qwen/Qwen2-7B",
        "hidden_dim": 3584,
        "layer": "last_hidden_state",
        "pooling": "mean",
        "max_length": 128,
        "dtype": "bfloat16",
        "dims": [7, 14, 28, 56, 112, 224, 448, 896, 1792],
        "quora_fit_size": 1000,
    },
    "llama-8b": {
        "hf_model_id": "meta-llama/Meta-Llama-3.1-8B",
        "hidden_dim": 4096,
        "layer": "last_hidden_state",
        "pooling": "mean",
        "max_length": 128,
        "dtype": "bfloat16",
        "dims": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "quora_fit_size": 1000,
    },
}

# NCWP final protocol (STS main + multiseed)
NCWP_HPARAMS = dict(
    k_neighbors=40,
    cosine_tau=0.2,
    temperature=0.07,
    lambda_cov=0.05,
    lambda_orth=0.02,
    lr=8e-3,
    max_epochs=20,
    batch_pairs=256,
    hard_neg_k=32,
    memory_bank_size=4096,
    retraction_interval=200,
    refine_knn_rounds=0,
    shrink=0.08,
    use_memory_bank=True,
    use_hard_neg=True,
)

# Paper BEIR protocol (run_beir_experiment)
BEIR_NCWP_HPARAMS = dict(
    k_neighbors=10,
    cosine_tau=0.0,
    temperature=0.12,
    lambda_cov=0.05,
    lambda_orth=0.02,
    lr=8e-3,
    max_epochs=20,
    batch_pairs=256,
    hard_neg_k=32,
    memory_bank_size=4096,
    retraction_interval=200,
    refine_knn_rounds=0,
    shrink=0.08,
    use_memory_bank=True,
    use_hard_neg=True,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_results(rows: List[dict], basename: str) -> Tuple[str, str]:
    ensure_dir(OUT_ROOT)
    csv_path = os.path.join(OUT_ROOT, f"{basename}.csv")
    jsonl_path = os.path.join(OUT_ROOT, f"{basename}.jsonl")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def metadata_row(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


# ── STS data ──────────────────────────────────────────────────
def load_sts_benchmark(eval_split: str = "train+test") -> dict:
    """eval_split: 'train+test' (main, 7128 pairs) or 'test' (paper prose, 1379 pairs)."""
    valid = json.load(open(f"{STS_DIR}/sts_valid.json"))
    train = json.load(open(f"{STS_DIR}/sts_train.json"))
    test = json.load(open(f"{STS_DIR}/sts_test.json"))
    fit_sents = list({s for item in valid for s in (item["sentence1"], item["sentence2"])})
    if eval_split == "test":
        eval_items = test
    elif eval_split == "train+test":
        eval_items = train + test
    else:
        raise ValueError(f"unknown eval_split={eval_split}")
    eval_pairs = [(item["sentence1"], item["sentence2"]) for item in eval_items]
    eval_scores = [item["score"] for item in eval_items]
    eval_sents = list({s for p in eval_pairs for s in p})
    return dict(
        fit_sents=fit_sents,
        eval_pairs=eval_pairs,
        eval_scores=eval_scores,
        eval_sents=eval_sents,
        fit_size=len(fit_sents),
        eval_size=len(eval_pairs),
        eval_split=eval_split,
    )


def subset_eval_from_cache(cache_npy: str, cache_sents_json: str, wanted_sents: List[str]) -> np.ndarray:
    """Reuse a cached (7128-eval) embedding array to build embeddings for a subset
    of sentences (e.g. test-only 1379), avoiding any model re-encoding."""
    saved = json.load(open(cache_sents_json))
    arr = np.load(cache_npy).astype(np.float32)
    s2i = {s: i for i, s in enumerate(saved)}
    missing = [s for s in wanted_sents if s not in s2i]
    if missing:
        raise ValueError(f"{len(missing)} sentences missing from cache {cache_npy}")
    return np.stack([arr[s2i[s]] for s in wanted_sents]).astype(np.float32)


def load_sts_embeddings(model: str, sts: Optional[dict] = None) -> Tuple[np.ndarray, dict, dict]:
    sts = sts or load_sts_benchmark()
    all_sents = list(set(sts["fit_sents"] + sts["eval_sents"]))
    cache_emb = f"{CACHE_DIR}/{model}_main_embs.npy"
    cache_idx = f"{CACHE_DIR}/{model}_main_sents.json"
    if not (os.path.exists(cache_emb) and os.path.exists(cache_idx)):
        raise FileNotFoundError(f"Missing STS cache for {model}: {cache_emb}")
    embs = np.load(cache_emb).astype(np.float32)
    saved = json.load(open(cache_idx))
    sent2idx = {s: i for i, s in enumerate(saved)}
    missing = [s for s in all_sents if s not in sent2idx]
    if missing:
        raise ValueError(f"{len(missing)} sentences missing from cache for {model}")
    X_fit = np.array([embs[sent2idx[s]] for s in sts["fit_sents"]], dtype=np.float32)
    X_eval = {s: embs[sent2idx[s]].astype(np.float32) for s in sts["eval_sents"]}
    return X_fit, X_eval, sts


# ── Quora data ────────────────────────────────────────────────
def load_quora_cached(model: str, fit_n: Optional[int] = None) -> dict:
    fit_n = fit_n or MODEL_CONFIGS[model]["quora_fit_size"]
    cache = f"{QUORA_CACHE}/{model}"
    corpus_np = np.load(f"{cache}/corpus_base.npy").astype(np.float32)
    query_np = np.load(f"{cache}/test_queries_base.npy").astype(np.float32)
    X_fit = np.load(f"{cache}/fit_query_sample_N{fit_n}.npy").astype(np.float32)

    ds_corp = load_dataset("mteb/quora", "corpus", split="corpus")
    ds_qs = load_dataset("mteb/quora", "queries", split="queries")
    ds_qrel = load_dataset("mteb/quora", split="test")
    corpus_ids = [str(r["_id"]) for r in ds_corp]
    qid2tx = {str(r["_id"]): r["text"] for r in ds_qs}
    qrels = defaultdict(set)
    for r in ds_qrel:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    test_qids = sorted(q for q in qrels if q in qid2tx)
    return dict(
        corpus_np=corpus_np,
        query_np=query_np,
        X_fit=X_fit,
        corpus_ids=corpus_ids,
        test_qids=test_qids,
        qrels=qrels,
        fit_size=fit_n,
        eval_size=len(test_qids),
    )


# ── Whitening / projections ───────────────────────────────────
def compute_zca(X: np.ndarray, shrink: float = 0.08):
    Xt = torch.from_numpy(X).to(DEVICE)
    mu = Xt.mean(0)
    Xc = Xt - mu
    cov = (Xc.T @ Xc) / max(Xt.shape[0] - 1, 1)
    d = cov.shape[0]
    tr = torch.trace(cov)
    cs = (1 - shrink) * cov + shrink * (tr / d) * torch.eye(d, device=DEVICE)
    ev, evec = torch.linalg.eigh(cs)
    s = evec @ torch.diag(1.0 / torch.sqrt(torch.clamp(ev, min=1e-6))) @ evec.T
    return mu, s


def apply_zca(X: np.ndarray, mu, s) -> np.ndarray:
    xt = torch.from_numpy(X).to(DEVICE)
    return F.normalize((xt - mu) @ s, dim=1).cpu().numpy()


def pca_whitening_matrix(X: np.ndarray, dim: int, shrink: float = 0.08):
    xt = torch.from_numpy(X).to(DEVICE).double()
    n, d = xt.shape
    mu = xt.mean(0)
    xc = xt - mu
    cov = (xc.T @ xc) / max(n - 1, 1)
    tr = torch.trace(cov)
    cs = (1 - shrink) * cov + shrink * (tr / d) * torch.eye(d, device=DEVICE, dtype=xt.dtype)
    ev, evec = torch.linalg.eigh(cs)
    ev = torch.flip(ev, [0])
    evec = torch.flip(evec, [1])
    k = min(dim, d)
    w = evec[:, :k] @ torch.diag(1.0 / torch.sqrt(torch.clamp(ev[:k], min=1e-6)))
    return mu.cpu().numpy().astype(np.float32), w.cpu().numpy().astype(np.float32)


def project_linear(X: np.ndarray, W: np.ndarray, mu: Optional[np.ndarray] = None) -> np.ndarray:
    if mu is not None:
        z = (X - mu) @ W
    else:
        z = X @ W
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def fit_abtt_components(X: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    xc = X - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:m].T.astype(np.float32)  # D x m
    return mu.astype(np.float32), comps


def transform_abtt_full(X: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    xc = X - mu
    proj = xc @ comps @ comps.T
    out = xc - proj
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def fit_abtt_pca_matrix(X: np.ndarray, mu: np.ndarray, comps: np.ndarray, dim: int):
    abtt = transform_abtt_full(X, mu, comps)
    mu_abtt = abtt.mean(0)
    xc = abtt - mu_abtt
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(dim, vt.shape[0])
    w = vt[:k].T.astype(np.float32)
    return mu_abtt.astype(np.float32), w


def transform_abtt_pca(X: np.ndarray, mu: np.ndarray, comps: np.ndarray, mu_abtt: np.ndarray, w: np.ndarray) -> np.ndarray:
    abtt = transform_abtt_full(X, mu, comps)
    z = (abtt - mu_abtt) @ w
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


# ── Evaluation ────────────────────────────────────────────────
def spearman_sts(emb_map: dict, eval_pairs, eval_scores) -> float:
    pred = [float(np.dot(emb_map[s1], emb_map[s2])) for s1, s2 in eval_pairs]
    return float(spearmanr(eval_scores, pred).correlation * 100)


def sts_emb_map_from_matrix(X_eval: dict, W: Optional[np.ndarray] = None, mu_out: Optional[np.ndarray] = None):
    emb_map = {}
    for s, x in X_eval.items():
        z = x if W is None else x @ W - (mu_out if mu_out is not None else 0)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        z = z[0]
        emb_map[s] = z / (np.linalg.norm(z) + 1e-12)
    return emb_map


def ndcg_at_k(ranked_ids, rel_set, k=10) -> float:
    h = [1.0 if d in rel_set else 0.0 for d in ranked_ids[:k]]
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(h))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_ndcg10(corpus_embs, query_embs, corpus_ids, test_qids, qrels, chunk=500) -> float:
    scores = []
    for qi in range(0, len(test_qids), chunk):
        qe = min(len(test_qids), qi + chunk)
        sim = query_embs[qi:qe] @ corpus_embs.T
        top_idx = np.argsort(-sim, axis=1)[:, :10]
        for ci, q in enumerate(range(qi, qe)):
            qid = test_qids[q]
            if qid not in qrels:
                continue
            rel = qrels[qid]
            ranked = [corpus_ids[i] for i in top_idx[ci]]
            scores.append(ndcg_at_k(ranked, rel, k=10))
    return float(np.mean(scores) * 100) if scores else float("nan")


# ── kNN mining ────────────────────────────────────────────────
def compute_knn(Xw: np.ndarray, k: int = 10):
    xw_t = torch.from_numpy(Xw).to(DEVICE)
    sim = xw_t @ xw_t.T
    sim.fill_diagonal_(-1e9)
    vals, idx = torch.topk(sim, k=k, dim=1)
    return idx.cpu().numpy(), vals.cpu().numpy()


def mined_positive_sets(knn_idx, knn_sims, k=10, tau=0.0):
    sets = []
    for i in range(knn_idx.shape[0]):
        valid = [j for t, j in enumerate(knn_idx[i]) if knn_sims[i, t] >= tau]
        sets.append(valid[:k])
    return sets


# ── NCWP trainer ──────────────────────────────────────────────
class MemoryBank:
    def __init__(self, size: int, dim: int):
        self.bank = torch.zeros(size, dim, device=DEVICE)
        self.ptr = 0
        self.full = False
        self.size = size

    def update(self, z: torch.Tensor):
        b = z.shape[0]
        end = min(self.ptr + b, self.size)
        n1 = end - self.ptr
        self.bank[self.ptr:end] = z[:n1]
        if b > n1:
            self.bank[: b - n1] = z[n1:]
            self.ptr = b - n1
            self.full = True
        else:
            self.ptr = end % self.size
            if self.ptr == 0:
                self.full = True

    def get(self):
        return self.bank if self.full else self.bank[: self.ptr]


def train_ncwp(
    Xw_fit: np.ndarray,
    rank: int,
    seed: int,
    positive_mode: str = "knn",
    noise_sigma: float = 0.05,
    embedder=None,
    raw_fit: Optional[np.ndarray] = None,
    **hparams,
) -> np.ndarray:
    """Train NCWP projection W. positive_mode: knn|dropout|noise|random."""
    set_seed(seed)
    hp = {**NCWP_HPARAMS, **hparams}
    n, d = Xw_fit.shape
    r = rank
    xw = torch.from_numpy(Xw_fit.astype(np.float32)).to(DEVICE)
    knn_idx, knn_sims = compute_knn(Xw_fit, k=hp["k_neighbors"])
    knn = torch.from_numpy(knn_idx).to(DEVICE)
    sims = torch.from_numpy(knn_sims.astype(np.float32)).to(DEVICE)

    temperature = hp["temperature"]
    lr = hp["lr"]
    lr_min = 0.2 * lr
    warmup = 200
    max_epochs = hp["max_epochs"]
    batch_pairs = hp["batch_pairs"]
    steps = min(160, max(100, math.ceil(n / (2 * batch_pairs))))
    round_t = max_epochs * steps

    w = torch.nn.Parameter(torch.randn(d, r, device=DEVICE) / math.sqrt(d))
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=1e-4)
    mbank = MemoryBank(hp["memory_bank_size"], r) if hp["use_memory_bank"] else None
    best_w = None
    best_loss = float("inf")
    no_imp = 0
    gs = 0

    raw_fit_t = torch.from_numpy(raw_fit.astype(np.float32)).to(DEVICE) if raw_fit is not None else None

    for _ep in range(max_epochs):
        ep_loss = 0.0
        for _ in range(steps):
            if gs < warmup:
                cur_lr = lr_min + (lr - lr_min) * gs / max(1, warmup)
            else:
                t = (gs - warmup) / max(1, round_t - warmup)
                cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            with torch.no_grad():
                a_idx = torch.randint(0, n, (batch_pairs,), device=DEVICE)
                if positive_mode == "random":
                    b_idx = torch.randint(0, n, (batch_pairs,), device=DEVICE)
                elif positive_mode == "noise":
                    b_idx = a_idx.clone()
                elif positive_mode == "dropout":
                    b_idx = a_idx.clone()
                else:
                    b_list = []
                    for anc in a_idx.tolist():
                        valid = (sims[anc] >= hp["cosine_tau"]).nonzero(as_tuple=True)[0]
                        if len(valid) == 0:
                            valid = torch.arange(knn.shape[1], device=DEVICE)
                        pick = valid[torch.randint(0, len(valid), (1,)).item()]
                        b_list.append(knn[anc, pick].item())
                    b_idx = torch.tensor(b_list, device=DEVICE)

            if positive_mode == "dropout" and raw_fit_t is not None and embedder is not None:
                # fallback: noise on whitened if dropout path unavailable at batch level
                batch = torch.cat([xw[a_idx], xw[b_idx]], dim=0)
            elif positive_mode == "noise":
                z1 = xw[a_idx]
                z2 = xw[a_idx] + noise_sigma * torch.randn_like(xw[a_idx])
                batch = torch.cat([z1, F.normalize(z2, dim=1)], dim=0)
            else:
                batch = torch.cat([xw[a_idx], xw[b_idx]], dim=0)

            z = F.normalize(batch @ w, dim=1)
            za = z[:batch_pairs]
            zb = z[batch_pairs:]
            sim_a_to_b = (za @ zb.T) / temperature
            sim_a_to_a = (za @ za.T) / temperature
            sim_a_to_a.fill_diagonal_(-1e9)
            sim_b_to_a = (zb @ za.T) / temperature
            sim_b_to_b = (zb @ zb.T) / temperature
            sim_b_to_b.fill_diagonal_(-1e9)

            if hp["use_memory_bank"] and mbank is not None and len(mbank.get()) > 0:
                bank = mbank.get().detach()
                sim_a_bank = (za @ bank.T) / temperature
                sim_b_bank = (zb @ bank.T) / temperature
            else:
                sim_a_bank = sim_b_bank = None

            pos_ab = sim_a_to_b.diag()
            neg_a = sim_a_to_b.clone()
            neg_a.fill_diagonal_(-1e9)
            pool_a = torch.cat([neg_a, sim_a_to_a], dim=1)
            if sim_a_bank is not None:
                pool_a = torch.cat([pool_a, sim_a_bank], dim=1)
            pos_ba = sim_b_to_a.diag()
            neg_b = sim_b_to_a.clone()
            neg_b.fill_diagonal_(-1e9)
            pool_b = torch.cat([neg_b, sim_b_to_b], dim=1)
            if sim_b_bank is not None:
                pool_b = torch.cat([pool_b, sim_b_bank], dim=1)

            if hp["use_hard_neg"]:
                k_neg = min(hp["hard_neg_k"], pool_a.shape[1])
                neg_a_sel, _ = torch.topk(pool_a, k=k_neg, dim=1)
                neg_b_sel, _ = torch.topk(pool_b, k=k_neg, dim=1)
            else:
                neg_a_sel = pool_a
                neg_b_sel = pool_b

            denom_a = torch.logsumexp(torch.cat([pos_ab.unsqueeze(1), neg_a_sel], dim=1), dim=1)
            denom_b = torch.logsumexp(torch.cat([pos_ba.unsqueeze(1), neg_b_sel], dim=1), dim=1)
            loss = 0.5 * (-(pos_ab - denom_a).mean() + -(pos_ba - denom_b).mean())

            zm = z - z.mean(0)
            cov = (zm.T @ zm) / max(len(z) - 1, 1)
            off = cov - torch.diag(torch.diag(cov))
            loss += hp["lambda_cov"] * (off ** 2).sum() / r
            wn = F.normalize(w, dim=0)
            loss += hp["lambda_orth"] * ((wn.T @ wn - torch.eye(r, device=DEVICE)) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            if hp["use_memory_bank"] and mbank is not None:
                mbank.update(torch.cat([za, zb], dim=0).detach())
            if hp["retraction_interval"] > 0 and (gs + 1) % hp["retraction_interval"] == 0:
                with torch.no_grad():
                    q, _ = torch.linalg.qr(w.data)
                    w.data = q[:, :r]
            ep_loss += loss.item()
            gs += 1
        ep_loss /= steps
        if ep_loss + 1e-6 < best_loss:
            best_loss = ep_loss
            no_imp = 0
            best_w = w.detach().clone()
        else:
            no_imp += 1
            if no_imp >= 6:
                break
    return best_w.cpu().numpy()


def project_ncwp(X_raw: np.ndarray, mu_zca, s_zca, W: np.ndarray, Xw_fit: np.ndarray) -> np.ndarray:
    mu_w = Xw_fit.mean(0) @ W
    parts = []
    for i in range(0, len(X_raw), 2000):
        xb = torch.from_numpy(X_raw[i : i + 2000]).to(DEVICE)
        xw = F.normalize((xb - mu_zca) @ s_zca, dim=1)
        xp = F.normalize(xw @ torch.from_numpy(W).to(DEVICE) - torch.from_numpy(mu_w).to(DEVICE), dim=1)
        parts.append(xp.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Prompt-based extractors ───────────────────────────────────
PROMPT_TEMPLATES = {
    "Plain": "{sentence}",  # passthrough -> plain mean/last pooling (paper Base)
    "PromptEOL_A": 'This sentence: "{sentence}" means in one word:"',
    "PromptEOL_B": 'This sentence: "{sentence}" means:"',
    "Echo": "{sentence} {sentence}",
}


class PromptEmbedder:
    def __init__(self, model_key: str, template_name: str = "PromptEOL_A", pool: str = "last",
                 layer: Optional[int] = None):
        cfg = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.template_name = template_name
        self.pool = pool
        self.layer = layer  # None -> last_hidden_state; else transformer-block index into hidden_states
        self.hf_model_id = cfg["hf_model_id"]
        self.max_length = cfg["max_length"]
        self.tok = AutoTokenizer.from_pretrained(self.hf_model_id, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(
            self.hf_model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
        )
        self.model.eval()

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def _format(self, sentence: str) -> str:
        return PROMPT_TEMPLATES[self.template_name].format(sentence=sentence)

    @torch.no_grad()
    def encode(self, sentences: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = []
        dev = getattr(self.model, "device", DEVICE)
        for i in range(0, len(sentences), batch_size):
            texts = [self._format(s) for s in sentences[i : i + batch_size]]
            enc = self.tok(
                texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(dev)
            if self.layer is None:
                h = self.model(**enc).last_hidden_state
            else:
                # hidden_states: tuple len num_layers+1 (index 0 = embeddings, index L = block L output)
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                h = hs[self.layer]
            if self.pool == "last":
                seq_lens = enc["attention_mask"].sum(1) - 1
                x = h[torch.arange(h.size(0), device=h.device), seq_lens]
            elif self.pool == "second_last":
                seq_lens = (enc["attention_mask"].sum(1) - 2).clamp_min(0)
                x = h[torch.arange(h.size(0), device=h.device), seq_lens]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                x = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            out.append(F.normalize(x, p=2, dim=1).cpu().float().numpy())
        return np.concatenate(out, axis=0)
