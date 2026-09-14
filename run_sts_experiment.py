"""
run_sts_experiment.py  — Tier 1-1: Sentence Similarity Datasets
MRPC + SICK를 BEIR-style retrieval 태스크로 재구성하여 NCWP 실험 수행.

Reformulation 방식:
  MRPC  : corpus = 전체 문장 집합, query = test sentence1,
          relevant = label=1인 sentence2 파트너
  SICK  : corpus = 전체 문장 집합, query = test sentence_A,
          relevant = relatedness_score >= 4.0인 sentence_B

실행:
  python run_sts_experiment.py --dataset mrpc  --model e5-base
  python run_sts_experiment.py --dataset sick  --model e5-base
  python run_sts_experiment.py --dataset mrpc  --model bge-base
  python run_sts_experiment.py --dataset sick  --model bge-base
"""

import os, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ── NVRTC fix ──────────────────────────────────────────────
for _p in ["/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
           "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"]:
    if os.path.exists(_p) and _p not in os.environ.get("LD_LIBRARY_PATH",""):
        os.environ["LD_LIBRARY_PATH"] = f"{_p}:{os.environ.get('LD_LIBRARY_PATH','')}"
        break

os.environ["HF_HOME"] = "/workspace/RAG/code/Make_embedding/nanoGPT"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
RESULTS_ROOT = "/workspace/NCWP"
SICK_REL_THRESHOLD = 4.0   # SICK relatedness_score >= 4.0 → relevant

MODEL_CONFIGS = {
    "e5-base":  {"hf_name": "intfloat/e5-base",         "pool": "mean",
                 "query_prefix": "query: ", "doc_prefix": "passage: "},
    "bge-base": {"hf_name": "BAAI/bge-base-en-v1.5",    "pool": "cls",
                 "query_prefix": "", "doc_prefix": ""},
    "llama-8b": {"hf_name": "meta-llama/Meta-Llama-3.1-8B", "pool": "mean",
                 "query_prefix": "", "doc_prefix": "", "use_auto": True, "llm": True},
    "qwen-4b":  {"hf_name": "Qwen/Qwen1.5-4B",          "pool": "mean",
                 "query_prefix": "", "doc_prefix": "", "use_auto": True, "llm": True},
}

import random
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ── Dataset Loading ────────────────────────────────────────

def load_mrpc():
    """MRPC를 retrieval 형식으로 재구성."""
    print("Loading MRPC (glue/mrpc)...")
    ds = load_dataset("glue", "mrpc")

    # 전체 문장 수집 (corpus = unique sentences)
    sentences = set()
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            sentences.add(row["sentence1"])
            sentences.add(row["sentence2"])
    corpus_ids   = list(sorted(sentences))
    corpus_texts = corpus_ids  # sentence itself is the text
    sent2id = {s: i for i, s in enumerate(corpus_ids)}

    # test qrels: label=1 인 쌍만 relevant (score=1)
    qrels_test = defaultdict(dict)
    for row in ds["test"]:
        if row["label"] == 1:
            qid = row["sentence1"]
            cid = row["sentence2"]
            qrels_test[qid][cid] = 1
            # 대칭: sentence2도 query가 될 수 있음
            qrels_test[cid][qid] = 1

    test_qids   = [q for q in sorted(qrels_test) if q in sent2id]
    test_qtexts = test_qids  # sentence is the query text

    # fit 용: train 전체 문장 (label 무관)
    fit_sentences = list({row["sentence1"] for row in ds["train"]} |
                         {row["sentence2"] for row in ds["train"]})

    print(f"  corpus={len(corpus_ids):,}  test_queries={len(test_qids):,}  "
          f"relevant_pairs={sum(len(v) for v in qrels_test.values())//2:,}")
    return corpus_ids, corpus_texts, test_qids, test_qtexts, dict(qrels_test), fit_sentences


def load_sick():
    """SICK를 retrieval 형식으로 재구성."""
    print("Loading SICK...")
    ds = load_dataset("sick", trust_remote_code=True)

    sentences = set()
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            sentences.add(row["sentence_A"])
            sentences.add(row["sentence_B"])
    corpus_ids   = list(sorted(sentences))
    corpus_texts = corpus_ids
    sent2id = {s: i for i, s in enumerate(corpus_ids)}

    # test qrels: relatedness_score >= SICK_REL_THRESHOLD → relevant
    qrels_test = defaultdict(dict)
    for row in ds["test"]:
        score = row["relatedness_score"]
        if score >= SICK_REL_THRESHOLD:
            qid = row["sentence_A"]
            cid = row["sentence_B"]
            qrels_test[qid][cid] = int(round(score))
            qrels_test[cid][qid] = int(round(score))

    test_qids   = [q for q in sorted(qrels_test) if q in sent2id]
    test_qtexts = test_qids

    fit_sentences = list({row["sentence_A"] for row in ds["train"]} |
                         {row["sentence_B"] for row in ds["train"]})

    print(f"  corpus={len(corpus_ids):,}  test_queries={len(test_qids):,}  "
          f"relevant_pairs={sum(len(v) for v in qrels_test.values())//2:,}")
    return corpus_ids, corpus_texts, test_qids, test_qtexts, dict(qrels_test), fit_sentences


# ── Embedder ───────────────────────────────────────────────

class Embedder:
    def __init__(self, mcfg):
        self.pool = mcfg["pool"]
        self.qpfx = mcfg.get("query_prefix", "")
        self.dpfx = mcfg.get("doc_prefix", "")
        self.tok  = AutoTokenizer.from_pretrained(mcfg["hf_name"], trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        kw = {"trust_remote_code": True}
        if mcfg.get("llm"):
            kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kw["attn_implementation"] = "eager"
            if mcfg.get("use_auto"): kw["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(mcfg["hf_name"], **kw)
        if not mcfg.get("use_auto"): self.model.to(DEVICE)
        self.model.eval()
        self.is_llm = mcfg.get("llm", False)

    @torch.no_grad()
    def encode(self, texts, prefix="", batch=256):
        if prefix: texts = [prefix + t for t in texts]
        embs = []
        dev = DEVICE if not self.is_llm else getattr(self.model, "device", DEVICE)
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i+batch], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(dev)
            h = self.model(**enc).last_hidden_state
            if self.pool == "cls":
                x = h[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                x = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embs.append(F.normalize(x, dim=1).cpu().float().numpy())
        return np.concatenate(embs) if embs else np.array([])


# ── Helpers (from run_beir_experiment) ────────────────────

def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _zca_shrink(X, shrink=0.08, eps=1e-6):
    mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=X.device)
    ev, evec = torch.linalg.eigh(Cs); ev = torch.clamp(ev, min=eps)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T
    return mu.squeeze(0), S

@torch.no_grad()
def _knn(X, k, chunk=2048):
    N, XT = X.shape[0], X.T
    idx = torch.empty((N,k), dtype=torch.long, device="cpu")
    sim = torch.empty((N,k), dtype=torch.float, device="cpu")
    for s in range(0, N, chunk):
        e = min(N, s+chunk); s_ = X[s:e] @ XT
        s_[:, torch.arange(s, e, device=X.device)] = -1e9
        tk = torch.topk(s_, k=k, dim=1)
        idx[s:e] = tk.indices.cpu(); sim[s:e] = tk.values.cpu()
    return idx, sim

def _orth_penalty(W):
    WT_W = W.T @ W
    return ((WT_W - torch.eye(WT_W.shape[0], device=W.device))**2).mean()

def _cov_penalty(Z):
    B = Z.shape[0]
    if B <= 1: return torch.tensor(0., device=Z.device)
    Zc = Z - Z.mean(0, keepdim=True); Cov = (Zc.T @ Zc)/(B-1)
    return ((Cov - torch.eye(Cov.shape[0], device=Z.device))**2).mean()

def _contrastive_loss(sim_row, pos_idx, topk=256):
    B = sim_row.shape[0]
    pos_l = sim_row.gather(1, pos_idx.view(-1,1)).squeeze(1)
    if topk is None or topk >= sim_row.shape[1]-1:
        denom = torch.logsumexp(sim_row, 1)
    else:
        mask = torch.ones_like(sim_row, dtype=torch.bool)
        mask[torch.arange(B, device=sim_row.device), pos_idx] = False
        negs = sim_row.masked_select(mask).view(B,-1)
        kk = min(topk, negs.shape[1])
        vals, _ = torch.topk(negs, k=kk, dim=1)
        denom = torch.logsumexp(torch.cat([vals, pos_l.unsqueeze(1)],1),1)
    return -(pos_l - denom).mean()

def train_ncwp(X_np, rank, k=10, tau=0.0, seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    N, D = X_np.shape; r = min(rank, D)
    X = torch.from_numpy(X_np).to(DEVICE)
    mu_in, S = _zca_shrink(X); Xw = F.normalize((X - mu_in) @ S, dim=1)
    k_ = max(1, min(k, N-1))
    knn_idx, knn_sim = _knn(Xw, k_)
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE)/math.sqrt(D))
    lr, lr_min = 8e-3, 1.6e-3; temp, temp_min = 0.12, 0.06
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    Bp = max(64, min(1024, N//64))
    steps = min(160, max(100, math.ceil(N/(2*Bp))))
    warmup = 100; patience = 6; best_W = None; best_loss = float("inf"); no_imp = 0
    tau_t = torch.tensor(tau, dtype=torch.float)
    gs = 0

    def _round(ki, ks):
        nonlocal gs, best_W, best_loss, no_imp
        T = 20 * steps
        for ep in range(1, 21):
            ep_loss = 0.
            for _ in range(steps):
                t = max(0., gs-warmup)/max(1, T-warmup)
                cur_lr = (lr_min + (lr-lr_min)*(gs/max(1,warmup))) if gs < warmup \
                          else lr_min + 0.5*(lr-lr_min)*(1+math.cos(math.pi*t))
                for pg in opt.param_groups: pg["lr"] = cur_lr
                t2 = gs/max(1,T-1)
                cur_temp = temp_min + 0.5*(temp-temp_min)*(1+math.cos(math.pi*t2))
                with torch.no_grad():
                    ac = torch.randint(0, N, (Bp,), dtype=torch.long)
                    bl = []
                    for anc in ac.tolist():
                        vm = ks[anc] >= tau_t; vn = ki[anc][vm]
                        if len(vn)==0: vn = ki[anc]
                        bl.append(vn[torch.randint(0,len(vn),(1,))].item())
                    bc = torch.tensor(bl, dtype=torch.long)
                    Xb = torch.cat([Xw[ac.to(DEVICE)], Xw[bc.to(DEVICE)]], 0)
                Z = F.normalize(Xb @ W, dim=1); sim_m = (Z @ Z.T)/cur_temp
                B2 = Z.shape[0]; B = B2//2
                sim_m = sim_m.masked_fill(torch.eye(B2, device=DEVICE, dtype=torch.bool), -1e9)
                i1 = torch.arange(B, device=DEVICE); i2 = i1+B
                loss = (0.5*(_contrastive_loss(sim_m[i1], i2) + _contrastive_loss(sim_m[i2], i1))
                        + 0.05*_cov_penalty(Z) + 0.02*_orth_penalty(W))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if (gs+1)%10==0:
                    with torch.no_grad():
                        Q,_ = torch.linalg.qr(W.data); W.data = Q[:,:r]
                ep_loss += float(loss.item()); gs += 1
            ep_loss /= max(1, steps)
            if ep_loss+1e-6 < best_loss:
                best_loss = ep_loss; no_imp = 0; best_W = W.detach().clone()
            else:
                no_imp += 1
                if no_imp >= patience: break

    _round(knn_idx, knn_sim)
    with torch.no_grad():
        Zf = F.normalize(Xw @ best_W, dim=1); ki2, ks2 = _knn(Zf, k_)
        W.data.copy_(best_W)
    _round(ki2, ks2)
    with torch.no_grad():
        SW = S @ best_W; Y = (X - mu_in) @ SW
    return SW.cpu().numpy(), mu_in.cpu().numpy(), Y.mean(0).cpu().numpy(), \
           torch.clamp(Y.std(0), min=1e-6).cpu().numpy()

def proj_ncwp(X, W, mu_in, mu_out, std_out):
    return _l2(((X - mu_in) @ W - mu_out) / std_out)

def compute_pca(X_np, shrink=0.08, eps=1e-6):
    X = torch.from_numpy(X_np).to(DEVICE).double()
    N, D = X.shape; mu = X.mean(0); Xc = X - mu
    Cov = (Xc.T @ Xc)/max(N-1,1); tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev,[0]); evec = torch.flip(evec,[1])
    return mu.cpu().float().numpy(), evec.cpu().float().numpy(), ev.cpu().float().numpy()


# ── Evaluation ────────────────────────────────────────────

def evaluate(c_embs, q_embs, corpus_ids, query_ids, qrels,
             k_values=(10,100), chunk=500):
    max_k = max(k_values)
    ndcg_at={k:[] for k in k_values}; recall_at={k:[] for k in k_values}; ap=[]
    for qi in range(0, len(query_ids), chunk):
        qe = min(len(query_ids), qi+chunk)
        sim = q_embs[qi:qe] @ c_embs.T
        if sim.shape[1] <= max_k:
            top_idx = np.argsort(-sim, axis=1)
        else:
            part = np.argpartition(-sim, max_k, axis=1)[:,:max_k]
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
                rs = {d for d,s in rel.items() if s>0}
                recall_at[k].append(len(set(top)&rs)/len(rs) if rs else 0.)
            tr = sum(1 for s in rel.values() if s>0); nr=0; a=0.
            for r,d in enumerate(ranked,1):
                if rel.get(d,0)>0: nr+=1; a+=nr/r
            ap.append(a/tr if tr>0 else 0.)
    m = {f"ndcg@{k}": np.mean(ndcg_at[k])*100 for k in k_values}
    m.update({f"recall@{k}": np.mean(recall_at[k])*100 for k in k_values})
    m["map"] = np.mean(ap)*100
    return m


# ── Main ─────────────────────────────────────────────────

def run_sts(dataset_name, model_key):
    mcfg = MODEL_CONFIGS[model_key]
    out_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "fit_corpus_sample")
    cache_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "embedding_cache", model_key)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)
    tag = model_key

    print(f"\n{'='*60}")
    print(f"  STS Retrieval: {dataset_name.upper()} / {model_key}")
    print(f"{'='*60}")

    # 1) Load dataset
    if dataset_name == "mrpc":
        corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test, fit_sentences = load_mrpc()
    else:
        corpus_ids, corpus_texts, test_qids, test_qtexts, qrels_test, fit_sentences = load_sick()

    base_dim = {"e5-base":768, "bge-base":768, "llama-8b":4096, "qwen-4b":2560}[model_key]

    # 2) Embed (with cache)
    emb = Embedder(mcfg)
    def _enc(texts, prefix, fname):
        p = os.path.join(cache_dir, fname)
        if os.path.exists(p):
            print(f"  [캐시] {fname}"); return np.load(p).astype(np.float32)
        print(f"  [인코딩] {fname} ({len(texts):,}개)...")
        X = emb.encode(texts, prefix=prefix).astype(np.float32)
        X = _l2(X); np.save(p, X); return X

    corpus_embs  = _enc(corpus_texts, mcfg.get("doc_prefix",""), "corpus_base.npy")
    query_embs   = _enc(test_qtexts,  mcfg.get("query_prefix",""), "test_queries_base.npy")
    fit_embs     = _enc(fit_sentences, mcfg.get("doc_prefix",""), "fit_corpus_sample.npy")

    # dims: half_range powers-of-two up to base_dim//2
    dims = []
    d = base_dim // 2
    while d >= 4: dims.append(d); d //= 2

    results = []
    ckpt_path = os.path.join(out_dir, f"{tag}_checkpoint.csv")
    if os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path).to_dict("records")
        results = ckpt
        done_set = {(r["method"], int(r["dim"])) for r in results}
    else:
        done_set = set()

    def _eval(ce, qe, method, dim):
        if (method, int(dim)) in done_set:
            return next(r for r in results if r["method"]==method and int(r["dim"])==int(dim))
        m = evaluate(ce, qe, corpus_ids, test_qids, qrels_test)
        row = {"dim":dim, "method":method, **m}
        print(f"  {method:<16} dim={dim:>5}  NDCG@10={m['ndcg@10']:.2f}  "
              f"Recall@100={m['recall@100']:.2f}  MAP={m['map']:.2f}")
        results.append(row); done_set.add((method, int(dim)))
        pd.DataFrame(results).to_csv(ckpt_path, index=False); return row

    # Base
    print("\n[Base]")
    _eval(corpus_embs, query_embs, "Base", base_dim)

    # PCA + 추가 baselines
    print("\nComputing PCA...")
    mu_pca, comps, scales = compute_pca(fit_embs, shrink=0.08)
    mu_rand = fit_embs.mean(0)
    print(f"\n[Standard methods × {len(dims)} dims]")
    for k in dims:
        print(f"\n  --- dim={k} ---")
        W_k = comps[:, :k]; s_k = scales[:k]
        # PCA-White
        c = _l2((corpus_embs - mu_pca) @ W_k / np.sqrt(s_k + 1e-8))
        q = _l2((query_embs  - mu_pca) @ W_k / np.sqrt(s_k + 1e-8))
        _eval(c, q, "PCA-White", k)
        # Soft-White
        sm = s_k.mean() if s_k.size > 0 else 1.0
        denom = np.sqrt((1.0 - 0.1)*s_k + 0.1*sm + 1e-8)
        c = _l2(((corpus_embs - mu_pca) @ W_k) / denom)
        q = _l2(((query_embs  - mu_pca) @ W_k) / denom)
        _eval(c, q, "Soft-White", k)
        # Random projection
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.normal(size=(fit_embs.shape[1], k)).astype(np.float32))
        W_rand = Q[:, :k]
        c = _l2((corpus_embs - mu_rand) @ W_rand)
        q = _l2((query_embs  - mu_rand) @ W_rand)
        _eval(c, q, "Random", k)
        # LPP
        from run_beir_experiment import fit_lpp_projector, proj_linear
        W_lpp, mu_lpp = fit_lpp_projector(fit_embs, k=k)
        c = _l2((corpus_embs - mu_lpp) @ W_lpp)
        q = _l2((query_embs  - mu_lpp) @ W_lpp)
        _eval(c, q, "LPP", k)

    # NCWP
    print(f"\n[NCWP × {len(dims)} dims]")
    wdir = os.path.join(out_dir, "weights", model_key, "NCWP_Default")
    os.makedirs(wdir, exist_ok=True)
    for k in dims:
        wp = os.path.join(wdir, f"dim_{k}.npz")
        if os.path.exists(wp):
            d = np.load(wp)
        else:
            print(f"  dim={k}: training...", end=" ", flush=True)
            W, mi, mo, so = train_ncwp(fit_embs, rank=k)
            np.savez(wp, W=W, mu_in=mi, mu_out=mo, std_out=so)
            print("done"); d = {"W":W,"mu_in":mi,"mu_out":mo,"std_out":so}
        c = proj_ncwp(corpus_embs, d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        q = proj_ncwp(query_embs,  d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        _eval(c, q, "NCWP", k)

    # Save + Plot
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"{tag}_results_main.csv")
    df.to_csv(csv_path, index=False)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    print(f"\n결과 저장: {csv_path}")
    cols = ["dim","method","ndcg@10","recall@100","map"]
    print(df[df.method.isin(["Base","PCA-White","NCWP"])].sort_values(["method","dim"])[cols].to_string(index=False))

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10,6))
    for method, grp in df[df.method.isin(["PCA-White","NCWP"])].groupby("method"):
        grp = grp.sort_values("dim")
        ax.plot(grp["dim"], grp["ndcg@10"], marker="o" if method=="NCWP" else "s",
                label=method, lw=2)
    base_val = df[df.method=="Base"]["ndcg@10"].values[0]
    ax.axhline(base_val, ls="--", color="black", label=f"Base ({base_val:.2f})")
    ax.set_xscale("log", base=2); ax.set_xlabel("Dim"); ax.set_ylabel("NDCG@10")
    ax.set_title(f"{dataset_name.upper()} / {model_key} — PCA-White vs NCWP")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4); plt.tight_layout()
    pdf_path = os.path.join(out_dir, f"{tag}_plots.pdf")
    plt.savefig(pdf_path, bbox_inches="tight"); plt.close()
    print(f"플롯 저장: {pdf_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["mrpc","sick"])
    p.add_argument("--model",   required=True, choices=list(MODEL_CONFIGS.keys()))
    args = p.parse_args()
    run_sts(args.dataset, args.model)
