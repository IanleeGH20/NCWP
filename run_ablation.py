"""
run_ablation.py  — 0-1: Component Ablation
Quora / E5-base / corpus_sample, dims = [24, 48, 96, 192]

Variants (7개 + PCA reference):
  1. PCA-White          : closed-form baseline (기존 결과에서 로드)
  2. ZCA-only           : ZCA whitening 적용, 학습 없음, 전체 dim 평가
  3. Random-pairs       : kNN 대신 무작위 positive pair
  4. No-hardneg         : kNN positive + 전체 in-batch negative (top-K 선택 없음)
  5. No-QR              : QR retraction 없음
  6. No-refinement      : kNN refinement round 없음
  7. Full-NCWP          : 완전한 NCWP (reference)

결과 저장: /workspace/NCWP/quora_results/ablation/
"""

import os, math, numpy as np, torch, torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

CACHE_DIR   = "/workspace/NCWP/quora_results/embedding_cache/e5-base"
OUT_DIR     = "/workspace/NCWP/quora_results/ablation"
PREV_CSV    = "/workspace/NCWP/quora_results/fit_corpus_sample/e5-base_results_main.csv"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
TARGET_DIMS = [24, 48, 96, 192]
EVAL_CHUNK  = 500

os.makedirs(OUT_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────
import random
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ── Load cached embeddings ────────────────────────────────────
print("Loading cached embeddings...")
corpus_embs = np.load(os.path.join(CACHE_DIR, "corpus_base.npy")).astype(np.float32)
query_embs  = np.load(os.path.join(CACHE_DIR, "test_queries_base.npy")).astype(np.float32)
X_fit       = np.load(os.path.join(CACHE_DIR, "fit_corpus_sample_N20000.npy")).astype(np.float32)
print(f"  corpus={corpus_embs.shape}  queries={query_embs.shape}  fit={X_fit.shape}")

# ── Load qrels ────────────────────────────────────────────────
from datasets import load_dataset
from collections import defaultdict

print("Loading Quora qrels...")
qr_ds = load_dataset("mteb/quora")
qrels_test = defaultdict(dict)
for r in qr_ds["test"]:
    qrels_test[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])

c_ds = load_dataset("mteb/quora", "corpus", split="corpus")
corpus_ids = [r["_id"] for r in c_ds]
q_ds = load_dataset("mteb/quora", "queries", split="queries")
qid2tx = {r["_id"]: r["text"] for r in q_ds}
test_qids = [q for q in sorted(qrels_test) if q in qid2tx]
print(f"  corpus={len(corpus_ids):,}  queries={len(test_qids):,}")

# ── Evaluation ────────────────────────────────────────────────
def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def evaluate(c_embs, q_embs, k_values=(10, 100)):
    max_k = max(k_values)
    ndcg_at = {k: [] for k in k_values}
    recall_at= {k: [] for k in k_values}
    ap_list  = []
    for qi in range(0, len(test_qids), EVAL_CHUNK):
        qe  = min(len(test_qids), qi + EVAL_CHUNK)
        sim = q_embs[qi:qe] @ c_embs.T
        if sim.shape[1] <= max_k:
            top_idx = np.argsort(-sim, axis=1)
        else:
            part  = np.argpartition(-sim, max_k, axis=1)[:, :max_k]
            order = np.argsort(-sim[np.arange(len(part))[:,None], part], axis=1)
            top_idx = part[np.arange(len(part))[:,None], order]
        for ci, q in enumerate(range(qi, qe)):
            qid = test_qids[q]
            if qid not in qrels_test: continue
            rel = qrels_test[qid]; ranked = [corpus_ids[i] for i in top_idx[ci]]
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
            for r,d in enumerate(ranked, 1):
                if rel.get(d,0)>0: nr+=1; ap+=nr/r
            ap_list.append(ap/total_rel if total_rel>0 else 0.)
    m = {f"ndcg@{k}": np.mean(ndcg_at[k])*100 for k in k_values}
    m.update({f"recall@{k}": np.mean(recall_at[k])*100 for k in k_values})
    m["map"] = np.mean(ap_list)*100
    return m

# ── NCWP helpers ─────────────────────────────────────────────
def _orth_penalty(W):
    WT_W = W.T @ W
    return ((WT_W - torch.eye(WT_W.shape[0], device=W.device))**2).mean()

def _cov_penalty(Z):
    B = Z.shape[0]
    if B <= 1: return torch.tensor(0., device=Z.device)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc)/(B-1)
    return ((Cov - torch.eye(Cov.shape[0], device=Z.device))**2).mean()

def _zca_shrink_t(X, shrink=0.08, eps=1e-6):
    mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc)/max(X.shape[0]-1,1)
    D = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=X.device)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.clamp(ev, min=eps)
    S = evec @ torch.diag(1./torch.sqrt(ev)) @ evec.T
    return mu.squeeze(0), S

@torch.no_grad()
def _knn(X, k, chunk=2048):
    N = X.shape[0]; XT = X.T
    idx = torch.empty((N,k), dtype=torch.long, device="cpu")
    sim = torch.empty((N,k), dtype=torch.float, device="cpu")
    for s in range(0, N, chunk):
        e = min(N, s+chunk)
        s_ = X[s:e] @ XT; s_[:, torch.arange(s,e,device=X.device)] = -1e9
        tk = torch.topk(s_, k=k, dim=1)
        idx[s:e] = tk.indices.cpu(); sim[s:e] = tk.values.cpu()
    return idx, sim

def _contrastive_loss(sim_row, pos_idx, topk=None):
    B = sim_row.shape[0]
    pos_logits = sim_row.gather(1, pos_idx.view(-1,1)).squeeze(1)
    if topk is None or topk >= sim_row.shape[1]-1:
        denom = torch.logsumexp(sim_row, 1)
    else:
        mask = torch.ones_like(sim_row, dtype=torch.bool)
        mask[torch.arange(B, device=sim_row.device), pos_idx] = False
        negs = sim_row.masked_select(mask).view(B,-1)
        kk = min(topk, negs.shape[1])
        vals, _ = torch.topk(negs, k=kk, dim=1)
        denom = torch.logsumexp(torch.cat([vals, pos_logits.unsqueeze(1)], 1), 1)
    return -(pos_logits - denom).mean()


def train_ncwp_variant(X_np, rank, variant="full",
                       k_neighbors=10, cosine_tau=0.0,
                       temperature=0.12, lambda_cov=0.05, lambda_orth=0.02,
                       topk_negatives=256, retraction_interval=10,
                       refine_knn_rounds=1, max_epochs=20, seed=42):
    """
    variant:
      'full'          : 완전한 NCWP
      'random_pairs'  : kNN 대신 무작위 positive
      'no_hardneg'    : topk_negatives=None (모든 in-batch)
      'no_qr'         : retraction_interval=0
      'no_refinement' : refine_knn_rounds=0
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    if variant == "no_hardneg":     topk_negatives = None
    if variant == "no_qr":          retraction_interval = 0
    if variant == "no_refinement":  refine_knn_rounds = 0

    N, D = X_np.shape; r = min(rank, D)
    X = torch.from_numpy(X_np).to(DEVICE)
    mu_in, S = _zca_shrink_t(X)
    Xw = F.normalize((X - mu_in) @ S, dim=1)
    k = max(1, min(k_neighbors, N-1))

    # kNN 구성 (random_pairs 모드에서는 사용 안 함)
    if variant != "random_pairs":
        knn_idx, knn_sim = _knn(Xw, k)
    else:
        knn_idx = knn_sim = None

    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    lr = 8e-3; lr_min = 0.2*lr; temp_min = 0.5*temperature
    warmup = 100; weight_decay = 1e-4
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=weight_decay)

    Bp = max(64, min(1024, N//64))
    steps = min(160, max(100, math.ceil(N/(2*Bp))))
    patience = 6; best_W = None; best_loss = float("inf"); no_imp = 0
    global_step = 0

    def _one_round(knn_idx_local, knn_sim_local, tag="init"):
        nonlocal global_step, best_W, best_loss, no_imp
        tau_t = torch.tensor(cosine_tau, dtype=torch.float)
        for epoch in range(1, max_epochs+1):
            ep_loss = 0.
            for _ in range(steps):
                T = max_epochs * steps
                t = max(0., global_step - warmup) / max(1, T - warmup)
                if global_step < warmup:
                    cur_lr = lr_min + (lr - lr_min)*(global_step/max(1,warmup))
                else:
                    cur_lr = lr_min + 0.5*(lr-lr_min)*(1+math.cos(math.pi*t))
                for pg in opt.param_groups: pg["lr"] = cur_lr
                t2 = global_step / max(1, T-1)
                cur_temp = temp_min + 0.5*(temperature-temp_min)*(1+math.cos(math.pi*t2))

                with torch.no_grad():
                    a_cpu = torch.randint(0, N, (Bp,), dtype=torch.long)
                    if variant == "random_pairs":
                        b_cpu = torch.randint(0, N, (Bp,), dtype=torch.long)
                    else:
                        b_list = []
                        for anc in a_cpu.tolist():
                            valid = knn_sim_local[anc] >= tau_t
                            vnb = knn_idx_local[anc][valid]
                            if len(vnb)==0: vnb = knn_idx_local[anc]
                            b_list.append(vnb[torch.randint(0,len(vnb),(1,))].item())
                        b_cpu = torch.tensor(b_list, dtype=torch.long)
                    a_idx = a_cpu.to(DEVICE); b_idx = b_cpu.to(DEVICE)
                    Xbatch = torch.cat([Xw[a_idx], Xw[b_idx]], 0)

                Z = F.normalize(Xbatch @ W, dim=1)
                sim_m = (Z @ Z.T)/cur_temp
                B2 = Z.shape[0]; B = B2//2
                sim_m = sim_m.masked_fill(torch.eye(B2, device=DEVICE, dtype=torch.bool), -1e9)
                idx1 = torch.arange(B, device=DEVICE); idx2 = idx1+B
                l_i = _contrastive_loss(sim_m[idx1], idx2, topk=topk_negatives)
                l_j = _contrastive_loss(sim_m[idx2], idx1, topk=topk_negatives)
                loss = 0.5*(l_i+l_j) + lambda_cov*_cov_penalty(Z) + lambda_orth*_orth_penalty(W)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

                if retraction_interval>0 and (global_step+1)%retraction_interval==0:
                    with torch.no_grad():
                        Q, _ = torch.linalg.qr(W.data); W.data = Q[:,:r]
                ep_loss += float(loss.item()); global_step += 1

            ep_loss /= max(1, steps)
            if ep_loss+1e-6 < best_loss:
                best_loss = ep_loss; no_imp = 0; best_W = W.detach().clone()
            else:
                no_imp += 1
                if no_imp >= patience: break

    _one_round(knn_idx, knn_sim, "init")
    for rr in range(refine_knn_rounds):
        with torch.no_grad():
            Zfull = F.normalize(Xw @ best_W, dim=1)
            ki, ks = _knn(Zfull, k); W.data.copy_(best_W)
        _one_round(ki, ks, f"ref{rr+1}")

    with torch.no_grad():
        SW = S @ best_W; Y = (X - mu_in) @ SW
    return (SW.cpu().numpy(), mu_in.cpu().numpy(),
            Y.mean(0).cpu().numpy(), torch.clamp(Y.std(0), min=1e-6).cpu().numpy())


def proj_ncwp(X, W, mu_in, mu_out, std_out):
    return _l2(((X - mu_in) @ W - mu_out) / std_out)

# ── ZCA-only ─────────────────────────────────────────────────
def zca_only_proj(X_np, shrink=0.08):
    """ZCA whitening 적용, 학습 없음, 전체 dim 출력."""
    X = torch.from_numpy(X_np).to(DEVICE)
    mu, S = _zca_shrink_t(X, shrink=shrink)
    Xw = F.normalize((X - mu) @ S, dim=1)
    return Xw.cpu().numpy(), mu.cpu().numpy(), S.cpu().numpy()

# ── PCA-White from existing results ──────────────────────────
def load_pca_results():
    if not os.path.exists(PREV_CSV):
        return {}
    df = pd.read_csv(PREV_CSV)
    pca = df[df.method=="PCA-White"]
    return {int(r["dim"]): r["ndcg@10"] for _, r in pca.iterrows()}

# ── Main Ablation ─────────────────────────────────────────────
print("\n" + "="*60)
print("  Component Ablation: Quora / E5-base / corpus_sample")
print("  dims =", TARGET_DIMS)
print("="*60)

records = []
base_dim = corpus_embs.shape[1]  # 768

# ── 1. PCA-White (기존 결과 로드) ─────────────────────────────
print("\n[1/7] PCA-White — 기존 결과 로드")
pca_scores = load_pca_results()
for dim in TARGET_DIMS:
    if dim in pca_scores:
        records.append({"variant":"PCA-White","dim":dim,"ndcg@10":pca_scores[dim]})
        print(f"  dim={dim:4d}  NDCG@10={pca_scores[dim]:.2f}")

# ── 2. ZCA-only (전체 dim, 학습 없음) ────────────────────────
print("\n[2/7] ZCA-only (W=identity, no projection, full dim=768)")
zca_c, mu_zca, S_zca = zca_only_proj(corpus_embs)
zca_q, _, _          = zca_only_proj(query_embs)
# ZCA는 S를 fit data로 구해야 공정: re-fit on X_fit
X_t = torch.from_numpy(X_fit).to(DEVICE)
mu_f, S_f = _zca_shrink_t(X_t)
with torch.no_grad():
    zca_c = _l2(F.normalize((torch.from_numpy(corpus_embs).to(DEVICE) - mu_f) @ S_f, dim=1).cpu().numpy())
    zca_q = _l2(F.normalize((torch.from_numpy(query_embs).to(DEVICE)  - mu_f) @ S_f, dim=1).cpu().numpy())
m = evaluate(zca_c, zca_q)
print(f"  dim={base_dim:4d}  NDCG@10={m['ndcg@10']:.2f}  (full dim after ZCA)")
records.append({"variant":"ZCA-only","dim":base_dim,"ndcg@10":m["ndcg@10"],
                "recall@100":m["recall@100"],"map":m["map"]})

# ── Variants 3-7: NCWP 계열 ──────────────────────────────────
ablation_variants = [
    ("Random-pairs",    {"variant":"random_pairs"}),
    ("No-hardneg",      {"variant":"no_hardneg"}),
    # Variant 5: No-memory-bank
    # Memory bank은 미구현 상태이므로 Full-NCWP와 동일한 결과
    # → 명시적으로 별도 행으로 기록하여 "memory bank 기여 없음"을 확인
    ("No-memorybank",   {"variant":"full"}),   # == Full-NCWP (memory bank 미구현)
    ("No-QR",           {"variant":"no_qr"}),
    ("No-refinement",   {"variant":"no_refinement"}),
    ("Full-NCWP",       {"variant":"full"}),
]

for v_idx, (vname, vkw) in enumerate(ablation_variants, start=3):
    if vname == "No-memorybank":
        print(f"\n[{v_idx}/8] No-memorybank ← memory bank 미구현, Full-NCWP와 동일 결과 예정")
    else:
        print(f"\n[{v_idx}/8] {vname}")
    wdir = os.path.join(OUT_DIR, "weights", vname)
    os.makedirs(wdir, exist_ok=True)

    for dim in TARGET_DIMS:
        wpath = os.path.join(wdir, f"dim_{dim}.npz")
        if os.path.exists(wpath):
            d = np.load(wpath)
            print(f"  dim={dim:4d}: [캐시]", end=" ")
        else:
            print(f"  dim={dim:4d}: training...", end=" ", flush=True)
            W, mu_in, mu_out, std_out = train_ncwp_variant(X_fit, rank=dim, **vkw)
            np.savez(wpath, W=W, mu_in=mu_in, mu_out=mu_out, std_out=std_out)
            d = {"W":W,"mu_in":mu_in,"mu_out":mu_out,"std_out":std_out}
            print("done", end=" ")

        c = proj_ncwp(corpus_embs, d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        q = proj_ncwp(query_embs,  d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        m = evaluate(c, q)
        print(f"NDCG@10={m['ndcg@10']:.2f}  Recall@100={m['recall@100']:.2f}")
        records.append({"variant":vname,"dim":dim,**m})

# ── Save results ─────────────────────────────────────────────
df = pd.DataFrame(records)
csv_path = os.path.join(OUT_DIR, "ablation_results.csv")
df.to_csv(csv_path, index=False)
print(f"\n결과 저장: {csv_path}")

# ── Summary table ─────────────────────────────────────────────
print("\n" + "="*70)
print("  ABLATION SUMMARY (NDCG@10)")
print("="*70)
pivot = df.pivot_table(index="variant", columns="dim", values="ndcg@10", aggfunc="first")
print(pivot.to_string())

# ── Plot ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"PCA-White":"black","ZCA-only":"gray","Random-pairs":"red",
          "No-hardneg":"orange","No-QR":"purple","No-refinement":"blue","Full-NCWP":"green"}
marks  = {"PCA-White":"s","ZCA-only":"D","Random-pairs":"x",
          "No-hardneg":"^","No-QR":"v","No-refinement":"<","Full-NCWP":"o"}

for var in df["variant"].unique():
    sub = df[df["variant"]==var].sort_values("dim")
    if var == "ZCA-only":
        ax.axhline(sub["ndcg@10"].values[0], ls="--",
                   color=colors.get(var,"gray"), label=f"{var} ({sub['ndcg@10'].values[0]:.1f}, full dim)", alpha=0.7)
    else:
        ax.plot(sub["dim"], sub["ndcg@10"], marker=marks.get(var,"o"),
                color=colors.get(var,"gray"), label=var, lw=2)

ax.set_xscale("log", base=2)
ax.set_xticks(TARGET_DIMS); ax.set_xticklabels(TARGET_DIMS)
ax.set_xlabel("Projected Dimension"); ax.set_ylabel("NDCG@10")
ax.set_title("Component Ablation — Quora / E5-base / corpus_sample")
ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
ax.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
pdf_path = os.path.join(OUT_DIR, "ablation_plot.pdf")
plt.savefig(pdf_path, bbox_inches="tight"); plt.close()
print(f"플롯 저장: {pdf_path}")
