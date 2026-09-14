"""
run_ablation_ext.py  — 필수 1 + 필수 2a
Part 1: ZCA-only at dim=[24,48,96,192] (E5-base / Quora)
Part 2: No-QR ablation on BGE-base / Quora

ZCA-only 저차원 구현:
  ZCA whitening → top-k PCA on whitened space → L2 normalize
  (whitened 공간의 PCA = 원래 공간 ZCA + PCA, contrastive 없음)

결과 저장:
  /workspace/NCWP/quora_results/ablation/ablation_ext_results.csv
"""

import os, math, numpy as np, torch, torch.nn.functional as F
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 42
OUT_DIR  = "/workspace/NCWP/quora_results/ablation"
TARGET_DIMS = [24, 48, 96, 192]
EVAL_CHUNK  = 500

os.makedirs(OUT_DIR, exist_ok=True)

import random
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ── Quora qrels 로드 ──────────────────────────────────────
from datasets import load_dataset

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

# ── Helpers ───────────────────────────────────────────────
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

def train_ncwp(X_np, rank, retraction_interval=10, seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    N, D = X_np.shape; r = min(rank, D)
    X = torch.from_numpy(X_np).to(DEVICE)
    mu_in, S = _zca_shrink(X); Xw = F.normalize((X - mu_in) @ S, dim=1)
    k = max(1, min(10, N-1)); knn_idx, knn_sim = _knn(Xw, k)
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE)/math.sqrt(D))
    lr, lr_min = 8e-3, 1.6e-3; temp, temp_min = 0.12, 0.06
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    Bp = max(64, min(1024, N//64))
    steps = min(160, max(100, math.ceil(N/(2*Bp))))
    warmup=100; patience=6; best_W=None; best_loss=float("inf"); no_imp=0; gs=0
    tau_t = torch.tensor(0.0, dtype=torch.float)

    def _round(ki, ks):
        nonlocal gs, best_W, best_loss, no_imp
        T = 20 * steps
        for ep in range(1, 21):
            ep_loss = 0.
            for _ in range(steps):
                t = max(0., gs-warmup)/max(1,T-warmup)
                cur_lr = (lr_min+(lr-lr_min)*(gs/max(1,warmup))) if gs<warmup \
                          else lr_min+0.5*(lr-lr_min)*(1+math.cos(math.pi*t))
                for pg in opt.param_groups: pg["lr"] = cur_lr
                t2 = gs/max(1,T-1)
                cur_temp = temp_min+0.5*(temp-temp_min)*(1+math.cos(math.pi*t2))
                with torch.no_grad():
                    ac = torch.randint(0,N,(Bp,),dtype=torch.long)
                    bl = []
                    for anc in ac.tolist():
                        vm = ks[anc]>=tau_t; vn = ki[anc][vm]
                        if len(vn)==0: vn = ki[anc]
                        bl.append(vn[torch.randint(0,len(vn),(1,))].item())
                    bc = torch.tensor(bl, dtype=torch.long)
                    Xb = torch.cat([Xw[ac.to(DEVICE)], Xw[bc.to(DEVICE)]], 0)
                Z = F.normalize(Xb @ W, dim=1); sim_m = (Z @ Z.T)/cur_temp
                B2=Z.shape[0]; B=B2//2
                sim_m=sim_m.masked_fill(torch.eye(B2,device=DEVICE,dtype=torch.bool),-1e9)
                i1=torch.arange(B,device=DEVICE); i2=i1+B
                loss=(0.5*(_contrastive_loss(sim_m[i1],i2)+_contrastive_loss(sim_m[i2],i1))
                      +0.05*_cov_penalty(Z)+0.02*_orth_penalty(W))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if retraction_interval>0 and (gs+1)%retraction_interval==0:
                    with torch.no_grad():
                        Q,_=torch.linalg.qr(W.data); W.data=Q[:,:r]
                ep_loss+=float(loss.item()); gs+=1
            ep_loss/=max(1,steps)
            if ep_loss+1e-6<best_loss: best_loss=ep_loss; no_imp=0; best_W=W.detach().clone()
            else:
                no_imp+=1
                if no_imp>=patience: break

    _round(knn_idx, knn_sim)
    with torch.no_grad():
        Zf=F.normalize(Xw @ best_W,dim=1); ki2,ks2=_knn(Zf,k); W.data.copy_(best_W)
    _round(ki2, ks2)
    with torch.no_grad():
        SW=S @ best_W; Y=(X-mu_in)@SW
    return SW.cpu().numpy(), mu_in.cpu().numpy(), Y.mean(0).cpu().numpy(), \
           torch.clamp(Y.std(0),min=1e-6).cpu().numpy()

def proj_ncwp(X, W, mu_in, mu_out, std_out):
    return _l2(((X-mu_in)@W-mu_out)/std_out)

# ── ZCA-only at dim=k ─────────────────────────────────────
@torch.no_grad()
def zca_then_pca(corpus_np, query_np, X_fit_np, dim):
    """
    ZCA whitening + PCA(k) — contrastive 학습 없음
    ZCA로 등방화 후 상위 k 주성분 투영
    (등방 공간에서 PCA = 원래 공간의 ZCA → 이후 k-dim 투영)
    """
    X = torch.from_numpy(X_fit_np).to(DEVICE)
    mu, S = _zca_shrink(X)
    Xw = F.normalize((X - mu) @ S, dim=1)
    # PCA on whitened space
    Xwc = Xw - Xw.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xwc, full_matrices=False)
    W_pca = Vt[:dim].T.cpu().numpy()  # (D, dim)
    S_np  = S.cpu().numpy(); mu_np = mu.cpu().numpy()
    def _transform(X_np):
        Xw_np = F.normalize(
            torch.from_numpy((X_np - mu_np) @ S_np).to(DEVICE), dim=1
        ).cpu().numpy()
        return _l2(Xw_np @ W_pca)
    return _transform(corpus_np), _transform(query_np)

# ── Evaluation ────────────────────────────────────────────
def evaluate(c_embs, q_embs, k_values=(10,100)):
    max_k = max(k_values)
    ndcg_at={k:[] for k in k_values}; recall_at={k:[] for k in k_values}; ap=[]
    for qi in range(0, len(test_qids), EVAL_CHUNK):
        qe = min(len(test_qids), qi+EVAL_CHUNK)
        sim = q_embs[qi:qe] @ c_embs.T
        if sim.shape[1] <= max_k: top_idx = np.argsort(-sim, axis=1)
        else:
            part=np.argpartition(-sim,max_k,axis=1)[:,:max_k]
            order=np.argsort(-sim[np.arange(len(part))[:,None],part],axis=1)
            top_idx=part[np.arange(len(part))[:,None],order]
        for ci,q in enumerate(range(qi,qe)):
            qid=test_qids[q]
            if qid not in qrels_test: continue
            rel=qrels_test[qid]; ranked=[corpus_ids[i] for i in top_idx[ci]]
            ideal=sorted(rel.values(),reverse=True)
            for k in k_values:
                top=ranked[:k]
                dcg=sum((2**rel.get(d,0)-1)/math.log2(r+2) for r,d in enumerate(top))
                idcg=sum((2**rv-1)/math.log2(r+2) for r,rv in enumerate(ideal[:k]))
                ndcg_at[k].append(dcg/idcg if idcg>0 else 0.)
                rs={d for d,s in rel.items() if s>0}
                recall_at[k].append(len(set(top)&rs)/len(rs) if rs else 0.)
            tr=sum(1 for s in rel.values() if s>0); nr=0; a=0.
            for r,d in enumerate(ranked,1):
                if rel.get(d,0)>0: nr+=1; a+=nr/r
            ap.append(a/tr if tr>0 else 0.)
    m={f"ndcg@{k}":np.mean(ndcg_at[k])*100 for k in k_values}
    m.update({f"recall@{k}":np.mean(recall_at[k])*100 for k in k_values})
    m["map"]=np.mean(ap)*100; return m

# ── Main ──────────────────────────────────────────────────
records = []

def run_model_ext(model_key, label):
    cache_dir = f"/workspace/NCWP/quora_results/embedding_cache/{model_key}"
    corpus_embs = np.load(f"{cache_dir}/corpus_base.npy").astype(np.float32)
    query_embs  = np.load(f"{cache_dir}/test_queries_base.npy").astype(np.float32)
    X_fit       = np.load(f"{cache_dir}/fit_corpus_sample_N20000.npy").astype(np.float32)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  corpus={corpus_embs.shape}  fit={X_fit.shape}")
    print(f"{'='*60}")

    # ── Part A: ZCA-only @ target dims ───────────────────
    print(f"\n[ZCA-only @ dims={TARGET_DIMS}]")
    for dim in TARGET_DIMS:
        c, q = zca_then_pca(corpus_embs, query_embs, X_fit, dim)
        m = evaluate(c, q)
        print(f"  ZCA-only  dim={dim:>4}  NDCG@10={m['ndcg@10']:.2f}  "
              f"Recall@100={m['recall@100']:.2f}")
        records.append({"model":model_key,"variant":"ZCA-only","dim":dim,**m})

    # ── Part B: No-QR @ target dims ──────────────────────
    print(f"\n[No-QR @ dims={TARGET_DIMS}]")
    wdir = os.path.join(OUT_DIR, "weights_ext", model_key, "No-QR")
    os.makedirs(wdir, exist_ok=True)
    for dim in TARGET_DIMS:
        wp = os.path.join(wdir, f"dim_{dim}.npz")
        if os.path.exists(wp):
            d = np.load(wp); print(f"  dim={dim:>4}: [캐시]", end=" ")
        else:
            print(f"  dim={dim:>4}: training...", end=" ", flush=True)
            W, mi, mo, so = train_ncwp(X_fit, dim, retraction_interval=0)
            np.savez(wp, W=W, mu_in=mi, mu_out=mo, std_out=so)
            d = {"W":W,"mu_in":mi,"mu_out":mo,"std_out":so}; print("done", end=" ")
        c = proj_ncwp(corpus_embs, d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        q = proj_ncwp(query_embs,  d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        m = evaluate(c, q)
        print(f"NDCG@10={m['ndcg@10']:.2f}")
        records.append({"model":model_key,"variant":"No-QR","dim":dim,**m})

    # ── Part C: Full-NCWP @ target dims ──────────────────
    print(f"\n[Full-NCWP @ dims={TARGET_DIMS}]")
    wdir2 = os.path.join(OUT_DIR, "weights_ext", model_key, "Full-NCWP")
    os.makedirs(wdir2, exist_ok=True)
    for dim in TARGET_DIMS:
        wp = os.path.join(wdir2, f"dim_{dim}.npz")
        if os.path.exists(wp):
            d = np.load(wp); print(f"  dim={dim:>4}: [캐시]", end=" ")
        else:
            print(f"  dim={dim:>4}: training...", end=" ", flush=True)
            W, mi, mo, so = train_ncwp(X_fit, dim, retraction_interval=10)
            np.savez(wp, W=W, mu_in=mi, mu_out=mo, std_out=so)
            d = {"W":W,"mu_in":mi,"mu_out":mo,"std_out":so}; print("done", end=" ")
        c = proj_ncwp(corpus_embs, d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        q = proj_ncwp(query_embs,  d["W"], d["mu_in"], d["mu_out"], d["std_out"])
        m = evaluate(c, q)
        print(f"NDCG@10={m['ndcg@10']:.2f}")
        records.append({"model":model_key,"variant":"Full-NCWP","dim":dim,**m})

# ── E5-base ───────────────────────────────────────────────
run_model_ext("e5-base", "E5-base / Quora / corpus_sample")

# ── BGE-base (필수 2) ─────────────────────────────────────
run_model_ext("bge-base", "BGE-base / Quora / corpus_sample")

# ── Save ─────────────────────────────────────────────────
df = pd.DataFrame(records)
csv_path = os.path.join(OUT_DIR, "ablation_ext_results.csv")
df.to_csv(csv_path, index=False)
print(f"\n결과 저장: {csv_path}")

# ── Summary ──────────────────────────────────────────────
print("\n" + "="*70)
print("  EXTENDED ABLATION SUMMARY (NDCG@10)")
print("="*70)
for model in ["e5-base","bge-base"]:
    sub = df[df.model==model]
    print(f"\n  [{model}]")
    pivot = sub.pivot_table(index="variant", columns="dim", values="ndcg@10")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

# ── Plot ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14,6))
colors = {"ZCA-only":"steelblue","No-QR":"darkorange","Full-NCWP":"green","PCA-White":"black"}
marks  = {"ZCA-only":"D","No-QR":"^","Full-NCWP":"o","PCA-White":"s"}

# PCA-White 기존 결과 로드
prev_df = pd.read_csv("/workspace/NCWP/quora_results/ablation/ablation_results.csv")
pca_e5  = prev_df[prev_df.variant=="PCA-White"][["dim","ndcg@10"]].copy()
pca_e5["model"] = "e5-base"

for ax, model in zip(axes, ["e5-base","bge-base"]):
    sub = df[df.model==model]
    for var in ["ZCA-only","No-QR","Full-NCWP"]:
        d = sub[sub.variant==var].sort_values("dim")
        ax.plot(d["dim"], d["ndcg@10"], marker=marks[var], color=colors[var],
                label=var, lw=2)
    if model == "e5-base":
        ax.plot(pca_e5["dim"], pca_e5["ndcg@10"], marker=marks["PCA-White"],
                color=colors["PCA-White"], label="PCA-White", lw=2, ls="--")
    ax.set_xscale("log", base=2); ax.set_xticks(TARGET_DIMS)
    ax.set_xticklabels(TARGET_DIMS); ax.set_xlabel("Dim")
    ax.set_ylabel("NDCG@10"); ax.set_title(f"{model} / Quora")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)

plt.suptitle("ZCA-only vs No-QR vs Full-NCWP\n(Quora, corpus_sample)", fontsize=13)
plt.tight_layout()
pdf_path = os.path.join(OUT_DIR, "ablation_ext_plot.pdf")
plt.savefig(pdf_path, bbox_inches="tight"); plt.close()
print(f"플롯 저장: {pdf_path}")
