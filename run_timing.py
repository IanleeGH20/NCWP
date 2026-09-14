"""
run_timing.py  ──  Offline Projection Training Time Comparison

리뷰어 질문 대응:
  "Please provide the concrete offline training time required to learn the NCWP
   projection matrix compared to simply computing a standard PCA-Whitening matrix."

실험 설계:
  - 캐시된 임베딩 재사용 (LLM 인코딩 시간 제외 → 프로젝션 학습만 측정)
  - 두 가지 fit 데이터 크기: N=1000, N=3000
  - 두 가지 모델 (임베딩 차원): qwen-4b (D=2560), llama-8b (D=4096)
  - 5회 반복 측정 → mean ± std
  - 비교 방법: PCA-Whitening, Soft-Whitening, NCWP

결과 저장:
  /workspace/NCWP/quora_results/timing/
    timing_results.csv
    timing_plot.png

실행:
  python run_timing.py
"""

import os, math, time, platform
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess
from matplotlib.backends.backend_pdf import PdfPages

# ── Config ────────────────────────────────────────────────
CACHE_ROOT = "/workspace/NCWP/quora_results/embedding_cache"
OUT_ROOT   = "/workspace/NCWP/quora_results/timing"
os.makedirs(OUT_ROOT, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEED      = 42
N_RUNS    = 5       # 반복 측정 횟수
TARGET_DIMS = [32, 64]

# 측정 대상 (모델명, fit_npy 경로)
MODEL_CONFIGS = {
    "qwen-4b":  {
        "N1000": os.path.join(CACHE_ROOT, "qwen-4b",  "fit_query_sample_N1000.npy"),
        "N3000": os.path.join(CACHE_ROOT, "qwen-4b",  "fit_query_sample_N3000.npy"),
    },
    "llama-8b": {
        "N3000": os.path.join(CACHE_ROOT, "llama-8b", "fit_query_sample_N3000.npy"),
    },
}


# ── Reproducibility ──────────────────────────────────────
def _set_seed(s=SEED):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ── Environment Info ──────────────────────────────────────
def print_env():
    print("=" * 60)
    print("  실험 환경")
    print("=" * 60)
    print(f"  OS       : {platform.system()} {platform.release()}")
    print(f"  Python   : {platform.python_version()}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Device   : {DEVICE}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  GPU [{i}]  : {props.name}  ({mem_gb:.1f} GB)")
        print(f"  GPU 수   : {n}개")
    try:
        r = subprocess.run(["nproc"], capture_output=True, text=True)
        print(f"  CPU 코어 : {r.stdout.strip()}개")
    except Exception:
        pass
    print("=" * 60)


# ── NCWP helpers ─────────────────────────────────────────
def _zca_shrink(X, shrink=0.08, eps=1e-6):
    orig = X.dtype; X = X.double()
    mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(X.shape[0]-1, 1)
    D_ = Cov.shape[0]; tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D_)*torch.eye(D_, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    S = evec @ torch.diag(1./torch.sqrt(torch.clamp(ev, min=eps))) @ evec.T
    return mu.squeeze(0).to(orig), S.to(orig)

@torch.no_grad()
def _topk_knn(X, k, chunk=2048):
    N = X.shape[0]; idx = torch.empty((N, k), dtype=torch.long, device="cpu")
    for s in range(0, N, chunk):
        e = min(N, s+chunk); sim = X[s:e] @ X.T
        sim[:, torch.arange(s, e, device=X.device)] = -1e9
        idx[s:e] = torch.topk(sim, k=k, dim=1).indices.cpu()
    return idx

def _orth_penalty(W):
    WT_W = W.T @ W
    return ((WT_W - torch.eye(WT_W.shape[0], device=W.device))**2).mean()

def _cov_penalty(Z):
    B = Z.shape[0]
    if B <= 1: return torch.tensor(0., device=Z.device)
    Zc = Z - Z.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / (B - 1)
    return ((Cov - torch.eye(Cov.shape[0], device=Z.device))**2).mean()


# ── Projection Methods ────────────────────────────────────
def run_pca_white(X_np, target_dim, shrink=0.08, eps=1e-6):
    """PCA-Whitening: 공분산 행렬 → 고유분해 → W"""
    X = torch.from_numpy(X_np).to(DEVICE).double()
    N, D = X.shape
    mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(N-1, 1); tr = torch.trace(Cov)
    Cs = (1-shrink)*Cov + shrink*(tr/D)*torch.eye(D, device=X.device, dtype=X.dtype)
    ev, evec = torch.linalg.eigh(Cs)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(target_dim, D)
    W = evec[:, :k] @ torch.diag(1./torch.sqrt(torch.clamp(ev[:k], min=eps)))
    if DEVICE == "cuda": torch.cuda.synchronize()
    return mu.squeeze(0).cpu().float().numpy(), W.cpu().float().numpy()


def run_soft_white(X_np, target_dim, shrink=0.08, eps=1e-6):
    """Soft-Whitening: PCA 방향만 취하고 whitening 완화"""
    X = torch.from_numpy(X_np).to(DEVICE).double()
    N, D = X.shape
    mu = X.mean(0, keepdim=True); Xc = X - mu
    Cov = (Xc.T @ Xc) / max(N-1, 1)
    # 정규화 없이 top-k 고유벡터만 추출
    ev, evec = torch.linalg.eigh(Cov)
    ev = torch.flip(ev, [0]); evec = torch.flip(evec, [1])
    k = min(target_dim, D)
    # soft: eigenvalue를 1/sqrt(ev+eps) 대신 1/(ev+eps)^0.25 으로 부드럽게
    W = evec[:, :k] @ torch.diag(1./(torch.clamp(ev[:k], min=eps)**0.25))
    if DEVICE == "cuda": torch.cuda.synchronize()
    return mu.squeeze(0).cpu().float().numpy(), W.cpu().float().numpy()


def run_ncwp(X_np, target_dim, k=10, temperature=0.12, lc=1.0, lo=1.0, lk=1.0,
             epochs=20, lr=8e-3, shrink=0.08):
    """NCWP: ZCA-whitening + kNN 그래프 + contrastive 학습"""
    X = torch.from_numpy(X_np).to(DEVICE); N, D = X.shape
    mu, S = _zca_shrink(X, shrink)
    Xw = F.normalize((X - mu) @ S, dim=1)
    k_ = max(1, min(k, N-1))
    knn = _topk_knn(Xw, k_)
    r = min(target_dim, D)
    W = torch.nn.Parameter(torch.randn(D, r, device=DEVICE) / math.sqrt(D))
    opt = torch.optim.AdamW([W], lr=lr)
    B = 64
    for _ in range(epochs):
        idx = torch.randperm(N)[:B]
        pos = knn[idx.cpu(), torch.randint(0, k_, (B,))].to(DEVICE)
        Z = F.normalize(torch.cat([Xw[idx.to(DEVICE)], Xw[pos]], 0) @ W, dim=1)
        sim = (Z @ Z.T) / temperature; sim.fill_diagonal_(-1e9)
        loss = (lk * (-(sim[:B].gather(1, torch.arange(B, 2*B, device=DEVICE).view(-1,1)).squeeze(1)
                        - torch.logsumexp(sim[:B], 1)).mean())
                + lc * _cov_penalty(Z) + lo * _orth_penalty(W))
        opt.zero_grad(); loss.backward(); opt.step()
    if DEVICE == "cuda": torch.cuda.synchronize()
    return W.detach().cpu().numpy()


# ── Timing ────────────────────────────────────────────────
def measure_time(fn, *args, n_runs=N_RUNS, **kwargs):
    """fn을 n_runs번 실행하여 wall-clock 시간 (초) 반환."""
    times = []
    for _ in range(n_runs):
        _set_seed(SEED)
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


# ── Main ──────────────────────────────────────────────────
def main():
    print_env()

    records = []

    for model_key, path_dict in MODEL_CONFIGS.items():
        for n_tag, emb_path in path_dict.items():
            if not os.path.exists(emb_path):
                print(f"[Skip] {emb_path} not found")
                continue

            X_all = np.load(emb_path)   # (N, D)
            N, D = X_all.shape
            print(f"\n{'─'*60}")
            print(f"  Model={model_key}  N={N}  D={D}  ({n_tag})")
            print(f"{'─'*60}")

            for target_dim in TARGET_DIMS:
                print(f"\n  target_dim={target_dim}")

                # PCA-Whitening
                mu_t, std_t = measure_time(run_pca_white, X_all, target_dim)
                print(f"    PCA-White    : {mu_t*1000:7.1f} ms  ± {std_t*1000:.1f} ms")
                records.append(dict(model=model_key, n_tag=n_tag, N=N, D=D,
                                    target_dim=target_dim, method="PCA-Whitening",
                                    mean_ms=mu_t*1000, std_ms=std_t*1000))

                # Soft-Whitening
                mu_t, std_t = measure_time(run_soft_white, X_all, target_dim)
                print(f"    Soft-White   : {mu_t*1000:7.1f} ms  ± {std_t*1000:.1f} ms")
                records.append(dict(model=model_key, n_tag=n_tag, N=N, D=D,
                                    target_dim=target_dim, method="Soft-Whitening",
                                    mean_ms=mu_t*1000, std_ms=std_t*1000))

                # NCWP
                mu_t, std_t = measure_time(run_ncwp, X_all, target_dim)
                print(f"    NCWP (ours)  : {mu_t*1000:7.1f} ms  ± {std_t*1000:.1f} ms")
                records.append(dict(model=model_key, n_tag=n_tag, N=N, D=D,
                                    target_dim=target_dim, method="NCWP",
                                    mean_ms=mu_t*1000, std_ms=std_t*1000))

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_ROOT, "timing_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    pdf_path = os.path.join(OUT_ROOT, "timing_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_timing(df, pdf=pdf)
    print(f"PDF 저장: {pdf_path}")


# ── Plotting ─────────────────────────────────────────────
def plot_timing(df, pdf=None):
    methods   = ["PCA-Whitening", "Soft-Whitening", "NCWP"]
    colors    = {"PCA-Whitening": "#4CAF50", "Soft-Whitening": "#FF9800", "NCWP": "#E91E63"}
    hatches   = {"PCA-Whitening": "",        "Soft-Whitening": "//",       "NCWP": "xx"}
    dim_alpha = {32: 1.0, 64: 0.55}

    # 플롯 구성: 행=모델, 열=N
    combos = df[["model", "n_tag", "N", "D"]].drop_duplicates().sort_values(["model", "N"])
    n_cols = len(combos)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    x = np.arange(len(methods))
    bar_w = 0.35

    for ax, (_, row) in zip(axes, combos.iterrows()):
        sub = df[(df["model"] == row["model"]) & (df["n_tag"] == row["n_tag"])]

        for di, dim in enumerate(TARGET_DIMS):
            d = sub[sub["target_dim"] == dim].set_index("method")
            offset = (di - 0.5) * bar_w

            for mi, method in enumerate(methods):
                if method not in d.index:
                    continue
                val  = d.loc[method, "mean_ms"]
                err  = d.loc[method, "std_ms"]
                bar  = ax.bar(x[mi] + offset, val, bar_w * 0.95,
                              color=colors[method], alpha=dim_alpha[dim],
                              hatch=hatches[method], edgecolor="white", lw=0.5,
                              label=f"{method} (dim={dim})" if di == 0 else "_")
                ax.errorbar(x[mi] + offset, val, yerr=err,
                            fmt="none", color="black", capsize=3, lw=1.2)
                # 값 레이블
                ax.text(x[mi] + offset, val + err + val*0.03,
                        f"{val:.0f}" if val >= 10 else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=10, fontsize=9)
        ax.set_ylabel("Training Time (ms)", fontsize=10)
        ax.set_title(f"{row['model']}\nN={row['N']:,}  D={row['D']:,}  ({row['n_tag']})",
                     fontsize=10, fontweight="bold")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.grid(True, which="both", axis="y", ls="--", alpha=0.4)

    # 범례: dim=32 (진한) vs dim=64 (연한)
    legend_patches = []
    for method in methods:
        legend_patches.append(
            mpatches.Patch(color=colors[method], label=method, hatch=hatches[method]))
    dim_patches = [
        mpatches.Patch(color="gray", alpha=1.0,  label="dim=32 (진한)"),
        mpatches.Patch(color="gray", alpha=0.55, label="dim=64 (연한)"),
    ]
    fig.legend(handles=legend_patches + dim_patches,
               loc="lower center", ncol=len(legend_patches) + 2,
               fontsize=9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Offline Projection Training Time: NCWP vs Baselines\n"
                 "(5 runs avg ± std, log scale, GPU)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    else:
        fname = os.path.join(OUT_ROOT, "timing_plot.pdf")
        plt.savefig(fname, bbox_inches="tight")
        print(f"그래프 저장: {fname}")
    plt.close()

    # 텍스트 요약표
    print(f"\n{'='*65}")
    print("  Training Time Summary  (mean ms ± std ms, 5 runs)")
    print(f"{'='*65}")
    for _, row in combos.iterrows():
        sub = df[(df["model"] == row["model"]) & (df["n_tag"] == row["n_tag"])]
        print(f"\n  [{row['model']}]  N={row['N']:,}  D={row['D']:,}")
        print(f"  {'Method':<18} {'dim=32':>14}  {'dim=64':>14}")
        print(f"  {'─'*50}")
        for method in methods:
            r32 = sub[(sub["method"]==method) & (sub["target_dim"]==32)]
            r64 = sub[(sub["method"]==method) & (sub["target_dim"]==64)]
            s32 = f"{r32['mean_ms'].values[0]:6.1f} ± {r32['std_ms'].values[0]:.1f}" if len(r32) else "  N/A"
            s64 = f"{r64['mean_ms'].values[0]:6.1f} ± {r64['std_ms'].values[0]:.1f}" if len(r64) else "  N/A"
            print(f"  {method:<18} {s32:>14} ms  {s64:>14} ms")


if __name__ == "__main__":
    main()
