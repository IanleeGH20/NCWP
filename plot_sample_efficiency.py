"""
plot_sample_efficiency.py

NCWP Sample Efficiency 실험 결과 시각화.

run_sample_efficiency.sh 실행 후 생성된 CSV들을 읽어서
  X축: fit 데이터 수 N
  Y축: NDCG@10 (또는 Recall@100, MAP)
  선:  Base (점선), PCA-White, NCWP_Default(=NCWP)

형태로 그래프를 저장한다.

출력:
  quora_results/fit_query_sample/sample_efficiency_{metric}.png
  scifact_results/fit_query_sample/sample_efficiency_{metric}.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_ROOT = "/workspace/NCWP"

DATASETS = {
    "quora": {"ns": [3000, 5000, 10000], "models": ["llama-8b", "qwen-4b"]},
}

METHODS_MAIN  = ["Base", "PCA-White", "Soft-White", "Random", "LPP", "NCWP"]
METRICS       = ["ndcg@10", "recall@100", "map"]

COLORS = {
    "Base":       "black",
    "PCA-White":  "#2196F3",
    "Soft-White": "#03A9F4",
    "Random":     "#9E9E9E",
    "LPP":        "#FF9800",
    "NCWP":       "#E91E63",
}
MARKS = {
    "Base": "o", "PCA-White": "s", "Soft-White": "D",
    "Random": "x", "LPP": "^", "NCWP": "*",
}


def load_efficiency_data(dataset_name, model_key, ns):
    """N별 CSV 읽어서 {method: {dim: {N: score}}} 구조로 반환"""
    res_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "fit_query_sample")
    records = []  # list of (N, method, dim, metric_dict)

    for n in ns:
        fpath = os.path.join(res_dir, f"{model_key}_N{n}_results_main.csv")
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath} (없음)")
            continue
        df = pd.read_csv(fpath)
        df["fit_n"] = n
        records.append(df)

    if not records:
        return None
    return pd.concat(records, ignore_index=True)


def plot_efficiency(dataset_name, model_key, ns, metric="ndcg@10", pdf=None):
    df = load_efficiency_data(dataset_name, model_key, ns)
    if df is None:
        print(f"  데이터 없음: {dataset_name}/{model_key}")
        return

    out_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "fit_query_sample")
    os.makedirs(out_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Base는 N에 무관하므로 N별 평균값으로 수평선 표시
    base_rows = df[df["method"] == "Base"]
    if not base_rows.empty:
        base_val = base_rows[metric].mean()
        ax.axhline(base_val, ls="--", color=COLORS["Base"], lw=1.5,
                   label=f"Base ({base_val:.2f})", zorder=2)

    # 각 방법별: dim=best(N별로 최댓값) 또는 dim별 subplot → 여기서는 best dim 사용
    for method in METHODS_MAIN:
        if method == "Base":
            continue
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        # N별로 best dim 성능 추출
        best_per_n = (sub.groupby("fit_n")[metric]
                        .max()
                        .reset_index()
                        .sort_values("fit_n"))
        ax.plot(best_per_n["fit_n"], best_per_n[metric],
                marker=MARKS.get(method, "o"), color=COLORS.get(method, None),
                label=method, lw=2, markersize=8, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of Fit Samples (N)", fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(f"{dataset_name.capitalize()} — {model_key} — Sample Efficiency\n"
                 f"({metric.upper()} @ best dim, fit_mode=query_sample)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    else:
        fname = os.path.join(out_dir, f"{model_key}_efficiency_{metric.replace('@','')}.pdf")
        plt.savefig(fname, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close()


def plot_efficiency_multi_dim(dataset_name, model_key, ns, method="NCWP", metric="ndcg@10", pdf=None):
    """
    특정 방법(NCWP)에 대해 dim별 곡선을 겹쳐서 보여주는 세부 그림.
    X=N, Y=metric, line=dim
    """
    df = load_efficiency_data(dataset_name, model_key, ns)
    if df is None:
        return

    out_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "fit_query_sample")
    os.makedirs(out_dir, exist_ok=True)

    sub = df[df["method"] == method]
    if sub.empty:
        print(f"  방법 {method} 데이터 없음")
        return

    dims = sorted(sub["dim"].unique())
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", len(dims))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, dim in enumerate(dims):
        d = sub[sub["dim"] == dim].sort_values("fit_n")
        ax.plot(d["fit_n"], d[metric],
                marker="o", color=palette[i], label=f"dim={dim}", lw=1.8, markersize=6)

    # Base 수평선
    base_rows = df[df["method"] == "Base"]
    if not base_rows.empty:
        bv = base_rows[metric].mean()
        ax.axhline(bv, ls="--", color="black", lw=1.5, label=f"Base ({bv:.2f})")

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of Fit Samples (N)", fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(f"{dataset_name.capitalize()} — {model_key} — {method} per dim\n"
                 f"({metric.upper()} vs N, fit_mode=query_sample)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    else:
        fname = os.path.join(out_dir, f"{model_key}_{method}_dim_efficiency_{metric.replace('@','')}.pdf")
        plt.savefig(fname, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close()


def plot_model_comparison(dataset_name, ns, metric="ndcg@10", pdf=None):
    """
    두 모델(llama-8b, qwen-4b)의 NCWP best-dim을 한 그림에 비교.
    """
    out_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "fit_query_sample")
    os.makedirs(out_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = {"llama-8b": "#E91E63", "qwen-4b": "#2196F3"}
    for model_key in ["llama-8b", "qwen-4b"]:
        df = load_efficiency_data(dataset_name, model_key, ns)
        if df is None:
            continue

        # Base
        bv = df[df["method"] == "Base"][metric].mean()
        ax.axhline(bv, ls=":", color=model_colors[model_key], alpha=0.5,
                   label=f"{model_key} Base ({bv:.2f})")

        # NCWP best dim
        sub = df[df["method"] == "NCWP"]
        if sub.empty:
            continue
        best = sub.groupby("fit_n")[metric].max().reset_index().sort_values("fit_n")
        ax.plot(best["fit_n"], best[metric],
                marker="*", color=model_colors[model_key],
                label=f"{model_key} NCWP (best dim)", lw=2, markersize=10)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of Fit Samples (N)", fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(f"{dataset_name.capitalize()} — Sample Efficiency (Model Comparison)\n"
                 f"{metric.upper()} @ best dim, fit_mode=query_sample", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    else:
        fname = os.path.join(out_dir, f"model_comparison_efficiency_{metric.replace('@','')}.pdf")
        plt.savefig(fname, bbox_inches="tight")
        print(f"  Saved: {fname}")
    plt.close()


# ============================================================
if __name__ == "__main__":
    for dataset_name, dcfg in DATASETS.items():
        ns     = dcfg["ns"]
        models = dcfg["models"]
        print(f"\n{'='*55}")
        print(f"  {dataset_name.upper()} — Sample Efficiency 그래프")
        print(f"{'='*55}")

        out_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_results", "fit_query_sample")
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, f"{dataset_name}_efficiency_plots.pdf")
        with PdfPages(pdf_path) as pdf:
            for model_key in models:
                print(f"\n  [{model_key}]")
                for metric in METRICS:
                    plot_efficiency(dataset_name, model_key, ns, metric, pdf=pdf)
                plot_efficiency_multi_dim(dataset_name, model_key, ns, method="NCWP", pdf=pdf)

            print(f"\n  [모델 비교]")
            plot_model_comparison(dataset_name, ns, pdf=pdf)
        print(f"PDF 저장: {pdf_path}")

    print("\n모든 그래프 생성 완료.")
