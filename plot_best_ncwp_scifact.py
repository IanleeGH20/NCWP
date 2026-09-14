"""
plot_best_ncwp_scifact.py

scifact_results/{fit_corpus, fit_mixed}/ 폴더의 CSV를 읽어
각 차원별 Best NCWP vs Baseline 비교 그래프를 생성.
방법 A(corpus)와 방법 C(mixed) 결과를 각각 그리고,
두 방법을 한 그래프에서 비교하는 플롯도 생성.

실행 예시 (컨테이너 내부):
  cd /workspace/NCWP
  python plot_best_ncwp_scifact.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

RESULTS_ROOT = "/workspace/NCWP/scifact_results"
PRIMARY      = "ndcg@10"
MODELS       = ["nano_gpt", "qwen-4b", "qwen-8b", "llama-8b"]
FIT_MODES    = {
    "corpus": "fit_corpus",   # 방법 A
    "mixed":  "fit_mixed",    # 방법 C
}


def load_best_ncwp(results_dir, model_name):
    """CSV 로드 후 Best NCWP + Baseline 데이터프레임 반환."""
    main_csv     = os.path.join(results_dir, f"{model_name}_results_main.csv")
    ablation_csv = os.path.join(results_dir, f"{model_name}_results_ablation.csv")
    dfs = []
    if os.path.exists(main_csv):     dfs.append(pd.read_csv(main_csv))
    if os.path.exists(ablation_csv): dfs.append(pd.read_csv(ablation_csv))
    if not dfs: return None, None

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["method"] = full_df["method"].astype(str)
    ncwp_mask    = full_df["method"].str.contains("NCWP", case=False)
    df_ncwp      = full_df[ncwp_mask]
    df_baselines = full_df[~ncwp_mask]

    best_ncwp = pd.DataFrame()
    if not df_ncwp.empty:
        best_ncwp = df_ncwp.groupby("dim")[PRIMARY].max().reset_index()
        best_ncwp["method"] = "Best NCWP"

    return df_baselines, best_ncwp


def plot_single(model_name, fit_mode_key, results_dir, pdf=None):
    """방법 A 또는 C 단독 그래프."""
    df_base, best_ncwp = load_best_ncwp(results_dir, model_name)
    if df_base is None:
        print(f"  [Skip] {results_dir}/{model_name}_results_main.csv not found.")
        return

    plot_data = pd.concat([df_base, best_ncwp], ignore_index=True).sort_values("dim")
    label_suffix = "corpus only" if fit_mode_key == "corpus" else "query+corpus"

    plt.figure(figsize=(10, 6))
    for method in df_base["method"].unique():
        subset = plot_data[plot_data["method"] == method]
        if method == "Base":
            base_val = subset[PRIMARY].mean()
            plt.axhline(y=base_val, color="black", linestyle="--",
                        label=f"Base ({base_val:.2f})", alpha=0.7)
        else:
            plt.plot(subset["dim"], subset[PRIMARY],
                     marker="o", label=method, alpha=0.7, linestyle="--")
    if not best_ncwp.empty:
        subset = plot_data[plot_data["method"] == "Best NCWP"]
        plt.plot(subset["dim"], subset[PRIMARY],
                 marker="*", label="NCWP", linewidth=2.5, color="red", markersize=10)

    plt.xscale("log", base=2)
    plt.xlabel("Dimension (log scale)")
    plt.ylabel(PRIMARY.upper())
    plt.title(f"{model_name} - SciFact {PRIMARY.upper()} [{label_suffix}]")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    if pdf is not None:
        pdf.savefig(bbox_inches="tight")
    else:
        out = os.path.join(results_dir, f"{model_name}_best_ncwp_graph.pdf")
        plt.savefig(out, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close()


def plot_comparison(model_name, pdf=None):
    """방법 A vs 방법 C NCWP 성능 비교 그래프."""
    corpus_dir = os.path.join(RESULTS_ROOT, FIT_MODES["corpus"])
    mixed_dir  = os.path.join(RESULTS_ROOT, FIT_MODES["mixed"])

    _, best_a = load_best_ncwp(corpus_dir, model_name)
    _, best_c = load_best_ncwp(mixed_dir,  model_name)

    # Base는 어느 쪽이든 동일하므로 corpus에서 가져옴
    df_base_a, _ = load_best_ncwp(corpus_dir, model_name)

    if best_a is None and best_c is None:
        return

    plt.figure(figsize=(10, 6))

    # Base 수평선
    if df_base_a is not None:
        base_val = df_base_a[df_base_a["method"] == "Base"][PRIMARY].mean()
        plt.axhline(y=base_val, color="black", linestyle="--",
                    label=f"Base ({base_val:.2f})", alpha=0.7)

    if best_a is not None and not best_a.empty:
        s = best_a.sort_values("dim")
        plt.plot(s["dim"], s[PRIMARY], marker="o", color="steelblue",
                 linewidth=2, label="NCWP (방법 A: corpus only)")

    if best_c is not None and not best_c.empty:
        s = best_c.sort_values("dim")
        plt.plot(s["dim"], s[PRIMARY], marker="*", color="red",
                 linewidth=2, markersize=9, label="NCWP (방법 C: query+corpus)")

    plt.xscale("log", base=2)
    plt.xlabel("Dimension (log scale)")
    plt.ylabel(PRIMARY.upper())
    plt.title(f"{model_name} - SciFact: 방법 A vs 방법 C")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    if pdf is not None:
        pdf.savefig(bbox_inches="tight")
    else:
        out = os.path.join(RESULTS_ROOT, f"{model_name}_A_vs_C_comparison.pdf")
        plt.savefig(out, bbox_inches="tight")
        print(f"  [비교] Saved → {out}")
    plt.close()


if __name__ == "__main__":
    pdf_path = os.path.join(RESULTS_ROOT, "scifact_best_ncwp_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        for model in MODELS:
            print(f"\n=== {model} ===")
            for mode_key, subdir in FIT_MODES.items():
                results_dir = os.path.join(RESULTS_ROOT, subdir)
                if os.path.isdir(results_dir):
                    plot_single(model, mode_key, results_dir, pdf=pdf)
            plot_comparison(model, pdf=pdf)
    print(f"\nPDF 저장: {pdf_path}")
