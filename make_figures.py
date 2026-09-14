"""
make_figures.py  — Figure 2~6 생성
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import pandas as pd
import os

OUT_DIR = "/workspace/NCWP"
C = {
    "pca":  "#378ADD",
    "zca":  "#EF9F27",
    "ncwp": "#A32D2D",
    "base": "#666666",
    "noqr": "#8E44AD",
    "nohn": "#16A085",
    "noref":"#E67E22",
    "randp":"#95A5A6",
}


# ══════════════════════════════════════════════════════════════
# Figure 2 — Pipeline Diagram
# ══════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3)
    ax.axis("off"); fig.patch.set_facecolor("white")

    BOXES = [
        (0.3,  "Frozen\nEncoder",      "#D5E8D4", "#82B366"),
        (2.5,  "ZCA-shrink\nWhitening","#FFF2CC", "#D6B656"),
        (4.7,  "k-NN\nMining",         "#DAE8FC", "#6C8EBF"),
        (6.9,  "Contrastive\nLearning","#F8CECC", "#B85450"),
        (9.1,  "Projection\nMatrix $W$","#E1D5E7", "#9673A6"),
    ]
    BW, BH, BY = 1.8, 1.1, 0.95

    for x, label, fc, ec in BOXES:
        rect = mpatches.FancyBboxPatch(
            (x, BY), BW, BH,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        ax.text(x + BW/2, BY + BH/2, label,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="#333", zorder=4)

    # Arrows between boxes
    ARROW_KW = dict(arrowstyle="->,head_width=0.22,head_length=0.18",
                    color="#555", lw=1.8)
    for i in range(len(BOXES)-1):
        x_start = BOXES[i][0] + BW + 0.03
        x_end   = BOXES[i+1][0] - 0.03
        y_mid   = BY + BH/2
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=ARROW_KW, zorder=5)

    # Sub-labels below each box
    sublabels = [
        "text → $\\mathbf{x} \\in \\mathbb{R}^D$",
        "$\\mathbf{x}_w = \\mathrm{norm}((\\mathbf{x}{-}\\mu)S)$",
        "$\\mathcal{P}(i) = \\{j \\in N_k(i)\\,|\\,\\cos \\geq \\tau\\}$",
        "Symmetric InfoNCE\n+ cov/orth penalty",
        "$\\mathbf{z} = (\\mathbf{x}{-}\\mu_{\\mathrm{in}})W$",
    ]
    for (x, _, _, _), sub in zip(BOXES, sublabels):
        ax.text(x + BW/2, BY - 0.28, sub,
                ha="center", va="top", fontsize=7.8,
                color="#555", style="italic")

    # Top annotations: frozen vs learnable
    ax.text(0.3 + BW/2, BY + BH + 0.18, "frozen",
            ha="center", fontsize=9, color="#82B366", fontweight="bold")
    for (x, _, _, _) in BOXES[1:]:
        ax.text(x + BW/2, BY + BH + 0.18, "unsupervised",
                ha="center", fontsize=9, color="#B85450", fontweight="bold")

    # QR retraction annotation (curved arrow under contrastive box)
    ax.annotate("",
        xy  =(BOXES[3][0] + 0.4, BY - 0.05),
        xytext=(BOXES[3][0] + BW - 0.4, BY - 0.05),
        arrowprops=dict(arrowstyle="<->", color="#B85450", lw=1.3,
                        connectionstyle="arc3,rad=0.4"))
    ax.text(BOXES[3][0] + BW/2, 0.38,
            "QR retraction\nevery $s$ steps",
            ha="center", va="center", fontsize=7.8, color="#B85450")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Figure 3 — Ablation Barplot
# ══════════════════════════════════════════════════════════════
def fig3():
    ab   = pd.read_csv(f"{OUT_DIR}/quora_results/ablation/ablation_results.csv")
    ext  = pd.read_csv(f"{OUT_DIR}/quora_results/ablation/ablation_ext_results.csv")

    TARGET_DIMS = [24, 48, 96, 192]
    VARIANTS = [
        ("PCA-White",    C["pca"],  "--"),
        ("ZCA-only",     C["zca"],  "//"),
        ("No-hardneg",   C["nohn"], ""),
        ("No-QR",        C["noqr"], ""),
        ("No-refinement",C["noref"],""),
        ("Full-NCWP",    C["ncwp"], ""),
    ]

    # Build lookup
    lookup = {}
    for _, row in ab.iterrows():
        lookup[(row["variant"], int(row["dim"]))] = row["ndcg@10"]
    for _, row in ext.iterrows():
        if row.get("model","") == "e5-base":
            key = (row["variant"], int(row["dim"]))
            if key not in lookup:
                lookup[key] = row["ndcg@10"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=False)
    fig.patch.set_facecolor("white")

    x = np.arange(len(VARIANTS))
    w = 0.72

    for ax, dim in zip(axes, TARGET_DIMS):
        ax.set_facecolor("#FAFAFA")
        vals = [lookup.get((v, dim), np.nan) for v, _, _ in VARIANTS]
        labels = [v.replace("-", "\u2013") for v, _, _ in VARIANTS]
        hatches = [h for _, _, h in VARIANTS]
        colors  = [c for _, c, _ in VARIANTS]

        bars = ax.bar(x, vals, width=w, color=colors, edgecolor="white",
                      linewidth=0.8, zorder=3, hatch=None)
        for bar, hatch, val in zip(bars, hatches, vals):
            bar.set_hatch(hatch)
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + 0.6, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7.2,
                        color="#333", rotation=0)

        ax.set_title(f"$r$ = {dim}", fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("nDCG@10" if dim == 24 else "", fontsize=10)
        ax.set_ylim(0, min(100, max(v for v in vals if not np.isnan(v)) * 1.18))
        ax.grid(True, axis="y", alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle("Ablation Study — Quora / E5-base", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Figure 4 — LLM Backbone Results
# ══════════════════════════════════════════════════════════════
def fig4():
    MODELS = [
        ("llama-8b", "LLaMA-3.1-8B", "FiQA"),
        ("qwen-4b",  "Qwen1.5-4B",   "FiQA"),
        ("llama-8b", "LLaMA-3.1-8B", "NFCorpus"),
        ("qwen-4b",  "Qwen1.5-4B",   "NFCorpus"),
    ]
    FIT_MODE = {"FiQA": "corpus_sample", "NFCorpus": "dev_queries"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.patch.set_facecolor("white")

    for ax, (model, mlabel, ds) in zip(axes, MODELS):
        dsl = ds.lower()
        fm  = FIT_MODE[ds]
        path = f"{OUT_DIR}/{dsl}_results/fit_{fm}/{model}_results_main.csv"
        if not os.path.exists(path):
            ax.axis("off"); continue
        df = pd.read_csv(path)
        ax.set_facecolor("#FAFAFA")

        base_v = float(df[df.method=="Base"]["ndcg@10"].values[0])
        base_d = float(df[df.method=="Base"]["dim"].values[0])

        for method, col, lw, ls, ms, mk in [
            ("PCA-White", C["pca"],  1.8, "--", 6, "s"),
            ("NCWP",      C["ncwp"], 2.6, "-",  7, "o"),
        ]:
            sub = df[df.method==method].sort_values("dim")
            line, = ax.plot(sub["dim"], sub["ndcg@10"],
                            color=col, lw=lw, ls=ls,
                            marker=mk, ms=ms, zorder=4,
                            label=method)
            if method == "NCWP":
                line.set_path_effects([pe.withStroke(linewidth=3.8,
                                                     foreground="white")])

        ax.axhline(y=base_v, color=C["base"], lw=1.1, ls=":", alpha=0.75)
        ax.text(sub["dim"].min()*1.05, base_v + 0.3,
                f"Base {base_v:.1f}", fontsize=8, color=C["base"])

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Dim $r$", fontsize=10)
        ax.set_ylabel("nDCG@10" if model == "llama-8b" and ds == "FiQA"
                       else "", fontsize=10)
        ax.set_title(f"{mlabel}\n({ds})", fontsize=10, pad=5)
        ax.grid(True, axis="y", alpha=0.22, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        if model == "llama-8b" and ds == "FiQA":
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
                      edgecolor="#ccc")

    fig.suptitle("LLM Backbone Results — FiQA & NFCorpus",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Figure 5 — Cross-dataset Scope
# ══════════════════════════════════════════════════════════════
def fig5():
    DATASETS = [
        ("quora",    "Quora",    "e5-base", "corpus_sample"),
        ("mrpc",     "MRPC",     "e5-base", "corpus_sample"),
        ("sick",     "SICK",     "e5-base", "corpus_sample"),
        ("scidocs",  "SCIDOCS",  "e5-base", "corpus_sample"),
        ("nfcorpus", "NFCorpus", "e5-base", "corpus_sample"),
        ("fiqa",     "FiQA",     "e5-base", "corpus_sample"),
    ]
    TARGET_DIMS = [24, 48, 96, 192]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, (ds, dslabel, model, fm) in zip(axes, DATASETS):
        path = f"{OUT_DIR}/{ds}_results/fit_{fm}/{model}_results_main.csv"
        if not os.path.exists(path):
            ax.axis("off"); continue
        df = pd.read_csv(path)
        ax.set_facecolor("#FAFAFA")

        base_v = float(df[df.method=="Base"]["ndcg@10"].values[0])
        ax.axhline(y=base_v, color=C["base"], lw=1.1, ls=":", alpha=0.7)
        ax.text(21, base_v + 0.5, f"Base {base_v:.1f}",
                fontsize=8, color=C["base"])

        for method, col, lw, ls, mk in [
            ("PCA-White", C["pca"],  1.8, "--", "s"),
            ("NCWP",      C["ncwp"], 2.5, "-",  "o"),
        ]:
            sub = df[(df.method==method) &
                     (df.dim.isin(TARGET_DIMS))].sort_values("dim")
            if sub.empty: continue
            line, = ax.plot(sub["dim"], sub["ndcg@10"],
                            color=col, lw=lw, ls=ls,
                            marker=mk, ms=6.5, zorder=4, label=method)
            if method == "NCWP":
                line.set_path_effects([pe.withStroke(linewidth=3.5,
                                                     foreground="white")])

        ax.set_xscale("log", base=2)
        ax.set_xticks(TARGET_DIMS)
        ax.set_xticklabels(TARGET_DIMS, fontsize=9)
        ax.set_xlabel("Dim $r$", fontsize=10)
        ax.set_ylabel("nDCG@10", fontsize=10)
        ax.set_title(dslabel, fontsize=11, fontweight="bold", pad=5)
        ax.grid(True, axis="y", alpha=0.22, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if ds == "quora":
            ax.legend(fontsize=9, loc="upper left", framealpha=0.9,
                      edgecolor="#ccc")

    fig.suptitle("NCWP vs. PCA-whitening Across Datasets  (E5-base)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Figure 6 — Anisotropy Diagnostic
# ══════════════════════════════════════════════════════════════
def fig6():
    df = pd.read_csv(f"{OUT_DIR}/quora_results/fit_corpus_sample/"
                     "e5-base_results_main.csv")
    dims = [24, 48, 96, 192, 384]

    pca_ani  = df[df.method=="PCA-White"].set_index("dim")["anisotropy"]
    ncwp_ani = df[df.method=="NCWP"].set_index("dim")["anisotropy"]
    pca_ndcg = df[df.method=="PCA-White"].set_index("dim")["ndcg@10"]
    ncwp_ndcg= df[df.method=="NCWP"].set_index("dim")["ndcg@10"]
    base_ani = float(df[df.method=="Base"]["anisotropy"].values[0])
    base_ndcg= float(df[df.method=="Base"]["ndcg@10"].values[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    # Left: Anisotropy vs Dim
    ax1.set_facecolor("#FAFAFA")
    ax1.axhline(y=base_ani, color=C["base"], lw=1.1, ls=":", alpha=0.75)
    ax1.text(25, base_ani * 1.01, f"Base {base_ani:.3f}",
             fontsize=8.5, color=C["base"])

    for method, col, ls, mk, label in [
        ("PCA-White", C["pca"],  "--", "s", "PCA-whitening"),
        ("NCWP",      C["ncwp"], "-",  "o", "NCWP"),
    ]:
        d_vals = [d for d in dims if d in (pca_ani if method=="PCA-White"
                                            else ncwp_ani).index]
        ani    = pca_ani if method=="PCA-White" else ncwp_ani
        line, = ax1.plot(d_vals, [ani[d] for d in d_vals],
                         color=col, lw=2, ls=ls, marker=mk, ms=6.5,
                         zorder=4, label=label)
        if method=="NCWP":
            line.set_path_effects([pe.withStroke(linewidth=3.5,
                                                  foreground="white")])

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(dims); ax1.set_xticklabels(dims)
    ax1.set_xlabel("Projected dimension $r$", fontsize=11)
    ax1.set_ylabel("Anisotropy", fontsize=11)
    ax1.set_title("(a) Anisotropy vs. Dimension", fontsize=11, pad=8)
    ax1.legend(fontsize=10, framealpha=0.9, edgecolor="#ccc")
    ax1.grid(True, axis="y", alpha=0.22); ax1.set_axisbelow(True)
    for sp in ["top","right"]: ax1.spines[sp].set_visible(False)

    # Right: Anisotropy vs NDCG (scatter)
    ax2.set_facecolor("#FAFAFA")
    for method, col, mk, label in [
        ("PCA-White", C["pca"],  "s", "PCA-whitening"),
        ("NCWP",      C["ncwp"], "o", "NCWP"),
    ]:
        ani  = pca_ani  if method=="PCA-White" else ncwp_ani
        ndcg = pca_ndcg if method=="PCA-White" else ncwp_ndcg
        d_vals = [d for d in dims if d in ani.index and d in ndcg.index]
        sc = ax2.scatter([ani[d] for d in d_vals],
                         [ndcg[d] for d in d_vals],
                         color=col, marker=mk, s=70, zorder=4,
                         label=label, alpha=0.85, edgecolors="white", lw=0.6)
        for d in d_vals:
            ax2.annotate(str(d), (ani[d], ndcg[d]),
                         xytext=(3, 3), textcoords="offset points",
                         fontsize=7.5, color=col)

    ax2.scatter([base_ani], [base_ndcg], color=C["base"], marker="*",
                s=120, zorder=5, label=f"Base (768-d)")
    ax2.set_xlabel("Anisotropy", fontsize=11)
    ax2.set_ylabel("nDCG@10", fontsize=11)
    ax2.set_title("(b) Isotropy–Quality Trade-off", fontsize=11, pad=8)
    ax2.legend(fontsize=10, framealpha=0.9, edgecolor="#ccc")
    ax2.grid(True, alpha=0.22); ax2.set_axisbelow(True)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)

    fig.suptitle("Anisotropy Diagnostic — Quora / E5-base",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Save all
# ══════════════════════════════════════════════════════════════
FIGS = [
    ("figure2_pipeline.pdf",   fig2),
    ("figure3_ablation.pdf",   fig3),
    ("figure4_llm.pdf",        fig4),
    ("figure5_crossdataset.pdf", fig5),
    ("figure6_anisotropy.pdf", fig6),
]

for fname, fn in FIGS:
    try:
        f = fn()
        out = os.path.join(OUT_DIR, fname)
        with PdfPages(out) as pdf:
            pdf.savefig(f, bbox_inches="tight", dpi=200)
        plt.close(f)
        print(f"✓  {fname}")
    except Exception as e:
        print(f"✗  {fname}: {e}")
        import traceback; traceback.print_exc()
