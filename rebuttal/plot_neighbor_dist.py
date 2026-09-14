#!/usr/bin/env python3
"""v2 spec §5.4: violin plot of gold STS scores among mined neighbors, per space.

Reads the raw score dumps from run_neighbor_score_dist --dump_raw and draws, per
backbone, the distribution of gold STS-B scores for neighbors mined in each space
(Random / Raw kNN / ZCA-whitened kNN / NCWP-projected kNN). A good space mines
higher-scoring (more semantically similar) neighbors.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V2 = "/workspace/NCWP/rebuttal_outputs_v2"
OUT = "/workspace/NCWP/rebuttal_outputs/figures"
MODELS = ["qwen-4b", "llama-8b", "qwen-8b"]
ORDER = ["Random", "Raw", "ZCA-whitened", "NCWP-projected"]
COLORS = ["#b0b0b0", "#7fb2e5", "#f4a259", "#e05a5a"]


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4.2), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    for ax, model in zip(axes, MODELS):
        path = f"{V2}/mined_neighbor_scores_raw_{model}.json"
        if not os.path.exists(path):
            ax.set_title(f"{model}\n(no data)"); continue
        data = json.load(open(path))
        series = [np.array(data.get(s, []), dtype=float) for s in ORDER]
        parts = ax.violinplot([s for s in series], showmeans=True, showextrema=False)
        for pc, c in zip(parts["bodies"], COLORS):
            pc.set_facecolor(c); pc.set_alpha(0.75)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
        for i, s in enumerate(series, 1):
            if len(s):
                ax.text(i, 5.15, f"μ={s.mean():.2f}\nn={len(s)}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(1, len(ORDER) + 1))
        ax.set_xticklabels(["Random", "Raw\nkNN", "ZCA\nkNN", "NCWP\nkNN"], fontsize=8)
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0, 5.6); ax.axhline(4.0, ls="--", lw=0.7, color="gray")
    axes[0].set_ylabel("Gold STS-B score of mined neighbor pairs")
    fig.suptitle("Mined-neighbor gold-score distribution by space (k=10, τ=0)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/sts_mined_neighbor_score_distribution.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}/sts_mined_neighbor_score_distribution.{{pdf,png}}")


if __name__ == "__main__":
    main()
