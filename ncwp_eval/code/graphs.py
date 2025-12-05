from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

_METHOD_STYLES: Dict[str, Dict] = {
    "Base": {"linestyle": "--", "marker": None},
    "PCA-Whitening": {"linestyle": "-", "marker": "o"},
    "Soft-Whitening": {"linestyle": ":", "marker": "s"},
    "LPP": {"linestyle": "-.", "marker": "D"},
    "Random Projection": {"linestyle": "-", "marker": "^"},
    "NCWP": {"linestyle": ":", "marker": "v"},
}

def plot_metrics_vs_dim(df: pd.DataFrame, metrics: List[Tuple[str, str]], title: str, out_path: str, show: bool = False) -> None:
    """
    metrics: list of (column_key, label)
    df columns should include: method, away_dim, and metric columns
    """
    methods = sorted(df["method"].unique().tolist())
    # Dynamic xticks from available dims (exclude Base row which uses away_dim=base_dim)
    dims_unique = sorted(df[df["method"] != "Base"]["away_dim"].unique().tolist())
    xticks = dims_unique if dims_unique else [2]
    plt.figure(figsize=(10, 5))
    for key, label in metrics:
        for m in methods:
            d = df[df["method"] == m].copy().sort_values("away_dim")
            if m == "Base":
                # Draw horizontal dashed line across specified xticks using the single Base value
                if key in df.columns:
                    base_rows = df[(df["method"] == "Base")]
                    if not base_rows.empty and key in base_rows.columns:
                        base_value = base_rows.iloc[0][key]
                        plt.plot(xticks, [base_value] * len(xticks), linestyle="--", linewidth=2, label=f"{m} — {label}")
                continue
            if key not in d.columns:
                continue
            # Plot other methods with markers (use all available dims)
            style = _METHOD_STYLES.get(m, {})
            plt.plot(
                d["away_dim"],
                d[key],
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2,
                label=f"{m} — {label}",
            )
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    # Dynamic xlim based on xticks
    if xticks:
        xmin, xmax = xticks[0], xticks[-1]
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        plt.xlim(xmin, xmax)
        plt.xticks(xticks, [str(x) for x in xticks])
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    if show:
        try:
            import matplotlib.pyplot as _plt  # local alias to avoid confusion
            _plt.show()
        except Exception:
            pass
    plt.close()


def plot_per_label_maps(df: pd.DataFrame, labels: List[str], out_dir: str, prefix: str = "map_vs_dim", show: bool = False) -> None:
    """
    For each label (GT column base name), produce a separate mAP-vs-dimension plot comparing methods.
    labels: e.g., ["nvidia/llama-embed-nemotron-8b","qwen3-4b","qwen3-8b","neighbors_union"]
    """
    methods = sorted(df["method"].unique().tolist())
    dims_unique = sorted(df[df["method"] != "Base"]["away_dim"].unique().tolist())
    xticks = dims_unique if dims_unique else [2]
    for lab in labels:
        key = f"{lab}_mAP"
        if key not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        for m in methods:
            d = df[df["method"] == m].copy().sort_values("away_dim")
            if m == "Base":
                base_rows = df[(df["method"] == "Base")]
                if not base_rows.empty and key in base_rows.columns:
                    base_value = base_rows.iloc[0][key]
                    plt.plot(xticks, [base_value] * len(xticks), linestyle="--", linewidth=2, label=m)
                continue
            if key not in d.columns:
                continue
            x = d["away_dim"].values
            y = d[key].values
            style = _METHOD_STYLES.get(m, {})
            plt.plot(
                x,
                y,
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2,
                label=m,
            )
        plt.title(f"mAP vs Dimension — {lab}")
        plt.xlabel("Dimension")
        plt.ylabel("mAP")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if xticks:
            xmin, xmax = xticks[0], xticks[-1]
            if xmin == xmax:
                xmin, xmax = xmin - 1, xmax + 1
            plt.xlim(xmin, xmax)
            plt.xticks(xticks, [str(x) for x in xticks])
        plt.tight_layout()
        safe_lab = lab.replace("/", "_")
        out_path = f"{out_dir}/{prefix}_{safe_lab}.png"
        plt.savefig(out_path, dpi=150)
        if show:
            try:
                import matplotlib.pyplot as _plt
                _plt.show()
            except Exception:
                pass
        plt.close()


def plot_per_label_panels(df: pd.DataFrame, label: str, out_path: str, show: bool = False) -> None:
    """
    Plot 1x2 panels (mAP / nDCG@maxK) vs dimension for a single label,
    comparing all methods present (Base/PCA/LPP/NCWP).
    """
    keys = [
        (f"{label}_mAP", "mAP"),
        (f"{label}_nDCG@maxK", "nDCG@10"),
    ]
    methods = sorted(df["method"].unique().tolist())
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    dims_unique = sorted(df[df["method"] != "Base"]["away_dim"].unique().tolist())
    xticks = dims_unique if dims_unique else [2]
    for ax, (k, title) in zip(axes, keys):
        for m in methods:
            d = df[df["method"] == m].copy().sort_values("away_dim")
            if m == "Base":
                base_rows = df[(df["method"] == "Base")]
                if not base_rows.empty and k in base_rows.columns:
                    base_value = base_rows.iloc[0][k]
                    ax.plot(xticks, [base_value] * len(xticks), linestyle="--", linewidth=2, label=m)
                continue
            if k not in d.columns:
                continue
            style = _METHOD_STYLES.get(m, {})
            ax.plot(
                d["away_dim"],
                d[k].values,
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2,
                label=m,
            )
        ax.set_title(f"{label} — {title}")
        ax.set_xlabel("Dimension (away_dim)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if xticks:
            xmin, xmax = xticks[0], xticks[-1]
            if xmin == xmax:
                xmin, xmax = xmin - 1, xmax + 1
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])
    plt.savefig(out_path, dpi=150)
    if show:
        try:
            import matplotlib.pyplot as _plt
            _plt.show()
        except Exception:
            pass
    plt.close()

