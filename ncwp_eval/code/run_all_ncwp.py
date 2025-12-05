import argparse
import os
import re
from typing import Dict, List, Set, Tuple

import torch
import pandas as pd

from .eval import run_ncwp_evaluation
from .graphs import plot_metrics_vs_dim, plot_per_label_maps, plot_per_label_panels


def _discover_weight_roots(weights_root: str) -> List[str]:
    """
    Find all directories (recursively) that contain a 'vocab.json' and at least one 'base_dim_*.pt' file.
    """
    weight_dirs: List[str] = []
    for root, dirs, files in os.walk(weights_root):
        if "vocab.json" in files and any(f.startswith("base_dim_") and f.endswith(".pt") for f in files):
            weight_dirs.append(root)
    weight_dirs.sort()
    return weight_dirs


def _list_available_base_dims(weight_dir: str) -> List[int]:
    """
    List unique base dimensions found in filenames like 'base_dim_{D}.pt' or 'base_dim_{D}_last.pt'.
    """
    dims: Set[int] = set()
    for name in os.listdir(weight_dir):
        m = re.match(r"base_dim_(\d+)(?:_last)?\.pt$", name)
        if m:
            dims.add(int(m.group(1)))
    return sorted(dims)


def _choose_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_dims_for_base(base_dim: int) -> List[int]:
    """
    Generate dims: 2,4,8,... up to (and including) base_dim//2.
    If base_dim < 4, returns [].
    """
    out: List[int] = []
    target_max = max(0, base_dim // 2)
    d = 2
    while d <= target_max:
        out.append(d)
        d *= 2
    return out


def _save_outputs(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Save] {out_csv}")

    # Build metric keys available in df to drive plotting
    metrics = []
    for col in ["nvidia/llama-embed-nemotron-8b", "qwen3-4b", "qwen3-8b", "neighbors_union", "U2", "U3", "overlap2", "overlap3"]:
        key = f"{col}_mAP"
        if key in df.columns:
            metrics.append((key, f"{col} mAP"))
    if metrics:
        plot_path = os.path.join(out_dir, "map_vs_dim.png")
        plot_metrics_vs_dim(df, metrics, "mAP vs Dimension — Base vs PCA vs LPP vs NCWP", plot_path)
        print(f"[Save] {plot_path}")
        # Per-label plots
        label_names = [m[0].replace("_mAP", "") for m in metrics]
        plot_per_label_maps(df, label_names, out_dir, prefix="map_vs_dim")
        # Also generate panel plots per label
        for lab in label_names:
            panel_path = os.path.join(out_dir, f"metrics_panels_{lab.replace('/','_')}.png")
            plot_per_label_panels(df, lab, panel_path)
        print(f"[Save] per-label plots under: {out_dir}")


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NCWP evaluation for all discovered weight directories and base dims.")
    p.add_argument("--corpus", "-c", required=True, help="Path to labeled corpus CSV (e.g., labeled_corpus1000.csv)")
    p.add_argument("--weights-root", "-w", required=True, help="Root directory that contains subfolders with weights")
    p.add_argument("--out-root", "-o", required=True, help="Root directory to store evaluation outputs")
    p.add_argument("--dims", nargs="*", type=int, help="Target dims to evaluate (default: 2..256 filtered by base_dim)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device preference (default: auto)")
    p.add_argument("--block-size", type=int, default=64, help="Block size used in the base model (default: 64)")
    return p


def main():
    args = build_parser().parse_args()
    device = _choose_device(args.device)
    print(f"[Info] device: {device}")

    weight_dirs = _discover_weight_roots(args.weights_root)
    if not weight_dirs:
        raise SystemExit(f"No valid weight directories found under: {args.weights_root}")

    for wdir in weight_dirs:
        rel_path = os.path.relpath(wdir, args.weights_root)
        base_dims = _list_available_base_dims(wdir)
        # Run from base_dim >= 4 (skip 2-dim base)
        base_dims = [bd for bd in base_dims if bd >= 4]
        if not base_dims:
            print(f"[Skip] No base_dim_*.pt files in: {wdir}")
            continue
        print(f"[Info] Evaluating weights in: {wdir} (dims: {base_dims})")

        for base_dim in base_dims:
            try:
                out_dir = os.path.join(args.out_root, rel_path, f"dim_{base_dim}")
                print(f"[Run] base_dim={base_dim} -> out: {out_dir}")
                dims = args.dims if args.dims else _default_dims_for_base(base_dim)
                df = run_ncwp_evaluation(
                    corpus_csv=args.corpus,
                    weights_dir=wdir,
                    base_dim=base_dim,
                    dims=dims,
                    device=device,
                    block_size=args.block_size,
                )
                _save_outputs(df, out_dir)
            except Exception as e:
                print(f"[Error] Failed for {wdir} base_dim={base_dim}: {e}")


if __name__ == "__main__":
    main()


