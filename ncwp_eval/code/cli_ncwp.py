import argparse
import os
import torch
import pandas as pd

from .eval import run_ncwp_evaluation, run_ncwp_sweep
from .graphs import plot_metrics_vs_dim, plot_per_label_maps, plot_per_label_panels


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate NCWP on a chosen base dimension.")
    p.add_argument("--corpus", "-c", required=True, help="Path to labeled_corpus_1000.csv")
    p.add_argument("--weights-dir", "-w", required=True, help="Directory containing base_dim_{d}.pt and vocab.json")
    p.add_argument("--base-dim", type=int, default=512, help="Base dimension to load (default: 512)")
    p.add_argument("--dims", nargs="*", type=int, help="Target dims to evaluate (default: 4 8 16 32 64 128 256)")
    p.add_argument("--prefer-cuda", action="store_true", help="(Deprecated) Prefer CUDA if available")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Device to use: auto (default; uses CUDA if available), or force cpu/cuda.")
    p.add_argument("--block-size", type=int, default=64, help="Block size used in training (default: 64)")
    p.add_argument("--out-dir", "-o", required=True, help="Output directory to save results and plots")
    # New options
    p.add_argument(
        "--weights-mode",
        choices=["best_only", "checkpoints", "base-dims"],
        default="best_only",
        help="Which base weights to evaluate: only best for base-dim, all checkpoints for base-dim, or all base-dims under weights-dir.",
    )
    p.add_argument(
        "--dims-auto-mode",
        choices=["none", "half_range", "powers_of_two"],
        default="none",
        help="Auto-generate dims if --dims is not provided. half_range -> 2,4,8..(base_dim//2), powers_of_two -> 2^k up to base_dim.",
    )
    p.add_argument(
        "--save-projectors",
        action="store_true",
        help="If set, save learned projector parameters under OUT_DIR/projectors/<base_id>/<method>/k_*.npz",
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="If set, display plots (useful in Jupyter) in addition to saving them.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print progress messages during NCWP training/fitting.",
    )
    p.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show a moving progress bar during NCWP training (requires tqdm).",
    )
    p.add_argument(
        "--include-best-last",
        action="store_true",
        help="In checkpoints mode, include 'best' and 'last' weights in addition to step_* checkpoints (default: only step_*).",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Determine device (default: auto)
    if args.prefer_cuda:
        # Backward-compatible behavior: prefer CUDA if available, else CPU
        device = "cuda" if torch.cuda.is_available() else "workaround_cpu"
    else:
        if args.device == "cpu":
            device = "cpu"
        elif args.device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    # Normalize potential placeholder
    if device == "workaround_cpu":
        device = "cpu"
    print(f"[Info] device: {device}")

    def _compute_dims_for_base(bd: int):
        if args.dims:
            return args.dims
        if args.dims_auto_mode == "half_range":
            out = []
            limit = max(2, int(bd) // 2)
            d = 2
            while d <= limit:
                out.append(d)
                d *= 2
            return out
        if args.dims_auto_mode == "powers_of_two":
            out = []
            d = 2
            while d <= int(bd):
                out.append(d)
                d *= 2
            return out
        return None

    os.makedirs(args.out_dir, exist_ok=True)
    if args.weights_mode == "best_only":
        dims_local = _compute_dims_for_base(args.base_dim)
        df = run_ncwp_evaluation(
            corpus_csv=args.corpus,
            weights_dir=args.weights_dir,
            base_dim=args.base_dim,
            dims=dims_local,
            device=device,
            block_size=args.block_size,
            save_projectors=args.save_projectors,
            projector_root=os.path.join(args.out_dir, "projectors"),
            progress=args.progress,
            progress_bar=args.progress_bar,
        )
        out_csv = os.path.join(args.out_dir, "comparison.csv")
        df.to_csv(out_csv, index=False)
        print(f"[Save] {out_csv}")
        # Plot mAP vs dims for all available GTs (auto-detect *_mAP columns, incl. U2/U3)
        label_names = sorted({c[:-4] for c in df.columns if c.endswith("_mAP")})
        metrics = [(f"{lab}_mAP", f"{lab} mAP") for lab in label_names]
        if label_names:
            plot_path = os.path.join(args.out_dir, "map_vs_dim.png")
            plot_metrics_vs_dim(df, metrics, "mAP vs Dimension — Base vs PCA vs NCWP", plot_path, show=args.show_plots)
            print(f"[Save] {plot_path}")
            plot_per_label_maps(df, label_names, args.out_dir, prefix="map_vs_dim", show=args.show_plots)
            for lab in label_names:
                panel_path = os.path.join(args.out_dir, f"metrics_panels_{lab.replace('/','_')}.png")
                plot_per_label_panels(df, lab, panel_path, show=args.show_plots)
            print(f"[Save] per-label plots under: {args.out_dir}")
    else:
        # sweep modes
        df = run_ncwp_sweep(
            corpus_csv=args.corpus,
            weights_dir=args.weights_dir,
            base_dim=(args.base_dim if args.weights_mode in ("best_only", "checkpoints") else None),
            dims=None,  # computed per-run below via dims_mode
            device=device,
            block_size=args.block_size,
            val_ratio=0.2,
            seed=42,
            mode=args.weights_mode,
            out_dir=args.out_dir,
            save_projectors=args.save_projectors,
            projector_root=os.path.join(args.out_dir, "projectors"),
            dims_mode=args.dims_auto_mode,
            progress=args.progress,
            progress_bar=args.progress_bar,
            include_best_last=args.include_best_last,
        )
        out_csv = os.path.join(args.out_dir, "comparison.csv")
        df.to_csv(out_csv, index=False)
        print(f"[Save] {out_csv}")
        # Generate per-run plots into runs/<base_id>/ (auto-detect *_mAP columns)
        label_names_all = sorted({c[:-4] for c in df.columns if c.endswith("_mAP")})
        metrics_all = [(f"{lab}_mAP", f"{lab} mAP") for lab in label_names_all]
        if label_names_all:
            # group by checkpoint label if present, else by base_dim
            if "checkpoint" in df.columns:
                groups = sorted(df["checkpoint"].dropna().unique().tolist())
            else:
                groups = [f"dim{int(bd)}" for bd in sorted(df["base_dim"].dropna().unique().tolist())]
            for g in groups:
                if "checkpoint" in df.columns:
                    sub = df[df["checkpoint"] == g]
                    subdir = os.path.join(args.out_dir, "runs", g)
                else:
                    bd_val = int(str(g).replace("dim", ""))
                    sub = df[df["base_dim"] == bd_val]
                    subdir = os.path.join(args.out_dir, "runs", g)
                os.makedirs(subdir, exist_ok=True)
                plot_path = os.path.join(subdir, "map_vs_dim.png")
                plot_metrics_vs_dim(sub, metrics_all, "mAP vs Dimension — Base vs PCA vs NCWP", plot_path, show=args.show_plots)
                plot_per_label_maps(sub, label_names_all, subdir, prefix="map_vs_dim", show=args.show_plots)
                for lab in label_names_all:
                    panel_path = os.path.join(subdir, f"metrics_panels_{lab.replace('/','_')}.png")
                    plot_per_label_panels(sub, lab, panel_path, show=args.show_plots)
            print(f"[Save] per-run plots under: {os.path.join(args.out_dir, 'runs')}")


if __name__ == "__main__":
    main()


