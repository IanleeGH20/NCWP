#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_training_summary(weights_dir: str) -> pd.DataFrame:
    summary_path = os.path.join(weights_dir, "training_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Not found: {summary_path}")
    df = pd.read_csv(summary_path)
    # Ensure expected columns exist
    expected = {"dim", "best_val", "best_step"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")
    # Keep the essentials and sort by dim
    df = df[["dim", "best_val", "best_step"]].sort_values("dim").reset_index(drop=True)
    return df


def _merge_and_compare(df_bpe: pd.DataFrame, df_wp: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(df_bpe, df_wp, on="dim", suffixes=("_bpe", "_wp"))
    # Lower best_val is better (validation loss)
    merged["winner"] = merged.apply(
        lambda r: "bpe" if r["best_val_bpe"] < r["best_val_wp"] else ("wordpiece" if r["best_val_wp"] < r["best_val_bpe"] else "tie"),
        axis=1,
    )
    merged["val_gap"] = merged["best_val_bpe"] - merged["best_val_wp"]  # negative => wordpiece better, positive => bpe better
    return merged


def _build_rankings(df_bpe: pd.DataFrame, df_wp: pd.DataFrame) -> pd.DataFrame:
    df_bpe_r = df_bpe.copy()
    df_bpe_r["tokenizer"] = "bpe"
    df_wp_r = df_wp.copy()
    df_wp_r["tokenizer"] = "wordpiece"
    stacked = pd.concat([df_bpe_r, df_wp_r], ignore_index=True)
    stacked = stacked[["tokenizer", "dim", "best_val", "best_step"]].sort_values("best_val", ascending=True)
    stacked["rank"] = range(1, len(stacked) + 1)
    cols = ["rank", "tokenizer", "dim", "best_val", "best_step"]
    return stacked[cols].reset_index(drop=True)


def _plot_val_vs_dim(df: pd.DataFrame, out_pdf: str, title: str) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["dim"], df["best_val_bpe"], marker="o", label="BPE best_val")
    plt.plot(df["dim"], df["best_val_wp"], marker="s", label="WordPiece best_val")
    plt.xlabel("Dimension")
    plt.ylabel("Validation loss (lower is better)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def run_compare(
    bpe_dir: str,
    wordpiece_dir: str,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df_bpe = _read_training_summary(bpe_dir)
    df_wp = _read_training_summary(wordpiece_dir)
    df = _merge_and_compare(df_bpe, df_wp)
    df_rank = _build_rankings(df_bpe, df_wp)

    out_csv = os.path.join(out_dir, "tokenizer_comparison.csv")
    df.to_csv(out_csv, index=False)
    out_rank_csv = os.path.join(out_dir, "tokenizer_rankings.csv")
    df_rank.to_csv(out_rank_csv, index=False)

    out_pdf = os.path.join(out_dir, "val_loss_vs_dim.pdf")
    _plot_val_vs_dim(df, out_pdf, "Validation Loss vs Dimension — BPE vs WordPiece")
    return df, df_rank, out_csv, out_rank_csv, out_pdf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare BPE and WordPiece weights using training_summary.csv (prints table and saves plot).")
    p.add_argument("--bpe-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "bpe")),
                   help="Path to weights/bpe directory (default: ../weights/bpe)")
    p.add_argument("--wordpiece-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "wordpiece")),
                   help="Path to weights/wordpiece directory (default: ../weights/wordpiece)")
    p.add_argument("--out-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "comparison_bpe_wordpiece")),
                   help="Output directory to save comparison CSV and plot (default: ../weights/comparison_bpe_wordpiece)")
    return p


def main():
    args = build_parser().parse_args()
    df, df_rank, out_csv, out_rank_csv, out_png = run_compare(args.bpe_dir, args.wordpiece_dir, args.out_dir)
    # 기본 출력: 차원과 무관한 성능 순위표
    print(df_rank.to_string(index=False))
    print(f"\n[Saved] Rankings CSV: {out_rank_csv}")
    print(f"[Saved] Comparison CSV: {out_csv}")
    print(f"[Saved] Plot: {out_png}")


if __name__ == "__main__":
    main()


