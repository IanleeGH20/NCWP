"""
make_comparison_table.py
─────────────────────────
Old (backup_v1, v1/v3 혼용) vs New (v4, warmup=200, tau=0.1) 비교 표 생성.
대상: Quora / E5-base / BGE-base / corpus_sample / fit
지표: nDCG@10
"""

import pandas as pd
import numpy as np
import os

BASE = "/workspace/NCWP"
BACKUP = os.path.join(BASE, "backup_v1/quora_fit_corpus_sample")
NEW    = os.path.join(BASE, "quora_results/fit_corpus_sample")

DIMS = [24, 48, 96, 192, 384]

def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def extract(df, method_kw, dims=DIMS):
    if df is None:
        return {}
    sub = df[df['method'].str.contains(method_kw, na=False, case=False)]
    out = {}
    for _, row in sub.iterrows():
        d = int(row['dim'])
        if d in dims:
            out[d] = round(float(row['ndcg@10']), 2)
    return out

def fmt_row(vals, dims=DIMS):
    return [f"{vals.get(d, '—'):>7}" for d in dims]

def delta(old_v, new_v):
    if old_v is None or new_v is None:
        return "—"
    d = new_v - old_v
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"

def make_table(model_key, csv_name_old, csv_name_new, seeds=None):
    old_df = load_csv(os.path.join(BACKUP, csv_name_old))
    new_df = load_csv(os.path.join(NEW,    csv_name_new))

    pca_old = extract(old_df, "PCA")
    pca_new = extract(new_df, "PCA")
    zca_old = extract(old_df, "ZCA")
    zca_new = extract(new_df, "ZCA")
    ncwp_old = extract(old_df, "NCWP")
    ncwp_new = extract(new_df, "NCWP")

    header = f"{'dim':>6}  " + "  ".join(f"{d:>7}" for d in DIMS)
    sep = "-" * (len(header) + 2)
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {model_key.upper()} / Quora / corpus_sample  (nDCG@10)")
    lines.append('='*60)
    lines.append(header)
    lines.append(sep)

    for label, old_v, new_v in [
        ("PCA-White [OLD]", pca_old, None),
        ("PCA-White [NEW]", None, pca_new),
        ("ZCA-only  [OLD]", zca_old, None),
        ("ZCA-only  [NEW]", None, zca_new),
        ("NCWP      [OLD]", ncwp_old, None),
        ("NCWP      [NEW]", None, ncwp_new),
        ("NCWP  Δ(N-O)",    ncwp_old, ncwp_new),
    ]:
        if label.endswith("Δ(N-O)"):
            row_vals = [delta(old_v.get(d), new_v.get(d)) for d in DIMS]
            lines.append(f"{'  Δ NCWP':>16}  " + "  ".join(f"{v:>7}" for v in row_vals))
        elif new_v is None:
            vals = old_v or {}
            lines.append(f"{label:>16}  " + "  ".join(f"{vals.get(d,'—'):>7}" for d in DIMS))
        else:
            vals = new_v or {}
            lines.append(f"{label:>16}  " + "  ".join(f"{vals.get(d,'—'):>7}" for d in DIMS))
    lines.append(sep)

    # Multi-seed (s42 only here; s43/s44 for old/new compared separately)
    if seeds:
        lines.append("  [Multi-seed NCWP — seed 42 / 43 / 44]")
        for s, csv_s_old, csv_s_new in seeds:
            s_old_df = load_csv(os.path.join(BACKUP, csv_s_old)) if csv_s_old else None
            s_new_df = load_csv(os.path.join(NEW,    csv_s_new)) if csv_s_new else None
            s_old = extract(s_old_df, "NCWP") if s_old_df is not None else {}
            s_new = extract(s_new_df, "NCWP") if s_new_df is not None else {}
            lines.append(f"  seed{s} [OLD]  " + "  ".join(f"{s_old.get(d,'—'):>7}" for d in DIMS))
            lines.append(f"  seed{s} [NEW]  " + "  ".join(f"{s_new.get(d,'—'):>7}" for d in DIMS))
            lines.append(f"  seed{s}   Δ    " + "  ".join(f"{delta(s_old.get(d), s_new.get(d)):>7}" for d in DIMS))
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OLD (v1/v3 혼용) vs NEW (v4: warmup=200, cosine_tau=0.1)")
    print("  비교 대상: Quora / corpus_sample / nDCG@10")
    print("="*60)

    # E5-base seed 42
    print(make_table(
        "E5-base",
        "e5-base_results_main.csv",
        "e5-base_results_main.csv",
        seeds=[
            (43, "e5-base_s43_results_main.csv", "e5-base_s43_results_main.csv"),
            (44, "e5-base_s44_results_main.csv", "e5-base_s44_results_main.csv"),
        ]
    ))

    # BGE-base
    print(make_table(
        "BGE-base",
        "bge-base_results_main.csv",
        "bge-base_results_main.csv",
    ))

    # --- std 계산 (multi-seed, NEW 완료 시) ---
    print("\n=== NCWP Multi-seed 통계 (E5-base / Quora) ===")
    for tag, csvs_new in [
        ("NEW (v4)", [
            "e5-base_results_main.csv",
            "e5-base_s43_results_main.csv",
            "e5-base_s44_results_main.csv",
        ]),
        ("OLD (v1/v3)", [
            None, None, None  # placeholder
        ]),
    ]:
        scores_per_dim = {d: [] for d in DIMS}
        csv_srcs = [
            os.path.join(NEW,    "e5-base_results_main.csv"),
            os.path.join(NEW,    "e5-base_s43_results_main.csv"),
            os.path.join(NEW,    "e5-base_s44_results_main.csv"),
        ] if tag.startswith("NEW") else [
            os.path.join(BACKUP, "e5-base_results_main.csv"),
            os.path.join(BACKUP, "e5-base_s43_results_main.csv"),
            os.path.join(BACKUP, "e5-base_s44_results_main.csv"),
        ]
        for src in csv_srcs:
            df = load_csv(src)
            vals = extract(df, "NCWP") if df is not None else {}
            for d in DIMS:
                if d in vals:
                    scores_per_dim[d].append(vals[d])

        print(f"\n  {tag}")
        header2 = f"{'dim':>6}  " + "  ".join(f"{d:>11}" for d in DIMS)
        print(header2)
        means = [np.mean(scores_per_dim[d]) if scores_per_dim[d] else float('nan') for d in DIMS]
        stds  = [np.std(scores_per_dim[d])  if len(scores_per_dim[d]) > 1 else float('nan') for d in DIMS]
        print("  mean   " + "  ".join(f"{m:>11.2f}" for m in means))
        print("  std    " + "  ".join(f"{s:>11.3f}" for s in stds))
