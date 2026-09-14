#!/usr/bin/env python3
"""Print the comparison numbers for the round-2 rebuttal experiments.

Reads the round-2 result tables and emits the exact figures needed for the
summary document, so no number is transcribed by hand from a log (the Quora
runner's stdout PCA column was mis-indexed; the CSVs are authoritative).
"""
import os

import pandas as pd

OUT1 = "/workspace/NCWP/rebuttal_outputs"
OUT2 = "/workspace/NCWP/rebuttal_outputs_v2"
MODELS = ["qwen-4b", "qwen-8b", "llama-8b"]
MAIN_R = {"qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448],
          "llama-8b": [32, 64, 128, 256]}


def show(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def quora_corpus_sample():
    show("1. Quora corpus_sample @ main per-model N (nDCG@10)")
    df = pd.read_csv(f"{OUT1}/table_quora_corpus_sample_mainN.csv")
    for m in MODELS:
        d = df[df.model_nickname == m]
        base = d[d.method == "Base_mean_pool"]
        n = int(base.fit_size.iloc[0])
        print(f"\n[{m}] fit N={n}  Base(full D)={base.metric_value.iloc[0]:.2f}")
        print(f"  {'r':>5} {'PCA':>7} {'ZCA':>7} {'Soft':>7} {'Rand':>7} {'LPP':>7} {'NCWP':>8}")
        for r in MAIN_R[m]:
            row = {}
            for meth in ["PCA_whitening", "ZCA_only", "Soft_whitening", "Random", "LPP", "NCWP"]:
                sel = d[(d.method == meth) & (d.r == r)]
                row[meth] = sel.metric_value.iloc[0] if len(sel) else float("nan")
            std = d[(d.method == "NCWP") & (d.r == r)].std_if_available.iloc[0]
            print(f"  {r:>5} {row['PCA_whitening']:>7.2f} {row['ZCA_only']:>7.2f} "
                  f"{row['Soft_whitening']:>7.2f} {row['Random']:>7.2f} {row['LPP']:>7.2f} "
                  f"{row['NCWP']:>7.2f}±{std:.2f}")

    old = f"{OUT1}/table_quora_corpus_sample_N1000.csv"
    if os.path.exists(old):
        do = pd.read_csv(old)
        print("\n  [delta vs previous uniform-N=1000 run, NCWP only]")
        for m in MODELS:
            for r in MAIN_R[m]:
                a = df[(df.model_nickname == m) & (df.method == "NCWP") & (df.r == r)]
                b = do[(do.model_nickname == m) & (do.method == "NCWP") & (do.r == r)]
                if len(a) and len(b):
                    print(f"    {m:9s} r={r:>4} mainN={a.metric_value.iloc[0]:.2f} "
                          f"N1000={b.metric_value.iloc[0]:.2f} "
                          f"delta={a.metric_value.iloc[0] - b.metric_value.iloc[0]:+.2f}")


def config_audit():
    show("2. NCWP config audit — STS-B train+test (Spearman x100)")
    df = pd.read_csv(f"{OUT2}/table_ncwp_config_audit.csv")
    for m in MODELS:
        d = df[df.model_nickname == m]
        print(f"\n[{m}]  {'r':>5} {'paper_appendix':>15} {'code_actual':>13} {'camera_ready':>13} "
              f"{'ca-pa':>7} {'cr-ca':>7}")
        for r in MAIN_R[m]:
            v = {}
            for c in ["paper_appendix", "code_actual", "camera_ready"]:
                sel = d[(d.config_name == c) & (d.r == r)]
                v[c] = (sel.metric_value.iloc[0], sel.std_if_available.iloc[0])
            print(f"        {r:>5} {v['paper_appendix'][0]:>9.2f}±{v['paper_appendix'][1]:.2f} "
                  f"{v['code_actual'][0]:>7.2f}±{v['code_actual'][1]:.2f} "
                  f"{v['camera_ready'][0]:>7.2f}±{v['camera_ready'][1]:.2f} "
                  f"{v['code_actual'][0] - v['paper_appendix'][0]:>+7.2f} "
                  f"{v['camera_ready'][0] - v['code_actual'][0]:>+7.2f}")
    print("\n  hyperparameters recorded per row:")
    cols = ["config_name", "hard_neg_K", "qr_interval", "lambda_cov", "lambda_orth",
            "memory_bank_size", "refine_knn_rounds", "k_neighbors", "temperature"]
    print(df[cols].drop_duplicates().to_string(index=False))


def bertflow():
    show("5. BERT-flow vs NCWP — STS-B train+test (Spearman x100)")
    bf = pd.read_csv(f"{OUT2}/table_bertflow.csv")
    ab = pd.read_csv(f"{OUT1}/table_abtt_baseline.csv")
    au = pd.read_csv(f"{OUT2}/table_ncwp_config_audit.csv")
    for m in MODELS:
        b = bf[bf.model_nickname == m]
        full = b[b.method == "BERTflow_full"]
        a = ab[(ab.model_nickname == m) & (ab.dataset == "stsb")]
        abtt_full = a[a.method == "ABTT_full"]
        base = a[a.method == "Base_mean_pool"]
        print(f"\n[{m}] Base={base.metric_value.iloc[0] if len(base) else float('nan'):.2f}  "
              f"ABTT_full={abtt_full.metric_value.iloc[0] if len(abtt_full) else float('nan'):.2f}  "
              f"BERTflow_full={full.metric_value.iloc[0]:.2f}±{full.std_if_available.iloc[0]:.2f}")
        print(f"  {'r':>5} {'BERTflow+PCA':>14} {'NCWP(code_actual)':>18} {'NCWP-BERTflow':>14}")
        for r in MAIN_R[m]:
            bfv = b[(b.method == "BERTflow_PCA_r") & (b.r == r)]
            nc = au[(au.model_nickname == m) & (au.config_name == "code_actual") & (au.r == r)]
            if len(bfv) and len(nc):
                print(f"  {r:>5} {bfv.metric_value.iloc[0]:>8.2f}±{bfv.std_if_available.iloc[0]:.2f} "
                      f"{nc.metric_value.iloc[0]:>12.2f}±{nc.std_if_available.iloc[0]:.2f} "
                      f"{nc.metric_value.iloc[0] - bfv.metric_value.iloc[0]:>+14.2f}")

    qf = f"{OUT2}/table_bertflow_quora.csv"
    if os.path.exists(qf):
        q = pd.read_csv(qf)
        qn = pd.read_csv(f"{OUT1}/table_quora_corpus_sample_mainN.csv")
        print("\n  -- Quora (nDCG@10) --")
        for m in MODELS:
            b = q[q.model_nickname == m]
            if not len(b):
                continue
            full = b[b.method == "BERTflow_full"]
            print(f"\n[{m}] BERTflow_full={full.metric_value.iloc[0]:.2f}")
            for r in MAIN_R[m]:
                bfv = b[(b.method == "BERTflow_PCA_r") & (b.r == r)]
                nc = qn[(qn.model_nickname == m) & (qn.method == "NCWP") & (qn.r == r)]
                if len(bfv) and len(nc):
                    print(f"  r={r:>5} BERTflow+PCA={bfv.metric_value.iloc[0]:>6.2f} "
                          f"NCWP={nc.metric_value.iloc[0]:>6.2f} "
                          f"delta={nc.metric_value.iloc[0] - bfv.metric_value.iloc[0]:>+6.2f}")


def testonly():
    show("3/4. test-only 1,379 re-evaluations (Spearman x100)")
    ls = pd.read_csv(f"{OUT2}/table_layer_selection_testonly1379.csv")
    print("\n  Layer-selection, Base per layer x pooling:")
    for m in MODELS:
        d = ls[(ls.model_nickname == m) & (ls.method.str.contains("_Base"))]
        parts = [f"{r.layer_name}/{r.pooling}={r.metric_value:.2f}" for _, r in d.iterrows()]
        print(f"    {m:9s} " + "  ".join(parts))
    print("\n  Layer-selection, NCWP at max r (last+mean vs second_last+mean):")
    for m in MODELS:
        d = ls[(ls.model_nickname == m) & (ls.r == MAIN_R[m][-1])]
        for _, r in d[d.method.str.contains("NCWP")].iterrows():
            print(f"    {m:9s} r={r.r:>4} {r.method:42s} {r.metric_value:.2f}±"
                  f"{r.std_if_available:.2f}")

    ew = pd.read_csv(f"{OUT1}/table_echo_whitening_testonly1379.csv")
    print("\n  Echo whitening (test-only 1,379):")
    for m in MODELS:
        d = ew[ew.model_nickname == m]
        zf = d[d.method == "Echo_ZCA_full"]
        raw = d[d.method == "Echo"]
        print(f"\n    [{m}] Echo_raw={raw.metric_value.iloc[0]:.2f} "
              f"Echo_ZCA_full={zf.metric_value.iloc[0]:.2f}")
        for r in MAIN_R[m]:
            p = d[(d.method == "Echo_PCA_whitening") & (d.r == r)]
            z = d[(d.method == "Echo_ZCA_only") & (d.r == r)]
            n = d[(d.method == "NCWP_on_Echo") & (d.r == r)]
            print(f"      r={r:>4} PCA={p.metric_value.iloc[0]:>6.2f} "
                  f"ZCA_only={z.metric_value.iloc[0]:>6.2f} "
                  f"NCWP={n.metric_value.iloc[0]:>6.2f}±{n.std_if_available.iloc[0]:.2f}")


if __name__ == "__main__":
    quora_corpus_sample()
    config_audit()
    testonly()
    bertflow()
