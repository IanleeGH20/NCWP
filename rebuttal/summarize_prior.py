#!/usr/bin/env python3
"""Print already-obtained results that the round-2 document should cite.

Round 2 only needed to re-run what was actually missing. Several comparison
columns the document wants already exist from earlier runs (round-1 train+test
counterparts, the ABTT baseline table, the STS test-only sweep) plus the
published paper tables. This prints those numbers so the document quotes them
verbatim instead of re-running the experiments.
"""
import os

import pandas as pd

OUT1 = "/workspace/NCWP/rebuttal_outputs"
OUT2 = "/workspace/NCWP/rebuttal_outputs_v2"
MODELS = ["qwen-4b", "qwen-8b", "llama-8b"]
MAIN_R = {"qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448],
          "llama-8b": [32, 64, 128, 256]}

# Published values transcribed from ncwp_emnlp_final_revised (1).tex
PAPER_STS_NCWP = {  # Tables 5/6/7 (appendix full sweeps)
    "qwen-4b": {40: 50.90, 80: 53.37, 160: 54.81, 320: 55.79},
    "qwen-8b": {56: 55.42, 112: 56.85, 224: 57.49, 448: 58.37},
    "llama-8b": {32: 57.86, 64: 60.82, 128: 62.10, 256: 62.99},
}
PAPER_QUORA = {  # Table 2 (qwen-4b main) + appendix Quora tables
    "qwen-4b": {40: (32.9, 31.5, 47.9), 80: (41.4, 42.0, 57.0),
                160: (49.9, 50.8, 61.7), 320: (57.5, 58.1, 64.2)},
    "qwen-8b": {56: (36.4, None, 52.26), 112: (44.5, None, 59.05),
                224: (51.9, None, 63.79), 448: (58.2, None, 65.97)},
    "llama-8b": {32: (None, None, None), 64: (41.5, None, 55.75),
                 128: (49.0, None, 62.34), 256: (56.1, None, 67.19)},
}


def show(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def layer_selection_traintest():
    show("[A] Round-1 layer-selection, train+test 7,128 (table_layer_selection.csv)")
    df = pd.read_csv(f"{OUT2}/table_layer_selection.csv")
    for m in MODELS:
        d = df[df.model_nickname == m]
        b = d[d.method.str.contains("_Base")]
        parts = [f"{r.layer_name}/{r.pooling}={r.metric_value:.2f}" for _, r in b.iterrows()]
        print(f"\n  [{m}] Base: " + "  ".join(parts))
        for _, r in d[d.method.str.contains("NCWP") & (d.r == MAIN_R[m][-1])].iterrows():
            print(f"      r={r.r:>4} {r.method:42s} {r.metric_value:.2f}±{r.std_if_available:.2f}")


def echo_traintest():
    show("[B] Round-1 Echo whitening, train+test 7,128 (table_echo_whitening.csv)")
    df = pd.read_csv(f"{OUT1}/table_echo_whitening.csv")
    for m in MODELS:
        d = df[df.model_nickname == m]
        raw = d[d.method == "Echo"].metric_value.iloc[0]
        zf = d[d.method == "Echo_ZCA_full"].metric_value.iloc[0]
        print(f"\n  [{m}] Echo_raw={raw:.2f}  Echo_ZCA_full={zf:.2f}")
        for r in MAIN_R[m]:
            p = d[(d.method == "Echo_PCA_whitening") & (d.r == r)].metric_value.iloc[0]
            z = d[(d.method == "Echo_ZCA_only") & (d.r == r)].metric_value.iloc[0]
            n = d[(d.method == "NCWP_on_Echo") & (d.r == r)]
            print(f"      r={r:>4} PCA={p:>6.2f} ZCA_only={z:>6.2f} "
                  f"NCWP={n.metric_value.iloc[0]:>6.2f}±{n.std_if_available.iloc[0]:.2f}")


def abtt_table():
    show("[C] ABTT baseline table (table_abtt_baseline.csv) — STS train+test / Quora")
    df = pd.read_csv(f"{OUT1}/table_abtt_baseline.csv")
    for ds in ["stsb", "quora"]:
        print(f"\n  -- {ds} --")
        for m in MODELS:
            d = df[(df.model_nickname == m) & (df.dataset == ds)]
            if not len(d):
                print(f"    [{m}] (없음)")
                continue
            base = d[d.method == "Base_mean_pool"]
            af = d[d.method == "ABTT_full"]
            print(f"    [{m}] Base={base.metric_value.iloc[0]:.2f} "
                  f"ABTT_full={af.metric_value.iloc[0]:.2f} (fit={int(base.fit_size.iloc[0])})")
            for r in MAIN_R[m]:
                cells = []
                for meth in ["PCA_whitening", "ABTT_PCA_r", "ZCA_shrink", "NCWP_mean_pool"]:
                    s = d[(d.method == meth) & (d.r == r)]
                    cells.append(f"{meth}={s.metric_value.iloc[0]:.2f}" if len(s)
                                 else f"{meth}=n/a")
                print(f"      r={r:>4} " + "  ".join(cells))


def sts_testonly():
    show("[D] Round-1 STS test-only 1,379 (table_stsb_testonly_1379.csv)")
    df = pd.read_csv(f"{OUT1}/table_stsb_testonly_1379.csv")
    for m in MODELS:
        d = df[df.model_nickname == m]
        base = d[d.method == "Base_mean_pool"]
        print(f"\n  [{m}] Base={base.metric_value.iloc[0]:.2f}")
        for r in MAIN_R[m]:
            cells = []
            for meth in ["PCA_whitening", "Soft_whitening", "ZCA_only", "NCWP"]:
                s = d[(d.method == meth) & (d.r == r)]
                cells.append(f"{meth}={s.metric_value.iloc[0]:.2f}" if len(s) else f"{meth}=n/a")
            print(f"      r={r:>4} " + "  ".join(cells))


def vs_paper_sts():
    show("[E] code_actual (round-2 audit) vs PUBLISHED paper main STS values")
    df = pd.read_csv(f"{OUT2}/table_ncwp_config_audit.csv")
    for m in MODELS:
        print(f"\n  [{m}]  {'r':>5} {'paper':>8} {'code_actual':>13} {'delta':>8}")
        for r in MAIN_R[m]:
            s = df[(df.model_nickname == m) & (df.config_name == "code_actual") & (df.r == r)]
            p = PAPER_STS_NCWP[m][r]
            v = s.metric_value.iloc[0]
            print(f"         {r:>5} {p:>8.2f} {v:>13.2f} {v - p:>+8.2f}")


def vs_paper_quora():
    show("[F] round-2 corpus_sample vs PUBLISHED paper Quora values")
    df = pd.read_csv(f"{OUT1}/table_quora_corpus_sample_mainN.csv")
    for m in MODELS:
        print(f"\n  [{m}]  {'r':>5} | {'PCA paper/new':>16} | {'ZCA paper/new':>16} "
              f"| {'NCWP paper/new':>18} | {'ΔNCWP':>7}")
        d = df[df.model_nickname == m]
        for r in MAIN_R[m]:
            pp, pz, pn = PAPER_QUORA[m].get(r, (None, None, None))
            np_ = d[(d.method == "PCA_whitening") & (d.r == r)].metric_value.iloc[0]
            nz = d[(d.method == "ZCA_only") & (d.r == r)].metric_value.iloc[0]
            nn = d[(d.method == "NCWP") & (d.r == r)].metric_value.iloc[0]
            f = lambda x: f"{x:.2f}" if x is not None else "  n/a"
            delta = f"{nn - pn:+.2f}" if pn is not None else "  n/a"
            print(f"         {r:>5} | {f(pp):>7} / {np_:>6.2f} | {f(pz):>7} / {nz:>6.2f} "
                  f"| {f(pn):>8} / {nn:>6.2f} | {delta:>7}")


if __name__ == "__main__":
    layer_selection_traintest()
    echo_traintest()
    abtt_table()
    sts_testonly()
    vs_paper_sts()
    vs_paper_quora()
