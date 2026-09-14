#!/usr/bin/env python3
"""Compare code_actual vs camera_ready (Eq 6/7 removed) on the camera-ready protocols.

Protocols: STS-B test-only 1,379 and BEIR-Quora corpus_sample fitting, 3 backbones
x 3 seeds. Also checks that the Quora code_actual rows reproduce the standalone
corpus_sample table, and prints per-seed values so a near-zero delta can be told
apart from the config silently not being applied.
"""
import ast

import pandas as pd

OUT1 = "/workspace/NCWP/rebuttal_outputs"
OUT2 = "/workspace/NCWP/rebuttal_outputs_v2"
MODELS = ["qwen-4b", "qwen-8b", "llama-8b"]
MAIN_R = {"qwen-4b": [40, 80, 160, 320], "qwen-8b": [56, 112, 224, 448],
          "llama-8b": [32, 64, 128, 256]}


def as_list(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def pick(df, model, r, config):
    s = df[(df.model_nickname == model) & (df.r == r) & (df.config_name == config)]
    assert len(s) == 1, f"{model} r={r} {config}: {len(s)} rows"
    return s.iloc[0]


def section(title, csv, metric):
    df = pd.read_csv(csv)
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    print(f"  {'백본':10s} {'r':>5} {'code_actual':>20} {'camera_ready':>20} {'Δ':>7}  per-seed 동일?")
    deltas = []
    for m in MODELS:
        for r in MAIN_R[m]:
            a, b = pick(df, m, r, "code_actual"), pick(df, m, r, "camera_ready")
            d = b.metric_value - a.metric_value
            deltas.append(d)
            pa, pb = as_list(a.per_seed), as_list(b.per_seed)
            same = "예 (설정 미적용 의심)" if pa == pb else "아니오 (정상)"
            print(f"  {m:10s} {r:>5} {a.metric_value:>13.2f}±{a.std_if_available:.2f} "
                  f"{b.metric_value:>13.2f}±{b.std_if_available:.2f} {d:>+7.2f}  {same}")
    s = pd.Series(deltas)
    print(f"\n  Δ(camera_ready − code_actual): 평균 {s.mean():+.3f}  "
          f"최대 |{s.abs().max():.2f}|  |Δ|>1.0 셀 {(s.abs() > 1.0).sum()}/{len(s)}  ({metric})")
    return df


def per_seed_detail(df, title):
    print(f"\n  -- per-seed 원값 ({title}) --")
    for m in MODELS:
        r = MAIN_R[m][1]
        a, b = pick(df, m, r, "code_actual"), pick(df, m, r, "camera_ready")
        print(f"    {m:10s} r={r:>4}  code_actual={as_list(a.per_seed)}  "
              f"camera_ready={as_list(b.per_seed)}")


def cross_check_quora(df):
    ref = pd.read_csv(f"{OUT1}/table_quora_corpus_sample_mainN.csv")
    print(f"\n  -- 교차 확인: Quora code_actual vs 기존 table_quora_corpus_sample_mainN --")
    worst = 0.0
    for m in MODELS:
        for r in MAIN_R[m]:
            a = pick(df, m, r, "code_actual").metric_value
            s = ref[(ref.model_nickname == m) & (ref.method == "NCWP") & (ref.r == r)]
            d = a - float(s.metric_value.iloc[0])
            worst = max(worst, abs(d))
    print(f"    12개 셀 최대 차이 = {worst:.4f} "
          f"→ {'동일 (러너 동등성 확인)' if worst < 0.005 else '차이 있음'}")


if __name__ == "__main__":
    sts = section("STS-B test-only 1,379 (Spearman x100)",
                  f"{OUT2}/table_ncwp_config_audit_testonly.csv", "Spearman")
    per_seed_detail(sts, "STS test-only")
    quora = section("BEIR-Quora corpus_sample fitting (nDCG@10)",
                    f"{OUT2}/table_ncwp_config_audit_quora.csv", "nDCG@10")
    per_seed_detail(quora, "Quora corpus_sample")
    cross_check_quora(quora)
