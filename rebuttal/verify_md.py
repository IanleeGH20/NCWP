#!/usr/bin/env python3
"""Cross-check every number typed into ADDITIONAL_RESULTS_round2.md against the CSVs.

The document quotes values from several tables by hand; this re-reads each source
CSV and asserts the quoted figure matches, so a transcription slip cannot survive
into the rebuttal text.
"""
import re
import sys

import pandas as pd

MD = "/workspace/NCWP/rebuttal_outputs/ADDITIONAL_RESULTS_round2.md"
OUT1 = "/workspace/NCWP/rebuttal_outputs"
OUT2 = "/workspace/NCWP/rebuttal_outputs_v2"
TOL = 0.005

# (source csv, filter dict, r, expected value quoted in the md)
CHECKS = []


def add(csv, model, method, r, expected, dataset=None, config=None, layer=None, pool=None):
    CHECKS.append(dict(csv=csv, model=model, method=method, r=r, expected=expected,
                       dataset=dataset, config=config, layer=layer, pool=pool))


# --- §1 Quora corpus_sample (new) ---
Q = f"{OUT1}/table_quora_corpus_sample_mainN.csv"
for model, rows in {
    "qwen-4b": [(40, 33.14, 32.93, 44.97), (80, 41.43, 41.85, 57.28),
                (160, 49.33, 49.89, 62.46), (320, 56.26, 56.72, 64.86)],
    "qwen-8b": [(56, 36.11, 34.88, 52.22), (112, 44.13, 43.00, 59.77),
                (224, 51.06, 50.99, 63.64), (448, 57.30, 57.25, 65.57)],
    "llama-8b": [(32, 32.85, 30.48, 43.74), (64, 41.50, 40.15, 57.21),
                 (128, 48.69, 47.65, 63.52), (256, 55.20, 54.84, 67.05)],
}.items():
    for r, pca, zca, ncwp in rows:
        add(Q, model, "PCA_whitening", r, pca)
        add(Q, model, "ZCA_only", r, zca)
        add(Q, model, "NCWP", r, ncwp)

# --- §1-b query_sample from the ABTT table ---
A = f"{OUT1}/table_abtt_baseline.csv"
for model, rows in {
    "qwen-4b": [(40, 44.41), (80, 56.81), (160, 62.74), (320, 65.33)],
    "qwen-8b": [(56, 52.65), (112, 60.68), (224, 64.15), (448, 66.11)],
    "llama-8b": [(32, 43.51), (64, 57.75), (128, 63.81)],
}.items():
    for r, v in rows:
        add(A, model, "NCWP_mean_pool", r, v, dataset="quora")

# --- §2 config audit ---
C = f"{OUT2}/table_ncwp_config_audit.csv"
for model, rows in {
    "qwen-4b": [(40, 50.84, 51.75, 52.05), (80, 53.17, 53.30, 53.73),
                (160, 55.18, 55.50, 55.49), (320, 56.39, 56.57, 56.65)],
    "qwen-8b": [(56, 55.25, 55.39, 55.44), (112, 56.71, 56.89, 57.11),
                (224, 57.69, 57.95, 57.92), (448, 58.53, 58.51, 58.63)],
    "llama-8b": [(32, 56.90, 57.63, 57.85), (64, 59.53, 59.75, 60.04),
                 (128, 61.06, 61.72, 61.78), (256, 62.23, 62.40, 62.37)],
}.items():
    for r, pa, ca, cr in rows:
        add(C, model, "NCWP_mean_pool", r, pa, config="paper_appendix")
        add(C, model, "NCWP_mean_pool", r, ca, config="code_actual")
        add(C, model, "NCWP_mean_pool", r, cr, config="camera_ready")

# --- §2-c final config on the camera-ready protocols ---
CT = f"{OUT2}/table_ncwp_config_audit_testonly.csv"
for model, rows in {
    "qwen-4b": [(40, 48.50, 48.48), (80, 52.65, 53.09),
                (160, 54.14, 53.26), (320, 54.87, 54.94)],
    "qwen-8b": [(56, 54.69, 54.42), (112, 56.06, 56.13),
                (224, 57.33, 57.15), (448, 58.72, 58.30)],
    "llama-8b": [(32, 55.75, 55.79), (64, 59.05, 59.28),
                 (128, 60.56, 60.78), (256, 62.12, 62.47)],
}.items():
    for r, ca, cr in rows:
        add(CT, model, "NCWP_mean_pool", r, ca, config="code_actual")
        add(CT, model, "NCWP_mean_pool", r, cr, config="camera_ready")

CQ = f"{OUT2}/table_ncwp_config_audit_quora.csv"
for model, rows in {
    "qwen-4b": [(40, 44.97, 44.97), (80, 57.28, 57.28),
                (160, 62.46, 62.46), (320, 64.86, 64.86)],
    "qwen-8b": [(56, 52.22, 52.22), (112, 59.77, 59.77),
                (224, 63.64, 63.63), (448, 65.57, 65.57)],
    "llama-8b": [(32, 43.74, 43.74), (64, 57.21, 57.21),
                 (128, 63.52, 63.52), (256, 67.05, 67.05)],
}.items():
    for r, ca, cr in rows:
        add(CQ, model, "NCWP_mean_pool", r, ca, config="code_actual")
        add(CQ, model, "NCWP_mean_pool", r, cr, config="camera_ready")

# --- §3 layer selection, both splits ---
for csv, rows in {
    f"{OUT2}/table_layer_selection.csv": {
        "qwen-4b": [(320, 56.65, 58.21)], "qwen-8b": [(448, 58.62, 60.73)],
        "llama-8b": [(256, 62.23, 63.27)]},
    f"{OUT2}/table_layer_selection_testonly1379.csv": {
        "qwen-4b": [(320, 55.29, 56.60)], "qwen-8b": [(448, 58.04, 60.50)],
        "llama-8b": [(256, 61.94, 63.36)]},
}.items():
    for model, rr in rows.items():
        for r, last, second in rr:
            add(csv, model, "LayerSelect_last_mean_NCWP", r, last)
            add(csv, model, "LayerSelect_second_last_mean_NCWP", r, second)

# --- §4 Echo whitening, both splits ---
for csv, rows in {
    f"{OUT1}/table_echo_whitening.csv": {
        "qwen-4b": [(40, 46.24, 52.41, 57.91), (320, 61.68, 61.23, 62.35)],
        "qwen-8b": [(56, 49.65, 55.57, 62.65), (448, 63.13, 63.16, 65.00)],
        "llama-8b": [(32, 47.52, 58.63, 62.23), (256, 62.96, 64.91, 67.22)]},
    f"{OUT1}/table_echo_whitening_testonly1379.csv": {
        "qwen-4b": [(40, 47.30, 54.90, 57.87), (320, 62.48, 61.79, 63.17)],
        "qwen-8b": [(56, 53.56, 59.92, 63.92), (448, 65.48, 65.29, 66.96)],
        "llama-8b": [(32, 51.27, 59.29, 62.94), (256, 64.51, 67.10, 68.03)]},
}.items():
    for model, rr in rows.items():
        for r, pca, zca, ncwp in rr:
            add(csv, model, "Echo_PCA_whitening", r, pca)
            add(csv, model, "Echo_ZCA_only", r, zca)
            add(csv, model, "NCWP_on_Echo", r, ncwp)

# --- §5 BERT-flow ---
B = f"{OUT2}/table_bertflow.csv"
for model, rows in {
    "qwen-4b": [(40, 50.22), (80, 53.20), (160, 55.10), (320, 56.07)],
    "qwen-8b": [(56, 53.56), (112, 55.80), (224, 56.98), (448, 57.58)],
    "llama-8b": [(32, 53.13), (64, 56.70), (128, 58.69), (256, 59.86)],
}.items():
    for r, v in rows:
        add(B, model, "BERTflow_PCA_r", r, v)
BQ = f"{OUT2}/table_bertflow_quora.csv"
for model, rows in {
    "qwen-4b": [(40, 36.29), (80, 51.79), (160, 59.99), (320, 63.28)],
    "qwen-8b": [(56, 47.66), (112, 57.85), (224, 62.15), (448, 64.08)],
    "llama-8b": [(32, 37.14), (64, 53.02), (128, 60.86), (256, 64.30)],
}.items():
    for r, v in rows:
        add(BQ, model, "BERTflow_PCA_r", r, v)

# --- §5 closed-form context from the ABTT table (STS) ---
for model, rows in {
    "qwen-4b": [(40, 40.79, 37.23, 41.95, 51.15), (320, 53.36, 45.03, 52.87, 56.44)],
    "qwen-8b": [(56, 43.96, 40.02, 46.70, 55.46), (448, 55.58, 45.55, 55.63, 58.75)],
    "llama-8b": [(32, 41.11, 42.52, 51.19, 57.58), (256, 56.84, 49.98, 59.72, 62.76)],
}.items():
    for r, pca, ap, zca, ncwp in rows:
        add(A, model, "PCA_whitening", r, pca, dataset="stsb")
        add(A, model, "ABTT_PCA_r", r, ap, dataset="stsb")
        add(A, model, "ZCA_shrink", r, zca, dataset="stsb")
        add(A, model, "NCWP_mean_pool", r, ncwp, dataset="stsb")


def check_memory_bank():
    """§2-d A/B table uses its own schema (model / r / memory_bank)."""
    df = pd.read_csv(f"{OUT2}/table_memory_bank_check.csv")
    expected = {
        ("qwen-4b", 20): (47.00, 47.66), ("qwen-4b", 80): (53.20, 53.79),
        ("qwen-4b", 320): (55.85, 55.99), ("qwen-8b", 20): (50.74, 51.48),
        ("qwen-8b", 80): (56.06, 57.00), ("qwen-8b", 320): (57.90, 58.05),
        ("llama-8b", 20): (54.75, 55.50), ("llama-8b", 80): (61.14, 61.76),
        ("llama-8b", 320): (63.25, 63.19),
    }
    fails, checked = [], 0
    for (model, r), (on, off) in expected.items():
        for state, exp in (("ON", on), ("OFF", off)):
            s = df[(df.model == model) & (df.r == r) & (df.memory_bank == state)]
            if len(s) != 1:
                fails.append(f"membank {model} r={r} {state}: {len(s)} rows")
                continue
            got = float(s.spearman.iloc[0])
            checked += 1
            if abs(got - exp) > TOL:
                fails.append(f"membank {model} r={r} {state}: md={exp:.2f} csv={got:.4f}")
    on_rows = df[df.memory_bank == "ON"]
    if not (on_rows.frac_from_bank_mean.between(0.8789, 0.8841).all()):
        fails.append("membank frac_from_bank_mean outside the quoted 87.89-88.41% range")
    if not ((on_rows.pool_width_max == 4608).all() and (on_rows.bank_rows_max == 4096).all()):
        fails.append("membank pool_width/bank_rows do not match the quoted 4608/4096")
    return checked, fails


def main():
    cache, fails, checked = {}, [], 0
    for c in CHECKS:
        if c["csv"] not in cache:
            cache[c["csv"]] = pd.read_csv(c["csv"])
        df = cache[c["csv"]]
        sel = df[(df.model_nickname == c["model"]) & (df.method == c["method"])
                 & (df.r == c["r"])]
        if c["dataset"]:
            sel = sel[sel.dataset == c["dataset"]]
        if c["config"]:
            sel = sel[sel.config_name == c["config"]]
        label = (f"{c['csv'].split('/')[-1]} {c['model']} {c['method']} r={c['r']}"
                 + (f" [{c['config']}]" if c["config"] else "")
                 + (f" [{c['dataset']}]" if c["dataset"] else ""))
        if len(sel) != 1:
            fails.append(f"{label}: {len(sel)} rows matched")
            continue
        got = float(sel.metric_value.iloc[0])
        checked += 1
        if abs(got - c["expected"]) > TOL:
            fails.append(f"{label}: md={c['expected']:.2f} csv={got:.4f}")

    mb_checked, mb_fails = check_memory_bank()
    checked += mb_checked
    fails += mb_fails

    md = open(MD, encoding="utf-8").read()
    n_tables = md.count("\n|---")
    print(f"checked {checked}/{len(CHECKS) + mb_checked} quoted values against CSVs")
    print(f"md: {len(md.splitlines())} lines, {n_tables} tables")
    if fails:
        print(f"\n{len(fails)} MISMATCH:")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print("all quoted values match the source CSVs")


if __name__ == "__main__":
    main()
