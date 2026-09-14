#!/usr/bin/env python3
"""Audit NCWP trainer hyperparameters vs rebuttal claims (K=256, QR=10, regularizers).

No GPU. Reads ncwp_ref defaults, inspects which rebuttal runners call ncwp_ref vs
common.train_ncwp, and compares key STS/Quora numbers against paper prose.
Output: rebuttal_outputs_v2/table_ncwp_hparam_audit.csv + ncwp_hparam_audit.md
"""
import inspect
import json
import os

import pandas as pd

from rebuttal.common import MODEL_CONFIGS, OUT_ROOT
from rebuttal.ncwp_ref import train_ncwp_variant

OUT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"

PAPER_STS_BASE = {"qwen-4b": 36.48, "qwen-8b": 37.02, "llama-8b": 46.35}
REBUTTAL_CLAIM = dict(
    topk_negatives=256,
    retraction_interval=10,
    lambda_cov=0.05,
    lambda_orth=0.02,
    refine_knn_rounds=1,
    k_neighbors_sts=40,
    k_neighbors_beir=10,
    temperature=0.12,
    trainer="rebuttal/ncwp_ref.py:train_ncwp_variant (verbatim from run_ablation.py)",
)

RUNNER_TRAINER = [
    ("run_abtt.py", "ncwp_ref", "full"),
    ("run_quora_corpus_sample.py", "ncwp_ref", "full"),
    ("run_echo_whitening.py", "ncwp_ref", "full"),
    ("run_layer_ncwp_from_cache.py", "ncwp_ref", "full"),
    ("run_prompt_baselines.py", "ncwp_ref", "full"),
    ("run_pooling_variants.py", "ncwp_ref", "full"),
    ("run_positive_ablation.py", "ncwp_ref", "full"),
    ("run_regularizer_ablation.py", "ncwp_ref", "variants"),
    ("run_sts_testonly.py", "common.train_ncwp", "WARNING: reimplementation, not ncwp_ref"),
]


def sig_defaults():
    sig = inspect.signature(train_ncwp_variant)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}


def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def main():
    os.makedirs(OUT_V2, exist_ok=True)
    defaults = sig_defaults()

    hparam_rows = [dict(setting=k, claimed=REBUTTAL_CLAIM.get(k, ""), code_default=v)
                   for k, v in defaults.items()]
    hparam_rows.extend([
        dict(setting="trainer_source", claimed=REBUTTAL_CLAIM["trainer"], code_default="ncwp_ref.py"),
        dict(setting="BEIR_k_neighbors", claimed=10, code_default="train_ncwp_variant default k_neighbors=10"),
        dict(setting="STS_k_neighbors", claimed=40, code_default="not in ncwp_ref defaults; BEIR uses k=10"),
    ])
    pd.DataFrame(hparam_rows).to_csv(os.path.join(OUT_V2, "table_ncwp_hparam_audit.csv"), index=False)

    runner_rows = [dict(runner=r, trainer=t, notes=n) for r, t, n in RUNNER_TRAINER]
    pd.DataFrame(runner_rows).to_csv(os.path.join(OUT_V2, "table_ncwp_runner_trainer_map.csv"), index=False)

    # Compare Base numbers: cached main vs layer-selection re-encode
    cmp_rows = []
    sts_test = load_csv(f"{OUT_ROOT}/table_stsb_testonly_1379.csv")
    layer = load_csv(f"{OUT_V2}/table_layer_selection.csv")
    pooling = load_csv(f"{OUT_V2}/table_pooling_variants_testonly1379.csv")

    for model in MODEL_CONFIGS:
        paper_base = PAPER_STS_BASE.get(model)
        cached_test = None
        layer_tt = None
        pool_test = None
        if sts_test is not None:
            m = sts_test[(sts_test.model_nickname == model) & (sts_test.method == "Base_mean_pool")]
            if len(m):
                cached_test = float(m.iloc[0].metric_value)
        if layer is not None:
            m = layer[(layer.model_nickname == model) & (layer.method == "LayerSelect_last_mean_Base")]
            if len(m):
                layer_tt = float(m.iloc[0].metric_value)
        if pooling is not None:
            m = pooling[(pooling.model_nickname == model) & (pooling.method == "Base_mean_pool")]
            if len(m):
                pool_test = float(m.iloc[0].metric_value)
        cmp_rows.append(dict(
            model=model,
            paper_train_test_base=paper_base,
            layer_selection_train_test_base=layer_tt,
            pooling_testonly1379_base=pool_test,
            sts_testonly_cached_base=cached_test,
            layer_vs_paper_delta=(round(layer_tt - paper_base, 2) if layer_tt and paper_base else None),
            note=("layer-selection re-encodes via PromptEmbedder (Plain); "
                  "37.96 vs 37.02 for qwen-8b is eval-split + encoding source mismatch"),
        ))
    pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT_V2, "table_ncwp_base_number_check.csv"), index=False)

    md = os.path.join(OUT_V2, "ncwp_hparam_audit.md")
    with open(md, "w") as f:
        f.write("# NCWP hyperparameter audit\n\n")
        f.write("## ncwp_ref defaults (from `inspect.signature`)\n\n")
        for k, v in defaults.items():
            f.write(f"- `{k}` = {v}\n")
        f.write("\n## Rebuttal claimed settings\n\n")
        for k, v in REBUTTAL_CLAIM.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Match verdict\n\n")
        match = (
            defaults.get("topk_negatives") == 256
            and defaults.get("retraction_interval") == 10
            and defaults.get("lambda_cov") == 0.05
            and defaults.get("lambda_orth") == 0.02
        )
        f.write(f"**ncwp_ref defaults match rebuttal K=256 / QR=10 / regularizers ON: {match}**\n\n")
        f.write("Regularizers are **enabled** (not removed). N6 ablation shows ~0 individual "
                "effect when QR=10 is on.\n\n")
        f.write("## Runners NOT using ncwp_ref\n\n")
        f.write("- `run_sts_testonly.py` uses `common.train_ncwp` (reimplementation). "
                "Do not use for NCWP claims; re-run with ncwp_ref if needed.\n\n")
        f.write("## Base number discrepancies (qwen-8b 37.96 vs 37.02)\n\n")
        f.write("- Paper 37.02: cached `sts_ablation_cache` embeddings, train+test eval.\n")
        f.write("- Layer-selection 37.96: fresh Plain PromptEmbedder encode, train+test eval.\n")
        f.write("- Pooling/test-only 39.27: cached main embeddings, test-only 1379 eval.\n")
        f.write("- Fix: run layer-selection with `--eval_split test` using cached layer embeddings.\n")

    print(f"Saved audit -> {OUT_V2}/")
    print(f"  table_ncwp_hparam_audit.csv")
    print(f"  table_ncwp_runner_trainer_map.csv")
    print(f"  table_ncwp_base_number_check.csv")
    print(f"  ncwp_hparam_audit.md")
    print(f"\nncwp_ref match rebuttal K/QR/reg: {match}")


if __name__ == "__main__":
    main()
