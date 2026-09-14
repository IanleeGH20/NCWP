#!/usr/bin/env python3
"""v2 spec §8 (Review 3, Reproducibility=2): reproducibility clarification.

Emits an exact backbone table + dataset provenance table + a checklist markdown
stating all data is public and how to reproduce. No GPU. Reads MODEL_CONFIGS so
the model facts stay in sync with the code. Output: rebuttal_outputs_v2/.
"""
import json
import os

import pandas as pd

from rebuttal.common import MODEL_CONFIGS

OUT_ROOT_V2 = "/workspace/NCWP/rebuttal_outputs_v2"

# STS-B validation fit size is shared; Quora fit size is per-model (from MODEL_CONFIGS)
STSB_FIT = 2910

DATASETS = [
    ("STSBenchmark", "HF mteb/stsbenchmark-sts (MTEB)", "test", 1379, "-", "-", STSB_FIT),
    ("STS12", "MTEB STS12", "test", None, "-", "-", STSB_FIT),
    ("STS13", "MTEB STS13", "test", None, "-", "-", STSB_FIT),
    ("STS14", "MTEB STS14", "test", None, "-", "-", STSB_FIT),
    ("STS15", "MTEB STS15", "test", None, "-", "-", STSB_FIT),
    ("STS16", "MTEB STS16", "test", None, "-", "-", STSB_FIT),
    ("SICK-R", "MTEB SICK-R", "test", None, "-", "-", STSB_FIT),
    ("BEIR-Quora", "HF mteb/quora (BEIR)", "test", None, 522931, 10000, "1000-2000 corpus (per backbone)"),
    ("CQADupStack", "HF BeIR/cqadupstack (BEIR)", "test", None, "per-subforum", "per-subforum", "per-subforum"),
]


def model_table():
    rows = []
    for nick, cfg in MODEL_CONFIGS.items():
        rows.append(dict(
            nickname=nick, hf_model_id=cfg["hf_model_id"], hidden_dim=cfg["hidden_dim"],
            layer=cfg["layer"], pooling=cfg["pooling"], max_length=cfg["max_length"],
            dtype=cfg["dtype"], fit_size_stsb=STSB_FIT, fit_size_quora=cfg["quora_fit_size"],
            note=("Qwen-4B = Qwen1.5-4B, Qwen-8B = Qwen2-7B (NOT Qwen2.5; fix citation)"
                  if nick.startswith("qwen") else ""),
        ))
    return rows


def dataset_table():
    cols = ["dataset", "source", "split", "n_test_examples", "n_corpus_items", "n_queries", "fit_subset_size"]
    return [dict(zip(cols, d)) for d in DATASETS], cols


def main():
    os.makedirs(OUT_ROOT_V2, exist_ok=True)
    mt = model_table()
    dt, dcols = dataset_table()
    pd.DataFrame(mt).to_csv(os.path.join(OUT_ROOT_V2, "table_reproducibility_details.csv"), index=False)
    pd.DataFrame(dt)[dcols].to_csv(os.path.join(OUT_ROOT_V2, "table_reproducibility_datasets.csv"), index=False)

    md = os.path.join(OUT_ROOT_V2, "reproducibility_checklist.md")
    with open(md, "w") as f:
        f.write("# Reproducibility checklist (Review 3: Reproducibility=2)\n\n")
        f.write("## Public data statement\n\n")
        f.write("All datasets used in the main and additional experiments are **public** "
                "(MTEB / BEIR via HuggingFace `datasets`). No institution-private or "
                "consortium-only data is used. No labels are used for fitting "
                "PCA/ZCA/ABTT/NCWP — only sentence text and a fixed random corpus subsample.\n\n")
        f.write("## Exact backbones\n\n")
        f.write("| Nickname | HF model ID | D | Layer | Pooling | max_len | dtype | fit(STS-B) | fit(Quora) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in mt:
            f.write(f"| {r['nickname']} | `{r['hf_model_id']}` | {r['hidden_dim']} | {r['layer']} | "
                    f"{r['pooling']} | {r['max_length']} | {r['dtype']} | {r['fit_size_stsb']} | {r['fit_size_quora']} |\n")
        f.write("\n> **Citation fix:** the paper cites Qwen2.5 for Qwen-4B/8B, but the released "
                "Qwen2.5 line has no 4B/8B. The actual checkpoints are **Qwen1.5-4B** (D=2560) and "
                "**Qwen2-7B** (D=3584). Update the citation and clarify the '8B' nickname (7B params).\n\n")
        f.write("## Fitting subset sizes (consolidate in §4.1)\n\n")
        f.write(f"- STS-B: {STSB_FIT} validation sentences.\n")
        f.write("- Quora: 2,000 corpus items (Qwen-4B); 1,000 (Llama-8B, Qwen-8B).\n\n")
        f.write("## Reproduction steps\n\n")
        f.write("1. Install deps (`requirements.txt`).\n2. Download public datasets (MTEB/BEIR via `datasets`).\n"
                "3. Extract frozen LLM hidden states (mean-pool last layer, L2-norm).\n"
                "4. Fit PCA/ZCA/ABTT/NCWP on the fixed corpus subsample (seed released).\n"
                "5. Evaluate STS (Spearman) / BEIR (nDCG@10).\n6. Regenerate tables.\n\n")
        f.write("## Release with code\n\n")
        f.write("- Fit-subset indices/seeds: `artifacts/fit_indices/*.json`.\n")
        f.write("- NCWP trainer: `run_ablation.py:train_ncwp_variant` (rebuttal mirror: `rebuttal/ncwp_ref.py`).\n")
    print(f"Saved reproducibility artifacts -> {OUT_ROOT_V2}/")
    print(f"  table_reproducibility_details.csv ({len(mt)} backbones)")
    print(f"  table_reproducibility_datasets.csv ({len(dt)} datasets)")
    print(f"  reproducibility_checklist.md")


if __name__ == "__main__":
    main()
