<h1 align="center">NCWP — Neighbor-Contrastive Whitening Projection</h1>

<p align="center">
  <b>Make frozen decoder-only LLM hidden states retrieval-usable — no fine-tuning, no labeled pairs, just one post-hoc linear projection.</b>
</p>

<p align="center">
  <img alt="Accepted at EMNLP 2026" src="https://img.shields.io/badge/EMNLP-2026-b31b1b">
  <img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="PyTorch 2.3" src="https://img.shields.io/badge/PyTorch-2.3-ee4c2c">
</p>

<p align="center">
  🎉 <b>Accepted to EMNLP 2026</b> &nbsp;•&nbsp; 📄 Paper: <b><i>coming soon</i></b>
</p>

<p align="center">
  <img src="figures/figure1_quora_example.png" alt="NCWP lifts the gold paraphrase from rank #1,034 to rank #1 on BEIR-Quora" width="840">
</p>

<p align="center">
  <sub><i>One post-hoc linear projection at <b>32× compression</b> lifts the gold paraphrase from rank <b>#1,034 → #1</b> on BEIR-Quora — Qwen1.5-4B frozen, no labeled pairs.</i></sub>
</p>

---

**NCWP** is a label-free, post-hoc alignment method for **symmetric
sentence-similarity retrieval** on frozen decoder-only LLMs. It whitens the
embedding space (ZCA-shrink) and learns a single orthogonal projection with a
**neighbor-contrastive** objective (kNN-mined positives + hard negatives),
yielding compact embeddings that **outperform PCA / ZCA / Soft-whitening /
Random / LPP at the same target dimension** on STS and BEIR retrieval — with the
backbone fully frozen and no gradient updates.

Across three decoder-only backbones (Llama-3.1-8B, Qwen1.5-4B, Qwen2-7B), NCWP
even **exceeds the raw Base hidden states** at aggressive compression — e.g.
**+23.8 Spearman** at 8× on STSBenchmark and **+13.6 nDCG@10** at 32× on
BEIR-Quora over Qwen1.5-4B Base.

---

## 1. What you need to provide

NCWP downloads models/datasets and encodes embeddings locally, so a few things
must be supplied by you:

| Requirement | Details |
|---|---|
| **NVIDIA GPU + CUDA** | Encoding and projection training run on GPU. Rough VRAM per backbone (bf16): Qwen1.5-4B ≈ 10 GB, Qwen2-7B ≈ 18 GB, Llama-3.1-8B ≈ 20 GB. BEIR-Quora retrieval over the 522,931-item corpus needs ≈ 6 GB more. A single 24 GB card is enough. |
| **HuggingFace token** | `export HF_TOKEN=hf_xxx`. Needed to download the backbones. **Llama-3.1-8B is gated** — request access once at its model page (`meta-llama/Meta-Llama-3.1-8B`) with the same HF account. Qwen checkpoints are ungated. |
| **Disk space** | Model weights (~8–16 GB each, cached under `~/.cache/huggingface`) + Base embedding caches under `cache/` (BEIR-Quora `corpus_base.npy` ≈ 5 GB per backbone). |
| **Internet (first run)** | Backbones and MTEB/BEIR datasets are fetched on first use, then cached. |
| **Docker + NVIDIA Container Toolkit** (recommended) | Or Python 3.10 with a CUDA-matched PyTorch for a bare-metal install. |

The paper's random seeds (`{42, 43, 44}`) and STSBenchmark split are already
bundled (`data/stsbenchmark/`); model weights and embedding caches are **not**
versioned (see `.gitignore`) and are produced by the steps below.

---

## 2. Environment — one Docker image runs everything

```bash
# 1) Build the image (installs all Python deps from requirements.txt)
docker build -t ncwp:cuda .

# 2) Start a container: mount this repo at /workspace/NCWP, pass your HF token
docker run -d --name ncwp --gpus all \
    -v "$PWD":/workspace/NCWP \
    -e HF_TOKEN=hf_xxx \
    ncwp:cuda tail -f /dev/null

# 3) Get a shell inside; run everything from the repo root
docker exec -it ncwp bash
cd /workspace/NCWP
```

<details>
<summary>Without Docker</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118   # match your CUDA
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
```

Verified on Python 3.10 with the torch this command currently resolves to
(2.7.1+cu118); any CUDA-matched torch >= 2.3 works.
</details>

All commands below are run from the repository root as Python modules
(`python -m ncwp.<script>`).

---

## 3. Step 0 — build the Base embedding caches (run once)

Every experiment reads pre-computed **Base** embeddings (last layer, mean-pooled,
L2-normalized, bf16). Build them once per backbone:

```bash
# STS only (fast — ~15k sentences per backbone)
python -m ncwp.build_cache --models qwen-4b --datasets sts

# STS + BEIR-Quora (Quora corpus = 522,931 items → slow, needs disk & time)
python -m ncwp.build_cache --models qwen-4b qwen-8b llama-8b --datasets sts quora

# per-layer x per-pooling STS caches (only needed for the layer-selection table)
python -m ncwp.build_cache --models qwen-4b qwen-8b llama-8b --datasets layers

# optional CQADupStack breadth caches
python -m ncwp.build_cache --models qwen-4b llama-8b --datasets cqa
```

Outputs go to `cache/` (git-ignored). This is the only step that loads the LLM
backbones; all fitting/evaluation below reuses these caches.

Rough wall-clock on one A6000 (per backbone): `sts` a few minutes, `layers`
~10–20 minutes, `cqa` ~20 minutes, `quora` 1–3 hours (dominated by encoding the
522k-item corpus). Each experiment runner afterwards takes minutes, except the
Quora sweeps (~10–30 minutes per backbone).

---

## 4. Training — fit an NCWP projector on your own corpus

Learn a projector from any unlabeled text (one item per line), no labels needed:

```bash
python -m ncwp.train \
    --model qwen-4b \
    --corpus my_corpus.txt \
    --dim 80 \
    --out projector_qwen4b_r80.npz
```

This encodes the corpus, fits ZCA-shrink whitening + the neighbor-contrastive
projection, and saves `(μ, whitening∘projection, output-stats)` to a single
`.npz`. Training a projector converges in minutes on one GPU.

---

## 5. Inference — apply a projector to new text

```bash
python -m ncwp.apply \
    --projector projector_qwen4b_r80.npz \
    --input queries.txt \
    --out queries_r80.npy
```

The output is an `(N × r)` float32 array of L2-normalized NCWP embeddings —
cosine similarity is then a drop-in retrieval score, e.g.:

```python
import numpy as np
docs = np.load("docs_r80.npy"); qs = np.load("queries_r80.npy")
scores = qs @ docs.T                    # cosine (already L2-normalized)
top10 = np.argsort(-scores, axis=1)[:, :10]
```

Inference is one matrix multiply + L2 norm per item — identical in cost to
PCA-whitening at the same target dimension.

---

## 6. Reproducing the paper tables

After Step 0, each table is regenerated by one command. NCWP numbers are the
mean ± std over seeds `{42, 43, 44}` (script defaults). Results are written to
`results/`, alongside the reference CSVs shipped in this repo.

**Pass the flags exactly as shown.** The script defaults are deliberately
smaller (one backbone, `train+test` split) for quick checks, so a bare command
runs fine but produces a *narrower* table than the reference CSV.

| Paper table(s) | Command | Reference output |
|---|---|---|
| STS official test — Tables 1, 6–8 | `python -m ncwp.run_sts_testonly --models qwen-4b qwen-8b llama-8b` | `results/table_stsb_testonly_1379.csv` |
| STS Random / LPP rows | `python -m ncwp.run_baselines_testonly` | `results/table_baselines_testonly1379.csv` |
| STS ABTT / ABTT+PCA rows | `python -m ncwp.run_abtt --model qwen-4b qwen-8b llama-8b` | `results/table_abtt_baseline.csv` |
| BEIR-Quora — Tables 2, 9–11 | `python -m ncwp.run_quora_corpus_sample --models qwen-4b qwen-8b llama-8b --basename table_quora_corpus_sample_mainN` | `results/table_quora_corpus_sample_mainN.csv` |
| Anisotropy diagnostics — Table 3 | `python run_t2_anisotropy.py` | `results/t2_anisotropy_results.csv` |
| Component ablation — Table 5 | `python run_sts_ablation_unified.py` | `results/sts_ablation_unified_eval7128.csv` |
| PromptEOL / Echo + NCWP — Table 4 | `python -m ncwp.run_prompt_baselines --model qwen-4b qwen-8b llama-8b --eval_split test` | `results/table_prompt_baseline_stsb_testonly1379.csv` |
| STS12–16 + SICK-R breadth | `python -m ncwp.run_sts_suite --model qwen-4b qwen-8b llama-8b` | `results/table_sts_suite.csv` |
| CQADupStack breadth | `python -m ncwp.run_cqadupstack` | `results/table_cqadupstack.csv` |
| Cross-corpus transfer | `python -m ncwp.run_cross_dataset --model qwen-4b qwen-8b llama-8b` | `results/table_cross_dataset_transfer_*.csv` |
| Pooling comparison | `python -m ncwp.run_pooling_variants --model qwen-4b qwen-8b llama-8b --eval_split test` | `results/table_pooling_variants_testonly1379.csv` |
| Layer selection + NCWP | `python -m ncwp.run_layer_ncwp_from_cache --eval_split test` | `results/table_layer_selection_testonly1379.csv` |
| Echo + whitening | `python -m ncwp.run_echo_whitening --eval_split test` | `results/table_echo_whitening_testonly1379.csv` |
| Positive-source control (50k fit) | `python -m ncwp.run_positive_search --fit_n 50000 --r_list 20 40 80 --k_list 1 5 --basename table_positive_search_max_fit50k` | `results/table_positive_search_max_fit50k.csv` |
| Mined-positive precision | `python -m ncwp.run_mined_precision --model qwen-4b qwen-8b llama-8b` | `results/table_mined_positive_precision.csv` |
| Mined-neighbor score distribution | `python -m ncwp.run_neighbor_score_dist --model qwen-4b qwen-8b llama-8b` | `results/table_sts_mined_neighbor_score_distribution.csv` |
| Soft-regularizer audit | `python -m ncwp.run_regularizer_ablation --dataset both --full` | `results/table_regularizer_ablation.csv` |
| Qualitative failure cases | `python -m ncwp.run_failure_cases` | `results/mined_positive_failure_cases.csv` |

**Run order matters for three of them:**

- *Layer selection* needs the per-layer caches:
  `python -m ncwp.build_cache --models qwen-4b qwen-8b llama-8b --datasets layers` first.
- *Echo + whitening* reads the Echo embeddings produced by the prompt-baseline
  runner, so run `run_prompt_baselines` before `run_echo_whitening`.
- *Cross-corpus transfer* reuses the STS12–16 caches from `run_sts_suite` (and,
  for its retrieval rows, `build_cache --datasets cqa`); run those first.

Every runner fails fast with the exact command to run if a cache is missing.

> **Reproducibility level.** Closed-form baselines and all retrieval (nDCG@10)
> results are **bit-exactly** reproducible; NCWP's learned projection carries
> small GPU floating-point non-determinism that is invisible on nDCG@10 but
> visible on the 1,379-pair STS Spearman, where re-runs stay **within the
> reported per-seed std** — hence the mean ± std reporting. Details:
> [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

> **Tip.** On multi-GPU boxes, run heavy encoding and NCWP training in separate
> processes (`build_cache` → then the runners). Mixing a resident
> multi-billion-parameter backbone with NCWP training in one process can
> intermittently stall the CUDA context; `run_layer_ncwp_from_cache.py` follows
> the decoupled pattern.

---

## 7. Repository layout

```
.
├── Dockerfile, requirements.txt        # reproducible environment
├── README.md, REPRODUCIBILITY.md, LICENSE
├── data/stsbenchmark/                  # bundled STS-B split (valid/train/test)
├── ncwp/                               # core method + CLIs + experiment runners
│   ├── common.py, ncwp_ref.py          # utilities + NCWP trainer
│   ├── build_cache.py                  # Step 0: encode Base embeddings
│   ├── train.py, apply.py              # train a projector / run inference
│   └── run_*.py                        # per-table experiment runners
├── run_t2_anisotropy.py                # anisotropy diagnostics (Table 3)
├── run_sts_ablation_unified.py         # component ablation (Table 5)
├── figures/                            # paper figures
└── results/                            # reference result tables (CSV / JSONL)
```

| nickname   | HF model                       | hidden dim |
|------------|--------------------------------|-----------:|
| `qwen-4b`  | `Qwen/Qwen1.5-4B`              | 2560 |
| `qwen-8b`  | `Qwen/Qwen2-7B`               | 3584 |
| `llama-8b` | `meta-llama/Meta-Llama-3.1-8B` | 4096 |

---

## 8. Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{lee2026ncwp,
  title     = {Label-Free Post-hoc Alignment of Frozen Decoder-Only LLM Hidden
               States for Symmetric Sentence-Similarity Retrieval},
  author    = {Lee, GangHo and Lee, Yong-Gu},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026},
  note      = {To appear}
}
```

The proceedings link and DOI will be added here once available.

---

## 9. License and attribution

- **Code** in this repository: MIT, see [LICENSE](LICENSE).
- **Bundled data** (`data/stsbenchmark/`): the STS Benchmark split, redistributed
  here only to pin the exact fitting/evaluation sentences. It keeps the original
  STS Benchmark terms (CC BY-SA 4.0) — see
  [`data/README.md`](data/README.md) for the source and citation.
- **Datasets fetched at runtime** (BEIR-Quora, CQADupStack, STS12–16, SICK-R via
  MTEB/BEIR) are governed by their own licenses; please cite them if you use
  those results.
- **Backbones** are downloaded from HuggingFace under their own terms, which you
  must accept: the Llama 3.1 Community License for
  `meta-llama/Meta-Llama-3.1-8B`, and the respective Qwen licenses for
  `Qwen/Qwen1.5-4B` and `Qwen/Qwen2-7B`. No model weights are redistributed here.
