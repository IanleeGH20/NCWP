# ncwp_eval

Evaluate NCWP (PCA-whitening based) on embeddings generated from a decoder-only model.
This project loads a base model (dimension configurable, default 512), encodes the corpus,
trains a PCA-whitening projection for target dimensions, and outputs comparison results and plots.

This corresponds to the second part extracted from the original notebook.

## Install

Prerequisite: Dependencies are unified at the repository root. If you've already installed them at
the root, you can skip installation here. For reference:

```bash
# From the GIT root (run once per environment)
pip install -r requirements.txt
```

Note: This project depends on the `decoder_lm` package to load the base model and vocabulary.
Make sure both are installed in the same environment (or use `python -m` to run modules directly).

## Usage

### 1. Basic Evaluation (Single Base Dimension)

After training a base model (e.g., dim 512), evaluate it against various target dimensions:

```bash
python -m ncwp_eval.code.cli_ncwp \
  --corpus ../../labeled_corpus1000.csv \
  --weights-dir ../../weights \
  --base-dim 512 \
  --dims 4 8 16 32 64 128 256 \
  --out-dir ../../ncwp_results \
  --progress-bar
```

### 2. Sweep Modes (Advanced)

Evaluate multiple checkpoints or base dimensions automatically.

**Evaluate all checkpoints for a specific base dim:**
```bash
python -m ncwp_eval.code.cli_ncwp \
  --corpus ../../labeled_corpus1000.csv \
  --weights-dir ../../weights \
  --base-dim 512 \
  --weights-mode checkpoints \
  --dims-auto-mode half_range \
  --out-dir ../../ncwp_results/checkpoints_sweep
```

**Evaluate all trained base dimensions found in the directory:**
```bash
python -m ncwp_eval.code.cli_ncwp \
  --corpus ../../labeled_corpus1000.csv \
  --weights-dir ../../weights \
  --weights-mode base-dims \
  --dims-auto-mode powers_of_two \
  --out-dir ../../ncwp_results/dims_sweep
```

## CLI Options

```bash
python -m ncwp_eval.code.cli_ncwp --help
```

### Main Options
- `--corpus / -c`: Path to labeled corpus CSV
- `--weights-dir / -w`: Directory containing model weights
- `--out-dir / -o`: Output directory for results and plots
- `--device`: Device to use (`auto`, `cuda`, `cpu`)

### Evaluation Modes
- `--weights-mode`:
  - `best_only` (default): Load only the final/best model for `--base-dim`.
  - `checkpoints`: Load all intermediate checkpoints for `--base-dim`.
  - `base-dims`: Automatically find and evaluate all `base_dim_*.pt` models in the folder.

### Dimension Settings
- `--base-dim`: The starting dimension to load (required for `best_only` and `checkpoints` modes).
- `--dims`: List of target dimensions to evaluate (e.g. `32 64 128`).
- `--dims-auto-mode`: Automatically generate target dims if `--dims` is missing.
  - `half_range`: 2, 4, ..., base_dim/2
  - `powers_of_two`: 2, 4, ..., base_dim

### Visualization & Misc
- `--show-plots`: Display plots inline (useful for Jupyter).
- `--progress-bar`: Show tqdm progress bar during evaluation.
- `--save-projectors`: Save learned PCA/NCWP matrices.

## Outputs

- `comparison.csv`: Detailed metrics (mAP for Base vs NCWP) for all runs.
- `map_vs_dim.png`: mAP vs Dimension plot.
- `runs/`: (In sweep mode) Contains individual plots for each checkpoint/base-dim run.

## Files

```
ncwp_eval/
  code/
    projector.py   # PCA-whitening (NCWP) and learnable projector scaffold
    metrics.py     # Retrieval metrics and positive-set builder
    eval.py        # End-to-end evaluation logic
    cli_ncwp.py    # CLI entry
    graphs.py      # Plotting utilities
  ncwp_eval.ipynb  # Notebook for interactive evaluation
  README.md
```
