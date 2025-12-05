# decoder_lm

Train simple decoder-only (Transformer) language models at multiple embedding dimensions.
The input is a labeled corpus (CSV with `text` column). The output is a set of model
weight files `base_dim_{d}.pt` for the requested dimensions.

This project is extracted from the original notebook and packaged for CLI usage.

## Install

Prerequisite: Dependencies are unified at the repository root. If you've already installed them,
you do not need to install anything here again. For reference:

```bash
# From the GIT root (run once per environment)
pip install -r requirements.txt
```

## Usage

### Basic Training
Train multiple dimensions (powers of two by default):

```bash
python -m decoder_lm.code.cli_train \
  --input ../../labeled_corpus1000.csv \
  --save-dir ../../weights \
  --steps 500
```

### Advanced Training (Specific Dims & Tokenizer)

```bash
python -m decoder_lm.code.cli_train \
  --input ../../labeled_corpus1000.csv \
  --save-dir ../../weights \
  --dims 2 4 8 16 32 64 128 256 512 1024 2048 \
  --steps 500 \
  --tok-type bpe \
  --vocab-size 10000
```

## CLI Options

```bash
python -m decoder_lm.code.cli_train --help
```

### Main Options
- `--input / -i`: Path to input CSV (labeled_corpus_1000.csv)
- `--save-dir / -s`: Base directory to save model weights
- `--dims`: List of dimensions to train (e.g. `64 128 256`). Defaults to powers of two.
- `--steps`: Training steps per dimension (default: 500)
- `--device`: Device to use (`auto`, `cuda`, `cpu`)

### Tokenization
- `--tok-type`: Tokenizer type: `ws` (whitespace, default), `bpe`, or `wordpiece`
- `--vocab-size`: Vocab size for subword tokenizers (default: 30000)
- `--min-freq`: Min frequency for subword training (default: 2)
- `--lowercase`: Lowercase text before tokenization

### Model Architecture
- `--n-layer`: Number of transformer layers (default: 4)
- `--n-head`: Number of attention heads (default: 8)
- `--block-size`: Context length (default: 64)

### Advanced / Checkpointing
- `--save-iter-interval`: If > 0, save intermediate checkpoints every N steps.
- `--save-dir-template`: Template for save path (e.g., `{base}/{tok}`).
- `--eval-every`: Validate every N steps (default: 50)
- `--patience`: Early stopping patience (default: 5)

## Files

```
decoder_lm/
  code/
    model.py      # Minimal GPT (decoder-only)
    data.py       # Tokenizer (Whitepace/BPE/WordPiece) and dataset builders
    train.py      # Training loop and saving logic
    cli_train.py  # CLI entrypoint
  README.md
```

## Notes

- This is a lightweight educational implementation, not optimized for large-scale training.
- The saved `vocab.json` will be reused by the NCWP evaluation step.
