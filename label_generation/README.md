# label_generation

Generate neighbor-based labels from a text corpus using sentence embeddings.  
This tool reads an input CSV with columns `id` and `text`, computes nearest
neighbors for selected embedding models, and writes a labeled CSV that includes:

- `nvidia/llama-embed-nemotron-8b`: semicolon-separated neighbor IDs
- `qwen3-4b`: semicolon-separated neighbor IDs (optional)
- `qwen3-8b`: semicolon-separated neighbor IDs (optional)
- `neighbors_union`: union of neighbor IDs across available models
- `neighbors_union_count`: length of `neighbors_union`

## Quickstart

1) Create and activate a virtual environment (recommended), or reuse the one you created at the repository root.

```bash
python -m venv .venv
. .venv/bin/activate
```

2) Dependencies: If you've already installed unified dependencies at the GIT root, skip this step. Otherwise:

```bash
# From the GIT root (run once per environment)
pip install -r requirements.txt
```

3) Run the pipeline:

```bash
python -m label_generation.code.cli \
  --input ../../corpus1000.csv \
  --output ../../labeled_corpus1000.csv \
  --threshold 0.80 \
  --models llama qwen4b qwen8b
```

Notes:
- Large models (e.g., Qwen3-Embedding-8B) may require significant disk and RAM.
- If you have a GPU and enough free memory, add `--prefer-cuda` to prefer CUDA.
- You can run a subset of models, e.g. `--models llama` to only use Nemotron-8B.

## CSV Format

Input (`corpus1000.csv`):
```
id,text
0,This is a sentence.
1,Another example.
...
```

Output (`labeled_corpus1000.csv`):
```
id,text,nvidia/llama-embed-nemotron-8b,qwen3-4b,qwen3-8b,neighbors_union,neighbors_union_count
0,This is a sentence.,12;45;...,,,...,3
1,Another example.,...,...,...,...,5
```

## CLI Options

```bash
python -m label_generation.code.cli --help
```

- `--input / -i`: Path to input CSV with `id,text`
- `--output / -o`: Output CSV path
- `--threshold / -t`: Cosine similarity threshold (default: 0.80)
- `--batch-size`: Unused placeholder (kept for parity), default 64
- `--prefer-cuda`: Prefer GPU if available
- `--max-len`: Tokenizer max_length (default: 512)
- `--models`: Any of `llama qwen4b qwen8b` (default uses all)

## Project Structure

```
label_generation/
  code/
    cli.py           # CLI entrypoint
    embeddings.py    # Encoders (Nemotron, HF mean pooling)
    main.py          # Pipeline
    utils.py         # IO, device, neighbor computation helpers
  README.md
```


