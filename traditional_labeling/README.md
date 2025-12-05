# traditional_labeling

Generate “traditional” labels using TF-IDF cosine, word-level Jaccard, and BM25.
This project is extracted from the original notebook and packaged as a CLI.

Input: `corpus1000.csv` (must contain columns: `id`, `text`)  
Output: `traditinal_labeled_corpus1000.csv`

The output CSV includes:
- `tf_idf_tops`: semicolon-separated neighbor IDs from TF-IDF (char n-grams 2..5), cosine ≥ threshold
- `jaccard_tops`: semicolon-separated neighbor IDs from Jaccard, ≥ threshold
- `bm25_tops`: semicolon-separated neighbor IDs from BM25 (query-specific percentile cutoff)
- `overlap2`: IDs that appear in at least 2 of the above lists
- `overlap3`: IDs that appear in all 3 lists

## Install

Prerequisite: Dependencies are unified at the repository root. If you've already installed them at
the root, you can skip installation here. For reference:

```bash
# From the GIT root (run once per environment)
pip install -r requirements.txt
```

## Usage

```bash
python -m traditional_labeling.code.cli \
  --input ../../corpus1000.csv \
  --output ../../traditinal_labeled_corpus1000.csv \
  --tfidf-thr 0.40 \
  --jaccard-thr 0.30 \
  --bm25-percentile 90 \
  --max-top 12
```

Notes:
- TF-IDF uses character n-grams (2..5) and cosine similarity.
- Jaccard uses a simple English-ish tokenizer (letters/digits/apostrophes).
- BM25 uses a lightweight custom implementation with a per-query percentile cutoff.

## Files

```
traditional_labeling/
  code/
    utils.py     # CSV IO, tokenization, helpers
    tfidf.py     # TF-IDF tops
    jaccard.py   # Jaccard tops
    bm25.py      # BM25 tops
    pipeline.py  # Orchestration + overlaps
    cli.py       # CLI entry
  README.md
```


