from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .utils import simple_word_tokens, to_semicolon_str


def build_bm25_index(corpus_tokens: List[List[str]]) -> Dict:
    """
    Build a minimal BM25 index: postings with term frequencies, doc lengths, df, avgdl.
    """
    N = len(corpus_tokens)
    postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    doc_lengths = np.array([len(tokens) for tokens in corpus_tokens], dtype=np.float64)
    for i, toks in enumerate(corpus_tokens):
        # term frequency per doc
        tf: Dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            postings[t].append((i, c))
    df = {t: len(postings[t]) for t in postings}
    avgdl = float(doc_lengths.mean()) if N > 0 else 0.0
    return {"N": N, "postings": postings, "df": df, "doc_lengths": doc_lengths, "avgdl": avgdl}


def bm25_idf(N: int, df_t: int) -> float:
    # Probabilistic IDF with smoothing to avoid negative IDF
    return float(np.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0))


def bm25_scores_for_query(query_tokens: List[str], index: Dict, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    N = index["N"]
    postings = index["postings"]
    df = index["df"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avgdl"]

    scores = np.zeros(N, dtype=np.float64)
    if N == 0:
        return scores
    # term frequency in query (rarely used for BM25 here; we treat query as set)
    seen = set()
    for term in query_tokens:
        if term in seen:
            continue
        seen.add(term)
        df_t = df.get(term, 0)
        if df_t == 0:
            continue
        idf = bm25_idf(N, df_t)
        for doc_idx, tf in postings[term]:
            denom = tf + k1 * (1.0 - b + b * (doc_lengths[doc_idx] / (avgdl + 1e-12)))
            scores[doc_idx] += idf * ((tf * (k1 + 1.0)) / (denom + 1e-12))
    return scores


def compute_bm25_tops(
    texts: Sequence[str],
    ids: np.ndarray,
    percentile: float = 90.0,
    max_top: int = 12,
) -> List[str]:
    """
    BM25 tops per sentence using percentile cutoff, then take top-K.
    Returns semicolon-separated ID strings per row.
    """
    tokens_list: List[List[str]] = [simple_word_tokens(t) for t in texts]
    index = build_bm25_index(tokens_list)
    N = len(tokens_list)
    out: List[str] = []
    for i in range(N):
        q = tokens_list[i]
        scores = bm25_scores_for_query(q, index)
        if N > i:
            scores[i] = -np.inf  # exclude self
        valid = scores[np.isfinite(scores)]
        if valid.size == 0:
            out.append("")
            continue
        cutoff = np.percentile(valid, percentile)
        idxs = np.where(scores >= cutoff)[0]
        if idxs.size == 0:
            out.append("")
            continue
        order = np.argsort(-scores[idxs])
        sel = idxs[order][:max_top]
        out.append(to_semicolon_str(ids[sel]))
    return out


