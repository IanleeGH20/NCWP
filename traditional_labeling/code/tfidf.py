from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .utils import to_semicolon_str


def compute_tfidf_tops(
    texts: Sequence[str],
    ids: np.ndarray,
    threshold: float = 0.40,
    max_top: int = 12,
) -> List[str]:
    """
    Char n-gram (2..5) TF-IDF cosine similarity.
    Returns semicolon-separated ID strings per row.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        norm="l2",
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    sim = linear_kernel(X, X)  # cosine for L2-normalized TF-IDF
    np.fill_diagonal(sim, 0.0)
    N = sim.shape[0]
    out: List[str] = []
    for i in range(N):
        row = sim[i]
        idxs = np.where(row >= threshold)[0]
        if idxs.size == 0:
            out.append("")
            continue
        order = np.argsort(-row[idxs])
        sel = idxs[order][:max_top]
        out.append(to_semicolon_str(ids[sel]))
    return out


