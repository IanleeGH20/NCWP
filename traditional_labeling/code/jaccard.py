from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .utils import simple_word_tokens, to_semicolon_str


def compute_jaccard_tops(
    texts: Sequence[str],
    ids: np.ndarray,
    threshold: float = 0.30,
    max_top: int = 12,
) -> List[str]:
    """
    Word-level Jaccard similarity.
    Returns semicolon-separated ID strings per row.
    """
    tokens_list: List[set[str]] = [set(simple_word_tokens(t)) for t in texts]
    N = len(tokens_list)
    out: List[str] = []
    for i in range(N):
        sims = []
        A = tokens_list[i]
        for j in range(N):
            if i == j:
                continue
            B = tokens_list[j]
            if not A and not B:
                s = 0.0
            else:
                inter = len(A & B)
                union = len(A | B)
                s = inter / union if union > 0 else 0.0
            if s >= threshold:
                sims.append((j, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        sel = [ids[j] for j, _ in sims[:max_top]]
        out.append(to_semicolon_str(sel))
    return out


