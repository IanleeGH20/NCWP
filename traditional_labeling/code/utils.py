import os
import re
from typing import Iterable, List

import numpy as np
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def read_corpus(input_csv: str, id_col: str = "id", text_col: str = "text") -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"CSV must contain '{id_col}' and '{text_col}' columns")
    return df


def write_corpus(df: pd.DataFrame, output_csv: str) -> None:
    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)


def to_semicolon_str(values: Iterable[int]) -> str:
    return ";".join(str(int(v)) for v in values)


def parse_semicolon_ids(cell) -> List[int]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    out: List[int] = []
    for t in s.split(";"):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(int(float(t)))
        except Exception:
            continue
    return out


def simple_word_tokens(text: str) -> List[str]:
    # Basic English-ish tokenizer: letters, digits, apostrophes
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


