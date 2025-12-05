import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PAD = "<PAD>"
UNK = "<UNK>"


class SimpleVocab:
    """
    Simple whitespace tokenizer + vocabulary.
    This is intentionally minimal to keep the project lightweight.
    """
    def __init__(self, tokens: List[str]):
        uniq = [PAD, UNK]
        seen = set(uniq)
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        self.itos = uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build_from_corpus(cls, texts: List[str], min_freq: int = 1) -> "SimpleVocab":
        from collections import Counter
        cnt = Counter()
        for s in texts:
            for t in s.split():
                cnt[t] += 1
        tokens = []
        for t, c in cnt.items():
            if c >= min_freq:
                tokens.append(t)
        tokens.sort()
        return cls(tokens)

    def encode(self, text: str) -> List[int]:
        ids = []
        for t in text.split():
            ids.append(self.stoi.get(t, self.stoi[UNK]))
        return ids

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SimpleVocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vocab = cls([])
        vocab.itos = obj["itos"]
        vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
        return vocab

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


class _BaseTokenizerVocab:
    """Interface-compatible wrapper around a subword tokenizer (BPE/WordPiece)."""
    def __init__(self, tokenizer, pad_token: str, unk_token: str, kind: str):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.kind = kind  # "bpe" or "wordpiece"

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    @property
    def pad_id(self) -> int:
        pid = self.tokenizer.token_to_id(self.pad_token)
        return 0 if pid is None else int(pid)

    @property
    def unk_id(self) -> int:
        uid = self.tokenizer.token_to_id(self.unk_token)
        return 1 if uid is None else int(uid)

    @property
    def vocab_size(self) -> int:
        try:
            return int(self.tokenizer.get_vocab_size(with_added_tokens=True))
        except Exception:
            return int(self.tokenizer.get_vocab_size())

    def save(self, path: str) -> None:
        """Save a compatible vocab.json for downstream (with itos), and dump tokenizer JSON file."""
        import os, json as _json
        out_dir = os.path.dirname(path)
        os.makedirs(out_dir, exist_ok=True)
        # Dump tokenizer json
        tok_path = os.path.join(out_dir, f"tokenizer_{self.kind}.json")
        try:
            self.tokenizer.save(tok_path)
        except Exception:
            # Fallback: serialize via to_str if available
            try:
                data = self.tokenizer.to_str()
                with open(tok_path, "w", encoding="utf-8") as f:
                    f.write(data)
            except Exception:
                pass
        # Build itos for compatibility
        try:
            vocab_dict = self.tokenizer.get_vocab()
            size = max(vocab_dict.values()) + 1 if vocab_dict else self.vocab_size
            itos = [None] * size
            for tok, idx in vocab_dict.items():
                if 0 <= idx < size:
                    itos[idx] = tok
            # Fill any Nones with a placeholder to maintain length
            for i in range(len(itos)):
                if itos[i] is None:
                    itos[i] = f"<unk_{i}>"
            with open(path, "w", encoding="utf-8") as f:
                _json.dump({"itos": itos, "tokenizer_json": tok_path, "type": self.kind}, f, ensure_ascii=False)
        except Exception:
            # Last resort: write minimal stub
            with open(path, "w", encoding="utf-8") as f:
                _json.dump({"type": self.kind, "tokenizer_json": tok_path, "vocab_size": self.vocab_size}, f, ensure_ascii=False)


def _train_bpe_tokenizer(texts: List[str], vocab_size: int, min_freq: int, lowercase: bool) -> "_BaseTokenizerVocab":
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import NFKC, Lowercase as TKLower, Sequence
    except Exception as e:
        raise RuntimeError("HuggingFace 'tokenizers' package is required for BPE. Ensure it's installed.") from e
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()
    if lowercase:
        tokenizer.normalizer = Sequence([NFKC(), TKLower()])
    else:
        tokenizer.normalizer = NFKC()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[PAD, UNK])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return _BaseTokenizerVocab(tokenizer, pad_token=PAD, unk_token=UNK, kind="bpe")


def _train_wordpiece_tokenizer(texts: List[str], vocab_size: int, min_freq: int, lowercase: bool) -> "_BaseTokenizerVocab":
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import NFKC, Lowercase as TKLower, Sequence
    except Exception as e:
        raise RuntimeError("HuggingFace 'tokenizers' package is required for WordPiece. Ensure it's installed.") from e
    tokenizer = Tokenizer(WordPiece(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()
    if lowercase:
        tokenizer.normalizer = Sequence([NFKC(), TKLower()])
    else:
        tokenizer.normalizer = NFKC()
    trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[PAD, UNK])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return _BaseTokenizerVocab(tokenizer, pad_token=PAD, unk_token=UNK, kind="wordpiece")


@dataclass
class LMConfig:
    block_size: int = 64


class LMDataset(Dataset):
    def __init__(self, texts: List[str], vocab: SimpleVocab, cfg: LMConfig):
        self.vocab = vocab
        self.cfg = cfg
        self.examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for s in texts:
            ids = vocab.encode(s)
            if not ids:
                continue
            # Chop into blocks
            for i in range(0, len(ids), cfg.block_size):
                chunk = ids[i:i + cfg.block_size]
                x = torch.full((cfg.block_size,), vocab.pad_id, dtype=torch.long)
                attn = torch.zeros((cfg.block_size,), dtype=torch.bool)
                L = min(len(chunk), cfg.block_size)
                x[:L] = torch.tensor(chunk[:L], dtype=torch.long)
                attn[:L] = True
                y = x.clone()
                y[:-1] = x[1:]  # next token prediction
                y[-1] = vocab.pad_id
                self.examples.append((x, y, attn))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def load_texts(input_csv: str) -> List[str]:
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    return df["text"].astype(str).tolist()


def build_dataloaders(
    texts: List[str],
    block_size: int,
    batch_size: int,
    seed: int = 42,
    val_ratio: float = 0.1,
    tok_type: str = "ws",            # "ws" | "bpe" | "wordpiece"
    vocab_size: int = 30000,
    min_freq: int = 2,
    lowercase: bool = False,
) -> Tuple[SimpleVocab, DataLoader, DataLoader]:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    n = len(texts)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = set(idx[:n_val])

    train_texts = [texts[i] for i in range(n) if i not in val_idx]
    val_texts = [texts[i] for i in range(n) if i in val_idx]

    if tok_type == "ws":
        vocab = SimpleVocab.build_from_corpus(train_texts, min_freq=1)
    elif tok_type == "bpe":
        vocab = _train_bpe_tokenizer(train_texts, vocab_size=vocab_size, min_freq=min_freq, lowercase=lowercase)
    elif tok_type == "wordpiece":
        vocab = _train_wordpiece_tokenizer(train_texts, vocab_size=vocab_size, min_freq=min_freq, lowercase=lowercase)
    else:
        raise ValueError(f"Unknown tok_type: {tok_type}. Use one of ['ws','bpe','wordpiece'].")
    cfg = LMConfig(block_size=block_size)
    ds_tr = LMDataset(train_texts, vocab, cfg)
    ds_va = LMDataset(val_texts, vocab, cfg)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)
    return vocab, dl_tr, dl_va


