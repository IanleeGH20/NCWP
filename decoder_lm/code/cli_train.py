import argparse
import os
import torch

from .data import build_dataloaders, load_texts
from .train import train_for_dims


def build_parser():
    p = argparse.ArgumentParser(description="Train decoder-only models at multiple dimensions.")
    p.add_argument("--input", "-i", required=True, help="Path to labeled_corpus_1000.csv")
    p.add_argument("--save-dir", "-s", required=True, help="Base directory to save model weights")
    p.add_argument(
        "--save-dir-template",
        type=str,
        default="{base}",
        help="Template for final save directory. Use placeholders {base} and {tok}. "
             "Examples: '{base}' (default), '{base}/{tok}', '/mnt/models/{tok}'.",
    )
    p.add_argument(
        "--use-tokenizer-subdir",
        action="store_true",
        help="If set, append '/{tok}' to --save-dir (ignored if --save-dir-template is provided with {tok}).",
    )
    p.add_argument("--dims", nargs="*", type=int, help="Specific dims to train (e.g., 2 4 8 16 32 64 128 256 512)")
    p.add_argument("--min-dim", type=int, default=2, help="Min dim if --dims not provided")
    p.add_argument("--max-dim", type=int, default=2048, help="Max dim if --dims not provided")
    p.add_argument("--step-dim", type=int, default=0, help="Step size for dims (0=>powers-of-two up to max)")
    p.add_argument("--steps", type=int, default=500, help="Training steps per dim (default: 500)")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--block-size", type=int, default=64, help="Context length")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--n-head", type=int, default=8, help="Num attention heads")
    p.add_argument("--n-layer", type=int, default=4, help="Num transformer layers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # Device selection
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use: auto (default, picks CUDA if available), cuda, or cpu",
    )
    p.add_argument("--prefer-cuda", action="store_true", help="(Deprecated) Prefer CUDA if available")
    # Training control
    p.add_argument("--eval-every", type=int, default=50, help="Validate every N steps (default: 50)")
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience in validation checks (default: 5)")
    # Checkpointing
    p.add_argument(
        "--save-iter-interval",
        type=int,
        default=0,
        help="If > 0, additionally save checkpoints every N training steps as base_dim_{D}_step_{S}.pt",
    )
    # Tokenization
    p.add_argument(
        "--tok-type",
        choices=["ws", "bpe", "wordpiece"],
        default="ws",
        help="Tokenizer type: ws (whitespace), bpe, or wordpiece (default: ws)",
    )
    p.add_argument("--vocab-size", type=int, default=30000, help="Vocab size for BPE/WordPiece (default: 30000)")
    p.add_argument("--min-freq", type=int, default=2, help="Min frequency for subword training (default: 2)")
    p.add_argument("--lowercase", action="store_true", help="Lowercase text before tokenization (BPE/WordPiece)")
    return p


def compute_dims(args) -> list[int]:
    if args.dims:
        return sorted(set([int(d) for d in args.dims if d >= 2]))
    if args.step_dim and args.step_dim > 0:
        return list(range(max(2, args.min_dim), max(args.min_dim, args.max_dim) + 1, args.step_dim))
    # default to powers of two
    dims = []
    d = max(2, args.min_dim)
    while d <= args.max_dim:
        dims.append(d)
        d *= 2
    return dims


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Back-compat: --prefer-cuda overrides --device
    if args.prefer_cuda:
        args.device = "cuda"
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[Warn] CUDA requested but not available. Falling back to CPU.")
    else:
        device = "cpu"
    print(f"[Info] device: {device}")

    texts = load_texts(args.input)
    vocab, dl_tr, dl_va = build_dataloaders(
        texts=texts,
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=0.1,
        tok_type=getattr(args, "tok_type", "ws"),
        vocab_size=getattr(args, "vocab_size", 30000),
        min_freq=getattr(args, "min_freq", 2),
        lowercase=getattr(args, "lowercase", False),
    )
    dims = compute_dims(args)
    print(f"[Info] dims: {dims}")
    # Resolve final save directory possibly including tokenizer-specific subdir
    base_dir = args.save_dir
    eff_save_dir = base_dir
    try:
        # If template includes placeholders, use it
        if "{tok}" in args.save_dir_template or "{base}" in args.save_dir_template:
            eff_save_dir = args.save_dir_template.format(base=base_dir, tel=None, tok=args.tok_type)
        elif args.use_tokenizer_subdir:
            eff_save_dir = os.path.join(base_dir, args.tok_type)
    except Exception:
        # Fallback to base dir on any formatting error
        eff_save_dir = base_dir
    os.makedirs(eff_save_dir, exist_ok=True)
    print(f"[Info] saving weights to: {eff_save_dir}")

    train_for_dims(
        vocab=vocab,
        dl_tr=dl_tr,
        dl_va=dl_va,
        dims=dims,
        steps=args.steps,
        lr=args.lr,
        device=device,
        block_size=args.block_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
        save_dir=eff_save_dir,
        eval_every=args.eval_every,
        patience=args.patience,
        save_iter_interval=args.save_iter_interval,
    )


if __name__ == "__main__":
    main()


