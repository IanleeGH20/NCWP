import argparse
from .main import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label_generation",
        description="Generate neighbor-based labels for a corpus CSV.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV (expects columns: id, text).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output labeled CSV.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.80,
        help="Cosine similarity threshold to include neighbors (default: 0.80).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding encoding (default: 64).",
    )
    parser.add_argument(
        "--prefer-cuda",
        action="store_true",
        help="Prefer CUDA if available (falls back to CPU automatically).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Tokenizer max_length for HuggingFace encoders (default: 512).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama", "qwen4b", "qwen8b"],
        choices=["llama", "qwen4b", "qwen8b"],
        help="Models to run. Default: llama qwen4b qwen8b",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        sim_threshold=args.threshold,
        batch_size=args.batch_size,
        prefer_cuda=args.prefer_cuda,
        max_len=args.max_len,
        models=args.models,
    )


if __name__ == "__main__":
    main()


