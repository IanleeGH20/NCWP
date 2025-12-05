import argparse

from .pipeline import build_overlap_labels


def build_parser():
    p = argparse.ArgumentParser(
        description="Build traditional metric-based labels (TF-IDF, Jaccard, BM25)."
    )
    p.add_argument("--input", "-i", required=True, help="Path to corpus1000.csv (must have id,text)")
    p.add_argument("--output", "-o", required=True, help="Path to traditinal_labeled_corpus1000.csv")
    p.add_argument("--tfidf-thr", type=float, default=0.40, help="TF-IDF cosine threshold (default: 0.40)")
    p.add_argument("--jaccard-thr", type=float, default=0.30, help="Jaccard threshold (default: 0.30)")
    p.add_argument("--bm25-percentile", type=float, default=90.0, help="BM25 percentile cutoff (default: 90)")
    p.add_argument("--max-top", type=int, default=12, help="Max neighbors per metric (default: 12)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    build_overlap_labels(
        input_csv=args.input,
        output_csv=args.output,
        tfidf_threshold=args.tfidf_thr,
        jaccard_threshold=args.jaccard_thr,
        bm25_percentile=args.bm25_percentile,
        max_top=args.max_top,
    )


if __name__ == "__main__":
    main()


