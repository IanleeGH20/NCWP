#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_beir_experiment as rbe


MISS_RANK = 10**9


def shorten(text, limit=220):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def topk_indices(scores, k):
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def first_relevant_rank(scores, pos_idx):
    if not pos_idx:
        return MISS_RANK, None
    pos_scores = scores[pos_idx]
    best_offset = int(np.argmax(pos_scores))
    best_idx = pos_idx[best_offset]
    best_score = pos_scores[best_offset]
    rank = int(np.sum(scores > best_score) + 1)
    return rank, best_idx


def analyze_method(query_ids, query_embs, corpus_embs, corpus_id_to_idx, qrels, topk=3, chunk=64):
    corpus_t = corpus_embs.T.astype(np.float32, copy=False)
    results = {}
    for start in range(0, len(query_ids), chunk):
        end = min(start + chunk, len(query_ids))
        score_block = query_embs[start:end] @ corpus_t
        for row_idx, qid in enumerate(query_ids[start:end]):
            rel = qrels.get(qid, {})
            pos_idx = [corpus_id_to_idx[cid] for cid, score in rel.items() if score > 0 and cid in corpus_id_to_idx]
            rank, best_idx = first_relevant_rank(score_block[row_idx], pos_idx)
            top_idx = topk_indices(score_block[row_idx], topk)
            results[qid] = {
                "rank": rank,
                "best_idx": best_idx,
                "top_idx": top_idx.tolist(),
            }
    return results


def render_doc_list(indices, corpus_ids, corpus_texts, positive_set):
    lines = []
    for rank, idx in enumerate(indices, start=1):
        cid = corpus_ids[idx]
        marker = " [relevant]" if cid in positive_set else ""
        lines.append(f"{rank}. `{cid}`{marker} - {shorten(corpus_texts[idx])}")
    return lines


def render_table_row(method, indices, corpus_ids, corpus_texts, positive_set):
    cells = [method]
    for rank, idx in enumerate(indices, start=1):
        cid = corpus_ids[idx]
        marker = " [relevant]" if cid in positive_set else ""
        cells.append(f"`{cid}`{marker}<br>{shorten(corpus_texts[idx], 120)}")
    return "| " + " | ".join(cells) + " |"


def build_candidates(query_ids, query_texts, qrels, corpus_ids, corpus_texts, by_method, num_examples):
    candidates = []
    query_map = dict(zip(query_ids, query_texts))
    for qid in query_ids:
        base_rank = by_method["Base"][qid]["rank"]
        pca_rank = by_method["PCA-White"][qid]["rank"]
        soft_rank = by_method["Soft-White"][qid]["rank"]
        ncwp_rank = by_method["NCWP"][qid]["rank"]

        if base_rank > 10 and pca_rank > 10 and soft_rank > 10 and ncwp_rank <= 3:
            best_other_rank = min(base_rank, pca_rank, soft_rank)
            candidates.append(
                {
                    "qid": qid,
                    "query": query_map[qid],
                    "base_rank": base_rank,
                    "pca_rank": pca_rank,
                    "soft_rank": soft_rank,
                    "ncwp_rank": ncwp_rank,
                    "best_other_rank": best_other_rank,
                    "gain": min(best_other_rank, 1000) - ncwp_rank,
                }
            )

    candidates.sort(key=lambda x: (-x["gain"], x["ncwp_rank"], x["qid"]))

    if len(candidates) >= num_examples:
        return candidates[:num_examples], "strict"

    relaxed = []
    for qid in query_ids:
        base_rank = by_method["Base"][qid]["rank"]
        pca_rank = by_method["PCA-White"][qid]["rank"]
        soft_rank = by_method["Soft-White"][qid]["rank"]
        ncwp_rank = by_method["NCWP"][qid]["rank"]

        if base_rank > 10 and pca_rank > 10 and soft_rank > 10 and ncwp_rank <= 10:
            best_other_rank = min(base_rank, pca_rank, soft_rank)
            relaxed.append(
                {
                    "qid": qid,
                    "query": query_map[qid],
                    "base_rank": base_rank,
                    "pca_rank": pca_rank,
                    "soft_rank": soft_rank,
                    "ncwp_rank": ncwp_rank,
                    "best_other_rank": best_other_rank,
                    "gain": min(best_other_rank, 1000) - ncwp_rank,
                }
            )
    relaxed.sort(key=lambda x: (-x["gain"], x["ncwp_rank"], x["qid"]))
    return relaxed[:num_examples], "relaxed"


def main():
    ap = argparse.ArgumentParser(description="Cherry-pick per-query cases where NCWP succeeds but Base/PCA fail.")
    ap.add_argument("--dataset", default="quora")
    ap.add_argument("--model", default="qwen-4b")
    ap.add_argument("--fit-sample", type=int, default=300)
    ap.add_argument("--eval-query-limit", type=int, default=2000)
    ap.add_argument("--eval-corpus-limit", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=320)
    ap.add_argument("--num-examples", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    # This analysis only needs cached embeddings and small matrix ops.
    rbe.cfg.device = "cpu"

    out_dir = ROOT / f"{args.dataset}_results" / "fit_query_sample"
    cache_dir = ROOT / f"{args.dataset}_results" / "embedding_cache" / args.model

    base_corpus = np.load(cache_dir / f"corpus_base_evalQ{args.eval_query_limit}_evalC{args.eval_corpus_limit}.npy").astype(np.float32)
    base_queries = np.load(cache_dir / f"test_queries_base_evalQ{args.eval_query_limit}_evalC{args.eval_corpus_limit}.npy").astype(np.float32)
    x_fit = np.load(cache_dir / f"fit_query_sample_N{args.fit_sample}.npy").astype(np.float32)

    (corpus_ids, corpus_texts,
     test_qids, test_qtexts,
     qrels_test, _dev_qtexts) = rbe.load_beir_data(args.dataset)
    (eval_corpus_ids, eval_corpus_texts,
     eval_qids, eval_qtexts,
     eval_qrels) = rbe.build_eval_subset(
        corpus_ids, corpus_texts,
        test_qids, test_qtexts, qrels_test,
        query_limit=args.eval_query_limit,
        corpus_limit=args.eval_corpus_limit,
        seed=rbe.cfg.seed,
    )

    corpus_id_to_idx = {cid: i for i, cid in enumerate(eval_corpus_ids)}

    base_dim = base_corpus.shape[1]
    mu_pca, w_pca, comps, scales = rbe.compute_pca_whitening_matrix(x_fit, base_dim)

    k = args.dim
    emb_by_method = {
        "Base": (base_corpus, base_queries),
        "PCA-White": (
            rbe.proj_linear(base_corpus, w_pca[:, :k], mu_pca),
            rbe.proj_linear(base_queries, w_pca[:, :k], mu_pca),
        ),
        "Soft-White": (
            rbe.proj_soft_white(base_corpus, mu_pca, comps, scales, k),
            rbe.proj_soft_white(base_queries, mu_pca, comps, scales, k),
        ),
    }

    weight_path = out_dir / "weights" / args.model / "NCWP_Default" / f"dim_{k}.npz"
    ncwp_data = np.load(weight_path)
    emb_by_method["NCWP"] = (
        rbe.proj_ncwp(base_corpus, ncwp_data["W"], ncwp_data["mu_in"], ncwp_data["mu_out"], ncwp_data["std_out"]),
        rbe.proj_ncwp(base_queries, ncwp_data["W"], ncwp_data["mu_in"], ncwp_data["mu_out"], ncwp_data["std_out"]),
    )

    by_method = {}
    for method, (c_embs, q_embs) in emb_by_method.items():
        by_method[method] = analyze_method(
            eval_qids,
            q_embs,
            c_embs,
            corpus_id_to_idx,
            eval_qrels,
            topk=args.topk,
            chunk=64,
        )

    candidates, mode = build_candidates(
        eval_qids, eval_qtexts, eval_qrels,
        eval_corpus_ids, eval_corpus_texts, by_method,
        args.num_examples,
    )

    stem = (
        f"{args.model}_N{args.fit_sample}_evalQ{args.eval_query_limit}_evalC{args.eval_corpus_limit}"
        f"_dim{args.dim}_ncwp_cherrypicks"
    )
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    table_csv_path = out_dir / f"{stem}_top{args.topk}_table.csv"
    table_md_path = out_dir / f"{stem}_top{args.topk}_table.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid", "query", "base_rank", "pca_rank", "soft_rank", "ncwp_rank", "gain",
            ],
        )
        writer.writeheader()
        for row in candidates:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with open(table_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["example_no", "qid", "query", "method"]
        fieldnames += [f"top{i}_docid" for i in range(1, args.topk + 1)]
        fieldnames += [f"top{i}_text" for i in range(1, args.topk + 1)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(candidates, start=1):
            qid = row["qid"]
            for method in ("Base", "PCA-White", "NCWP"):
                out_row = {
                    "example_no": i,
                    "qid": qid,
                    "query": row["query"],
                    "method": method,
                }
                top_idx = by_method[method][qid]["top_idx"]
                for rank, idx in enumerate(top_idx, start=1):
                    cid = eval_corpus_ids[idx]
                    marker = " [relevant]" if cid in eval_qrels.get(qid, {}) else ""
                    out_row[f"top{rank}_docid"] = f"{cid}{marker}"
                    out_row[f"top{rank}_text"] = shorten(eval_corpus_texts[idx], 200)
                writer.writerow(out_row)

    lines = []
    lines.append(f"# NCWP Cherry-picked Examples ({args.dataset}, {args.model})")
    lines.append("")
    lines.append(f"- Fit sample N: `{args.fit_sample}`")
    lines.append(f"- Eval: `evalQ={args.eval_query_limit}`, `evalC={args.eval_corpus_limit}`")
    lines.append(f"- Compared dimension: `{args.dim}`")
    lines.append(f"- Selection mode: `{mode}`")
    lines.append("- Criteria:")
    lines.append("  - strict: Base / PCA-White / Soft-White first relevant rank > 10, NCWP first relevant rank <= 3")
    lines.append("  - relaxed fallback: Base / PCA-White / Soft-White first relevant rank > 10, NCWP first relevant rank <= 10")
    lines.append("")

    for i, row in enumerate(candidates, start=1):
        qid = row["qid"]
        positive_set = {cid for cid, score in eval_qrels.get(qid, {}).items() if score > 0}
        best_idx = by_method["NCWP"][qid]["best_idx"]
        lines.append(f"## Example {i}")
        lines.append("")
        lines.append(f"- Query ID: `{qid}`")
        lines.append(f"- Query: {shorten(row['query'], 500)}")
        lines.append(f"- First relevant rank: Base `{row['base_rank']}`, PCA `{row['pca_rank']}`, Soft `{row['soft_rank']}`, NCWP `{row['ncwp_rank']}`")
        if best_idx is not None:
            lines.append(f"- NCWP best relevant doc: `{eval_corpus_ids[best_idx]}` - {shorten(eval_corpus_texts[best_idx], 300)}")
        lines.append("")
        for method in ("Base", "PCA-White", "Soft-White", "NCWP"):
            lines.append(f"### {method} Top-{args.topk}")
            lines.extend(render_doc_list(by_method[method][qid]["top_idx"], eval_corpus_ids, eval_corpus_texts, positive_set))
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    table_lines = []
    table_lines.append(f"# NCWP Cherry-picked Top-{args.topk} Table ({args.dataset}, {args.model})")
    table_lines.append("")
    table_lines.append(f"- Fit sample N: `{args.fit_sample}`")
    table_lines.append(f"- Eval: `evalQ={args.eval_query_limit}`, `evalC={args.eval_corpus_limit}`")
    table_lines.append(f"- Compared dimension: `{args.dim}`")
    table_lines.append(f"- Selection mode: `{mode}`")
    table_lines.append("")
    for i, row in enumerate(candidates, start=1):
        qid = row["qid"]
        positive_set = {cid for cid, score in eval_qrels.get(qid, {}).items() if score > 0}
        table_lines.append(f"## Example {i}")
        table_lines.append("")
        table_lines.append(f"- Query ID: `{qid}`")
        table_lines.append(f"- Query: {shorten(row['query'], 500)}")
        table_lines.append(f"- First relevant rank: Base `{row['base_rank']}`, PCA `{row['pca_rank']}`, NCWP `{row['ncwp_rank']}`")
        table_lines.append("")
        header_cells = ["Method"] + [f"Top-{i}" for i in range(1, args.topk + 1)]
        table_lines.append("| " + " | ".join(header_cells) + " |")
        table_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
        for method in ("Base", "PCA-White", "NCWP"):
            table_lines.append(
                render_table_row(method, by_method[method][qid]["top_idx"], eval_corpus_ids, eval_corpus_texts, positive_set)
            )
        table_lines.append("")

    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved Table CSV: {table_csv_path}")
    print(f"Saved Table Markdown: {table_md_path}")
    print(f"Selection mode: {mode}")
    print(f"Num examples: {len(candidates)}")
    for row in candidates[:5]:
        print(
            f"{row['qid']}: Base={row['base_rank']} PCA={row['pca_rank']} "
            f"Soft={row['soft_rank']} NCWP={row['ncwp_rank']} | {shorten(row['query'], 120)}"
        )


if __name__ == "__main__":
    main()
