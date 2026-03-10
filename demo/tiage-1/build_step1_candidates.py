import argparse
import csv
import json
import math
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_nodes_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "node_id": int(r["node_id"]),
                    "split": r["split"].strip(),
                    "dialog_id": int(r["dialog_id"]),
                    "turn_id": int(r["turn_id"]),
                    "text": r["text"],
                    "shift_label": int(r["shift_label"]),
                }
            )
    return rows


def parse_centrality(
    centrality_dir: Path,
) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, List[int]]]:
    node_centrality: Dict[int, float] = {}
    node_slice: Dict[int, int] = {}
    slice_nodes: Dict[int, List[int]] = defaultdict(list)

    for p in sorted(centrality_dir.glob("tiage_*.csv")):
        m = re.match(r"tiage_(\d+)\.csv$", p.name)
        if not m:
            continue
        sid = int(m.group(1))
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                node_str, val_str = line.split(",", 1)
                node_id = int(node_str)
                val = float(val_str)
                node_centrality[node_id] = val
                node_slice[node_id] = sid
                slice_nodes[sid].append(node_id)
    return node_centrality, node_slice, slice_nodes


def build_split_order(rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {"train": [], "dev": [], "test": []}
    for r in rows:
        if r["split"] in grouped:
            grouped[r["split"]].append(r)
    for split in grouped:
        grouped[split].sort(key=lambda x: (x["dialog_id"], x["turn_id"], x["node_id"]))
    return grouped


def build_dialog_histories(rows: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    by_dialog: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        by_dialog[(r["split"], r["dialog_id"])].append(r)
    for k in by_dialog:
        by_dialog[k].sort(key=lambda x: x["turn_id"])
    return by_dialog


def minmax_norm(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def parse_range(spec: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", spec)
    if not m:
        raise ValueError(f"Invalid range format: {spec}. Expected like 101-200")
    start = int(m.group(1))
    end = int(m.group(2))
    if start < 1 or end < start:
        raise ValueError(f"Invalid range values: {spec}")
    return start, end


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def build_slice_tfidf_model(node_ids: List[int], node_text: Dict[int, str]) -> dict:
    docs_tokens: List[List[str]] = [tokenize(node_text.get(nid, "")) for nid in node_ids]
    n_docs = max(1, len(docs_tokens))

    df: Counter = Counter()
    for toks in docs_tokens:
        df.update(set(toks))

    idf: Dict[str, float] = {}
    for tok, d in df.items():
        idf[tok] = math.log((1.0 + n_docs) / (1.0 + d)) + 1.0

    doc_vectors: List[Dict[str, float]] = []
    doc_norms: List[float] = []
    for toks in docs_tokens:
        tf = Counter(toks)
        vec: Dict[str, float] = {}
        for tok, c in tf.items():
            vec[tok] = float(c) * idf.get(tok, 0.0)
        norm = math.sqrt(sum(v * v for v in vec.values()))
        doc_vectors.append(vec)
        doc_norms.append(norm)

    return {
        "nodes": node_ids,
        "idf": idf,
        "doc_vectors": doc_vectors,
        "doc_norms": doc_norms,
    }


def text_to_tfidf_vector(text: str, idf: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    tf = Counter(tokenize(text))
    vec: Dict[str, float] = {}
    for tok, c in tf.items():
        if tok in idf:
            vec[tok] = float(c) * idf[tok]
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return vec, norm


def cosine_scores(
    query_vec: Dict[str, float],
    query_norm: float,
    doc_vectors: List[Dict[str, float]],
    doc_norms: List[float],
) -> np.ndarray:
    if query_norm <= 1e-12:
        return np.zeros(len(doc_vectors), dtype=np.float64)

    scores = np.zeros(len(doc_vectors), dtype=np.float64)
    for i, dvec in enumerate(doc_vectors):
        dnorm = doc_norms[i]
        if dnorm <= 1e-12:
            continue
        # Iterate over the smaller dict for faster dot-product.
        if len(query_vec) <= len(dvec):
            dot = sum(v * dvec.get(tok, 0.0) for tok, v in query_vec.items())
        else:
            dot = sum(v * query_vec.get(tok, 0.0) for tok, v in dvec.items())
        scores[i] = dot / (query_norm * dnorm + 1e-12)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 candidate node selection for TIAGE.")
    parser.add_argument(
        "--nodes_csv",
        type=str,
        default="demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv",
    )
    parser.add_argument(
        "--centrality_dir",
        type=str,
        default="demo/DGCN3/Centrality/alpha_1.5",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="demo/tiage-1/step1_candidates_top10.json",
    )
    # parser.add_argument("--train_range", type=str, default="101-200")
    # parser.add_argument("--dev_range", type=str, default="1-50")
    # parser.add_argument("--test_range", type=str, default="1-50")
    parser.add_argument("--train_range", type=str, default="1-300")
    parser.add_argument("--dev_range", type=str, default="1-100")
    parser.add_argument("--test_range", type=str, default="1-100")
    parser.add_argument("--context_size", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--lambda_c", type=float, default=0.6)
    args = parser.parse_args()

    if not (0.0 <= args.lambda_c <= 1.0):
        raise ValueError("--lambda_c must be in [0, 1]")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.context_size < 0:
        raise ValueError("--context_size must be >= 0")

    root = Path(".").resolve()
    nodes_csv = (root / args.nodes_csv).resolve()
    centrality_dir = (root / args.centrality_dir).resolve()
    output_json = (root / args.output_json).resolve()

    rows = parse_nodes_csv(nodes_csv)
    node_text: Dict[int, str] = {r["node_id"]: r["text"] for r in rows}
    node_centrality, node_slice, slice_nodes = parse_centrality(centrality_dir)
    split_rows = build_split_order(rows)
    dialog_hist = build_dialog_histories(rows)

    # Build one TF-IDF space per time slice for semantic relevance scoring.
    slice_models: Dict[int, dict] = {}
    for sid, nodes in slice_nodes.items():
        dedup_nodes = sorted(set(nodes))
        slice_models[sid] = build_slice_tfidf_model(dedup_nodes, node_text)
    global_nodes = sorted(node_centrality.keys())
    global_model = build_slice_tfidf_model(global_nodes, node_text)

    range_cfg = {
        "train": parse_range(args.train_range),
        "dev": parse_range(args.dev_range),
        "test": parse_range(args.test_range),
    }

    output = {
        "config": {
            "nodes_csv": str(nodes_csv),
            "centrality_dir": str(centrality_dir),
            "context_size": args.context_size,
            "top_k": args.top_k,
            "lambda_c": args.lambda_c,
            "score_formula": "S_cand = lambda_c * C_norm + (1 - lambda_c) * R",
            "ranges": {
                "train": args.train_range,
                "dev": args.dev_range,
                "test": args.test_range,
            },
        },
        "splits": {"train": [], "dev": [], "test": []},
    }

    for split in ("train", "dev", "test"):
        ordered = split_rows[split]
        start, end = range_cfg[split]
        start_idx = start - 1
        end_idx = min(end, len(ordered))
        if start_idx >= len(ordered):
            continue
        selected = ordered[start_idx:end_idx]

        for sample_index, cur in enumerate(selected, start=start):
            cur_node = cur["node_id"]
            sid = node_slice.get(cur_node)
            model_pack = slice_models.get(sid, global_model)
            cand_nodes = model_pack["nodes"]
            idf = model_pack["idf"]
            doc_vectors = model_pack["doc_vectors"]
            doc_norms = model_pack["doc_norms"]

            # Context is up to the previous N utterances in the same dialog.
            dialog_rows = dialog_hist[(split, cur["dialog_id"])]
            turn_pos = next(i for i, r in enumerate(dialog_rows) if r["node_id"] == cur_node)
            hist = dialog_rows[:turn_pos]
            hist = hist[-args.context_size :] if args.context_size > 0 else []
            context_texts = [h["text"] for h in hist]
            response_text = cur["text"]
            query_text = " ".join(context_texts + [response_text]).strip()

            q_vec, q_norm = text_to_tfidf_vector(query_text, idf)
            rel = cosine_scores(q_vec, q_norm, doc_vectors, doc_norms)

            c = np.array([node_centrality.get(nid, 0.0) for nid in cand_nodes], dtype=np.float64)
            c_norm = minmax_norm(c)
            r_norm = minmax_norm(rel)
            score = args.lambda_c * c_norm + (1.0 - args.lambda_c) * r_norm

            rank_idx = np.argsort(-score)[: args.top_k]
            candidates = []
            for rank, idx in enumerate(rank_idx, start=1):
                nid = int(cand_nodes[int(idx)])
                candidates.append(
                    {
                        "rank": rank,
                        "node_id": nid,
                        "score": float(score[int(idx)]),
                        "centrality": float(c[int(idx)]),
                        "centrality_norm": float(c_norm[int(idx)]),
                        "relevance": float(rel[int(idx)]),
                        "relevance_norm": float(r_norm[int(idx)]),
                        "text": node_text.get(nid, ""),
                    }
                )

            output["splits"][split].append(
                {
                    "sample_index_in_split": sample_index,
                    "node_id": cur_node,
                    "dialog_id": cur["dialog_id"],
                    "turn_id": cur["turn_id"],
                    "slice_id": sid,
                    "slice_fallback_global": sid is None or sid not in slice_models,
                    "split": split,
                    "shift_label": cur["shift_label"],
                    "context": context_texts,
                    "response": response_text,
                    "candidates_topk": candidates,
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {output_json}")
    print(f"[COUNT] train={len(output['splits']['train'])} dev={len(output['splits']['dev'])} test={len(output['splits']['test'])}")


if __name__ == "__main__":
    main()
