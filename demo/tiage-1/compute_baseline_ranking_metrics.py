import argparse
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def tfidf_cosine_scores(query: str, docs: List[str]) -> List[float]:
    q_toks = tokenize(query)
    d_toks = [tokenize(x) for x in docs]
    n_docs = max(1, len(d_toks))

    df = Counter()
    for toks in d_toks:
        df.update(set(toks))
    idf = {t: math.log((1 + n_docs) / (1 + c)) + 1 for t, c in df.items()}

    def vec(toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        return {t: float(c) * idf.get(t, 0.0) for t, c in tf.items() if t in idf}

    def norm(v: Dict[str, float]) -> float:
        return math.sqrt(sum(x * x for x in v.values()))

    qv = vec(q_toks)
    qn = norm(qv)
    if qn < 1e-12:
        return [0.0 for _ in docs]

    sims: List[float] = []
    for toks in d_toks:
        dv = vec(toks)
        dn = norm(dv)
        if dn < 1e-12:
            sims.append(0.0)
            continue
        if len(qv) <= len(dv):
            dot = sum(v * dv.get(t, 0.0) for t, v in qv.items())
        else:
            dot = sum(v * qv.get(t, 0.0) for t, v in dv.items())
        sims.append(dot / (qn * dn + 1e-12))
    return sims


def minmax_norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def kendall_tau_a(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n < 2:
        return 0.0
    c = 0
    d = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = y_true[i] - y_true[j]
            b = y_pred[i] - y_pred[j]
            s = a * b
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    denom = n * (n - 1) / 2
    return (c - d) / denom if denom > 0 else 0.0


def ndcg_at_k(y_true: List[float], y_pred: List[float], k: int) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    order_pred = sorted(range(n), key=lambda i: y_pred[i], reverse=True)
    order_true = sorted(range(n), key=lambda i: y_true[i], reverse=True)

    def dcg(order: List[int]) -> float:
        s = 0.0
        for r, idx in enumerate(order[:k], start=1):
            gain = float(y_true[idx])
            s += gain / math.log2(r + 1)
        return s

    best = dcg(order_true)
    if best < 1e-12:
        return 0.0
    return dcg(order_pred) / best


def load_node_slice_map(centrality_dir: Path) -> Dict[int, int]:
    node_slice: Dict[int, int] = {}
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
                nid, _ = line.split(",", 1)
                node_slice[int(nid)] = sid
    return node_slice


def load_snapshot_y(processed_dir: Path, device: torch.device) -> Dict[int, torch.Tensor]:
    y_map: Dict[int, torch.Tensor] = {}
    for p in sorted(processed_dir.glob("*.pickle"), key=lambda x: int(x.stem)):
        with p.open("rb") as f:
            g = pickle.load(f)
        y_map[int(p.stem)] = g.y.to(device)
    return y_map


def build_llm_map(llm_json: Path) -> Dict[Tuple[str, int, int, int], dict]:
    data = json.load(llm_json.open("r", encoding="utf-8"))
    m: Dict[Tuple[str, int, int, int], dict] = {}
    for split in ("train", "dev", "test"):
        for r in data.get("splits", {}).get(split, []):
            key = (split, int(r["dialog_id"]), int(r["turn_id"]), int(r["node_id"]))
            llm = r.get("llm_rerank", {})
            if not llm:
                continue
            score_map: Dict[str, float] = {}
            for s in llm.get("scores", []):
                try:
                    score_map[str(s.get("node"))] = float(s.get("score"))
                except (TypeError, ValueError):
                    continue
            rank_nodes = [str(x) for x in llm.get("reranked_nodes", [])]
            m[key] = {"score_map": score_map, "rank_nodes": rank_nodes}
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline ranking metrics on test samples.")
    parser.add_argument("--step1_json", type=str, default="demo/tiage-1/step1_candidates_top10.json")
    parser.add_argument("--llm_rerank_json", type=str, default="demo/tiage-1/step2_reranked_annotation.json")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="llm_or_weak",
        choices=["llm_or_weak", "llm_only", "weak_only"],
        help="Ground-truth target mode for metric evaluation.",
    )
    parser.add_argument("--centrality_dir", type=str, default="demo/DGCN3/Centrality/alpha_1.5")
    parser.add_argument("--processed_dir", type=str, default="demo/DGCN3/datasets/processed_data/tiage")
    parser.add_argument("--lambda_c", type=float, default=0.6)
    parser.add_argument("--metrics_json", type=str, default="demo/tiage-1/baseline_ranking_metrics.json")
    parser.add_argument("--metrics_csv", type=str, default="demo/tiage-1/baseline_ranking_metrics.csv")
    args = parser.parse_args()

    root = Path(".").resolve()
    step1_json = (root / args.step1_json).resolve()
    llm_rerank_json = (root / args.llm_rerank_json).resolve()
    centrality_dir = (root / args.centrality_dir).resolve()
    processed_dir = (root / args.processed_dir).resolve()
    metrics_json = (root / args.metrics_json).resolve()
    metrics_csv = (root / args.metrics_csv).resolve()

    data = json.load(step1_json.open("r", encoding="utf-8"))
    node_slice = load_node_slice_map(centrality_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_map = load_snapshot_y(processed_dir, device)
    llm_map = build_llm_map(llm_rerank_json) if llm_rerank_json.exists() else {}

    # If step1 has config lambda, use it unless overridden explicitly.
    if "config" in data and "lambda_c" in data["config"]:
        lambda_cfg = float(data["config"]["lambda_c"])
    else:
        lambda_cfg = args.lambda_c
    lambda_c = float(args.lambda_c if args.lambda_c is not None else lambda_cfg)

    methods = ["Centrality Only", "Relevance Only", "Centrality + Relevance"]
    stats = {
        m: {"tau": [], "ndcg5": [], "ndcg3": [], "count": 0}
        for m in methods
    }
    target_source_counter = {"llm": 0, "weak": 0}

    test_samples = data.get("splits", {}).get("test", [])
    for s in test_samples:
        cands = s.get("candidates_topk", [])
        if not cands:
            continue
        node_ids = [int(c["node_id"]) for c in cands]
        base_c = [float(c.get("centrality", 0.0)) for c in cands]
        texts = [str(c.get("text", "")) for c in cands]

        sid = s.get("slice_id")
        if sid is None:
            sid = node_slice.get(int(s["node_id"]))
        if sid is None or int(sid) not in y_map:
            continue
        sid = int(sid)
        y = y_map[sid]
        if max(node_ids) >= y.size(0):
            continue
        y_true = [float(y[n].detach().cpu().item()) for n in node_ids]

        shift = int(s.get("shift_label", 0))
        context = s.get("context", [])
        response = s.get("response", "")
        query = response if shift == 1 else " ".join(context)
        rel = tfidf_cosine_scores(query, texts)

        c_norm = minmax_norm(base_c)
        r_norm = minmax_norm(rel)
        combo = [lambda_c * c + (1.0 - lambda_c) * r for c, r in zip(c_norm, r_norm)]

        # Build evaluation target according to requested mode:
        # llm_or_weak: use llm target if available else weak target
        # llm_only: use llm target only
        # weak_only: use weak target only
        weak_target = combo
        key = (str(s.get("split", "test")), int(s["dialog_id"]), int(s["turn_id"]), int(s["node_id"]))
        llm_entry = llm_map.get(key)
        llm_target: List[float] = []
        if llm_entry is not None:
            score_map = llm_entry.get("score_map", {})
            rank_nodes = llm_entry.get("rank_nodes", [])
            if score_map:
                llm_target = [float(score_map.get(str(n), float("nan"))) for n in node_ids]
            if (not llm_target) or any(math.isnan(x) for x in llm_target):
                rank_pos = {n: i for i, n in enumerate(rank_nodes)}
                max_rank = max(1, len(rank_nodes))
                llm_target = [float(max_rank - rank_pos.get(str(n), max_rank)) for n in node_ids]
        llm_target_valid = bool(llm_target) and any(x != llm_target[0] for x in llm_target)
        if llm_target_valid:
            llm_target = minmax_norm(llm_target)

        if args.target_mode == "llm_only":
            if not llm_target_valid:
                continue
            y_true = llm_target
            target_source_counter["llm"] += 1
        elif args.target_mode == "weak_only":
            y_true = weak_target
            target_source_counter["weak"] += 1
        else:
            if llm_target_valid:
                y_true = llm_target
                target_source_counter["llm"] += 1
            else:
                y_true = weak_target
                target_source_counter["weak"] += 1

        pred_map = {
            "Centrality Only": c_norm,
            "Relevance Only": r_norm,
            "Centrality + Relevance": combo,
        }
        for m in methods:
            pred = pred_map[m]
            stats[m]["tau"].append(kendall_tau_a(y_true, pred))
            stats[m]["ndcg5"].append(ndcg_at_k(y_true, pred, 5))
            stats[m]["ndcg3"].append(ndcg_at_k(y_true, pred, 3))
            stats[m]["count"] += 1

    result_rows = []
    for m in methods:
        cnt = max(1, len(stats[m]["tau"]))
        row = {
            "Method": m,
            "Kendall Tau": sum(stats[m]["tau"]) / cnt,
            "NDCG@5": sum(stats[m]["ndcg5"]) / cnt,
            "NDCG@3": sum(stats[m]["ndcg3"]) / cnt,
            "Count": len(stats[m]["tau"]),
        }
        result_rows.append(row)

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lambda_c": lambda_c,
                "target_mode": args.target_mode,
                "llm_rerank_json": str(llm_rerank_json),
                "target_source_count": target_source_counter,
                "rows": result_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("Method,Kendall Tau,NDCG@5,NDCG@3,Count\n")
        for r in result_rows:
            f.write(
                f"{r['Method']},{r['Kendall Tau']:.6f},{r['NDCG@5']:.6f},{r['NDCG@3']:.6f},{r['Count']}\n"
            )

    print(f"[SAVED] {metrics_json}")
    print(f"[SAVED] {metrics_csv}")
    for r in result_rows:
        print(
            f"{r['Method']}: Kendall Tau={r['Kendall Tau']:.6f}, "
            f"NDCG@5={r['NDCG@5']:.6f}, NDCG@3={r['NDCG@3']:.6f}, Count={r['Count']}"
        )
    print(f"target_source_count: llm={target_source_counter['llm']} weak={target_source_counter['weak']}")


if __name__ == "__main__":
    main()
