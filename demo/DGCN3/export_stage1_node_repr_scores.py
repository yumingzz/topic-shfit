import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from model import NodeImportanceModel


def parse_range(spec: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", spec)
    if not m:
        raise ValueError(f"Invalid range: {spec}. Expected like 101-200.")
    start, end = int(m.group(1)), int(m.group(2))
    if start < 1 or end < start:
        raise ValueError(f"Invalid range values: {spec}")
    return start, end


def load_nodes_csv(path: Path) -> Dict[str, List[dict]]:
    splits = {"train": [], "dev": [], "test": []}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            if split not in splits:
                continue
            splits[split].append(
                {
                    "node_id": int(row["node_id"]),
                    "split": split,
                    "dialog_id": int(row["dialog_id"]),
                    "turn_id": int(row["turn_id"]),
                    "text": row["text"],
                    "shift_label": int(row["shift_label"]),
                }
            )
    return splits


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
                node_str, _val = line.split(",", 1)
                node_slice[int(node_str)] = sid
    return node_slice


def load_snapshots(processed_dir: Path, device: torch.device) -> Dict[int, object]:
    snapshots: Dict[int, object] = {}
    files = sorted(processed_dir.glob("*.pickle"), key=lambda p: int(p.stem))
    for p in files:
        with p.open("rb") as f:
            g = pickle.load(f)
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        if hasattr(g, "y") and g.y is not None:
            g.y = g.y.to(device)
        snapshots[int(p.stem)] = g
    return snapshots


def build_model(model_path: Path, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device) -> NodeImportanceModel:
    model = NodeImportanceModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def compute_stage1_for_slice(model: NodeImportanceModel, graph, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    z = model.encoder(graph.x, graph.edge_index, normalize=normalize)
    c = model.importance_predictor(z).squeeze(-1)
    return z, c


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stage-1 node hidden representations g_i and centrality scores C_i.")
    parser.add_argument("--nodes_csv", type=str, default="demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv")
    parser.add_argument("--centrality_dir", type=str, default="demo/DGCN3/Centrality/alpha_1.5")
    parser.add_argument("--processed_dir", type=str, default="demo/DGCN3/datasets/processed_data/tiage")
    parser.add_argument("--model_path", type=str, default="demo/DGCN3/model_registry/node_importance_tiage.pkl")
    parser.add_argument("--output_json", type=str, default="demo/tiage-1/stage1_node_repr_scores_selected.json")

    parser.add_argument("--train_range", type=str, default="101-200")
    parser.add_argument("--dev_range", type=str, default="1-50")
    parser.add_argument("--test_range", type=str, default="1-50")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(".").resolve()
    nodes_csv = (root / args.nodes_csv).resolve()
    centrality_dir = (root / args.centrality_dir).resolve()
    processed_dir = (root / args.processed_dir).resolve()
    model_path = (root / args.model_path).resolve()
    output_json = (root / args.output_json).resolve()

    split_data = load_nodes_csv(nodes_csv)
    node_slice = load_node_slice_map(centrality_dir)
    train_start, train_end = parse_range(args.train_range)
    dev_start, dev_end = parse_range(args.dev_range)
    test_start, test_end = parse_range(args.test_range)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshots = load_snapshots(processed_dir, device=device)
    if not snapshots:
        raise RuntimeError(f"No snapshot pickle files found in: {processed_dir}")
    first_graph = snapshots[min(snapshots.keys())]
    input_dim = int(first_graph.x.size(1))
    model = build_model(
        model_path=model_path,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        device=device,
    )

    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def export_for_split(split: str, start: int, end: int) -> List[dict]:
        rows = split_data[split]
        selected = rows[start - 1 : min(end, len(rows))]
        results: List[dict] = []
        for idx, row in enumerate(selected, start=start):
            node_id = row["node_id"]
            sid = node_slice.get(node_id)
            if sid is None or sid not in snapshots:
                results.append(
                    {
                        "sample_index_in_split": idx,
                        "split": split,
                        "node_id": node_id,
                        "dialog_id": row["dialog_id"],
                        "turn_id": row["turn_id"],
                        "shift_label": row["shift_label"],
                        "slice_id": sid,
                        "status": "missing_slice",
                    }
                )
                continue

            if sid not in cache:
                cache[sid] = compute_stage1_for_slice(model, snapshots[sid], normalize=args.normalize)

            z, c = cache[sid]
            g_i = z[node_id].detach().cpu().tolist()
            c_i = float(c[node_id].detach().cpu().item())

            results.append(
                {
                    "sample_index_in_split": idx,
                    "split": split,
                    "node_id": node_id,
                    "dialog_id": row["dialog_id"],
                    "turn_id": row["turn_id"],
                    "shift_label": row["shift_label"],
                    "slice_id": sid,
                    "g_i": g_i,
                    "C_i": c_i,
                    "status": "ok",
                }
            )
        return results

    output = {
        "config": {
            "nodes_csv": str(nodes_csv),
            "centrality_dir": str(centrality_dir),
            "processed_dir": str(processed_dir),
            "model_path": str(model_path),
            "device": str(device),
            "ranges": {"train": args.train_range, "dev": args.dev_range, "test": args.test_range},
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "normalize": bool(args.normalize),
        },
        "splits": {
            "train": export_for_split("train", train_start, train_end),
            "dev": export_for_split("dev", dev_start, dev_end),
            "test": export_for_split("test", test_start, test_end),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"[SAVED] {output_json}")
    print(
        "[COUNT] train={} dev={} test={}".format(
            len(output["splits"]["train"]),
            len(output["splits"]["dev"]),
            len(output["splits"]["test"]),
        )
    )


if __name__ == "__main__":
    main()
