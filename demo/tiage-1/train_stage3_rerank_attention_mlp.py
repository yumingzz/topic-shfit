import argparse
import json
import math
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def cosine_tfidf(query: str, docs: List[str]) -> List[float]:
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


def ndcg_at_k(y_true: List[float], y_pred: List[float], k: int = 5) -> float:
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
                nid_str, _ = line.split(",", 1)
                node_slice[int(nid_str)] = sid
    return node_slice


def load_snapshots(processed_dir: Path, device: torch.device) -> Dict[int, object]:
    snapshots: Dict[int, object] = {}
    for p in sorted(processed_dir.glob("*.pickle"), key=lambda x: int(x.stem)):
        with p.open("rb") as f:
            g = pickle.load(f)
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        snapshots[int(p.stem)] = g
    return snapshots


def build_stage2_map(stage2_json: Path) -> Dict[Tuple[str, int, int, int], dict]:
    data = json.load(stage2_json.open("r", encoding="utf-8"))
    m: Dict[Tuple[str, int, int, int], dict] = {}
    for split in ("train", "dev", "test"):
        for r in data["splits"][split]:
            k = (split, int(r["dialog_id"]), int(r["turn_id"]), int(r["node_id"]))
            m[k] = r
    return m


class MPNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.mp1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
        self.mp2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.mp3 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        z = self.mp1(x, edge_index)
        z = self.bn1(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.1, training=self.training)

        z = self.mp2(z, edge_index)
        z = self.bn2(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.1, training=self.training)

        z = self.mp3(z, edge_index)
        z = self.bn3(z)
        if normalize:
            z = F.normalize(z, p=2.0, dim=-1)
        z = F.dropout(z, p=0.1, training=self.training)
        return z


class NodeImportanceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = MPNN(input_dim, hidden_dim, output_dim)
        self.importance_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Keep these branches to stay state_dict-compatible with trained stage-1 checkpoints.
        self.local_predictive_encoder = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.global_predictive_encoder = nn.Linear(output_dim * 2, output_dim)


class Stage3Reranker(nn.Module):
    def __init__(self, node_dim: int, ctx_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, node_dim)
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 3 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_repr: torch.Tensor, base_c: torch.Tensor, h_t: torch.Tensor, p_shift: torch.Tensor) -> torch.Tensor:
        # node_repr: [K, D], base_c: [K], h_t: [H], p_shift: scalar
        ctx = self.ctx_proj(h_t.unsqueeze(0)).squeeze(0)  # [D]
        q = self.q_proj(ctx)  # [D]
        k = self.k_proj(node_repr)  # [K, D]
        attn = (k * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(node_repr.size(-1))  # [K]

        ctx_expand = ctx.unsqueeze(0).expand_as(node_repr)  # [K, D]
        p_expand = torch.full((node_repr.size(0), 1), float(p_shift), device=node_repr.device)
        feat = torch.cat(
            [
                node_repr,
                ctx_expand,
                node_repr * ctx_expand,
                base_c.unsqueeze(-1),
                p_expand,
            ],
            dim=-1,
        )  # [K, 3D+2]

        score = self.mlp(feat).squeeze(-1) + attn
        return score


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Logistic pairwise ranking: log(1 + exp(-s_ij * (pred_i - pred_j)))
    diff_t = target.unsqueeze(1) - target.unsqueeze(0)
    sign = torch.sign(diff_t)
    valid = sign != 0
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    diff_p = pred.unsqueeze(1) - pred.unsqueeze(0)
    loss = F.softplus(-sign * diff_p)
    return loss[valid].mean()


def build_samples(
    step1_json: Path,
    stage2_map: Dict[Tuple[str, int, int, int], dict],
    node_slice: Dict[int, int],
    slice_embeds: Dict[int, torch.Tensor],
    device: torch.device,
) -> Dict[str, List[dict]]:
    data = json.load(step1_json.open("r", encoding="utf-8"))
    out = {"train": [], "dev": [], "test": []}

    for split in ("train", "dev", "test"):
        for s in data["splits"][split]:
            key = (split, int(s["dialog_id"]), int(s["turn_id"]), int(s["node_id"]))
            stage2 = stage2_map.get(key)
            if stage2 is None:
                continue

            context = s.get("context", [])
            response = s.get("response", "")
            shift_label = int(s.get("shift_label", stage2.get("label", 0)))
            candidates = s["candidates_topk"]
            if not candidates:
                continue

            node_ids = [int(c["node_id"]) for c in candidates]
            base_c = [float(c.get("centrality", 0.0)) for c in candidates]
            texts = [str(c.get("text", "")) for c in candidates]

            # Build target ranking signal (weak supervision):
            # combine base centrality and context-conditioned semantic relevance.
            query = response if shift_label == 1 else " ".join(context)
            rel = cosine_tfidf(query, texts)
            c_norm = minmax_norm(base_c)
            r_norm = minmax_norm(rel)
            alpha = 0.35 if shift_label == 1 else 0.65
            target = [alpha * c + (1.0 - alpha) * r for c, r in zip(c_norm, r_norm)]

            node_repr_rows: List[torch.Tensor] = []
            for nid in node_ids:
                sid = node_slice.get(nid, s.get("slice_id"))
                if sid is None or sid not in slice_embeds:
                    node_repr_rows.append(torch.zeros(256, device=device))
                    continue
                emb = slice_embeds[sid]
                if nid >= emb.size(0):
                    node_repr_rows.append(torch.zeros(emb.size(1), device=device))
                else:
                    node_repr_rows.append(emb[nid])

            h_t = torch.tensor(stage2["h_t"], dtype=torch.float, device=device)
            p_shift = float(stage2["p_shift"])
            sample = {
                "meta": key,
                "node_ids": node_ids,
                "node_repr": torch.stack(node_repr_rows, dim=0),  # [K, 256]
                "base_c": torch.tensor(c_norm, dtype=torch.float, device=device),  # normalized scalar
                "target": torch.tensor(target, dtype=torch.float, device=device),
                "h_t": h_t,
                "p_shift": p_shift,
            }
            out[split].append(sample)

    return out


@torch.no_grad()
def evaluate(model: Stage3Reranker, samples: List[dict]) -> Tuple[float, float, float, List[dict]]:
    model.eval()
    taus: List[float] = []
    ndcgs: List[float] = []
    ndcgs3: List[float] = []
    rows: List[dict] = []

    for s in samples:
        pred = model(s["node_repr"], s["base_c"], s["h_t"], s["p_shift"]).detach().cpu().tolist()
        tgt = s["target"].detach().cpu().tolist()
        tau = kendall_tau_a(tgt, pred)
        nd = ndcg_at_k(tgt, pred, k=5)
        nd3 = ndcg_at_k(tgt, pred, k=3)
        taus.append(tau)
        ndcgs.append(nd)
        ndcgs3.append(nd3)

        order = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)
        rows.append(
            {
                "split": s["meta"][0],
                "dialog_id": s["meta"][1],
                "turn_id": s["meta"][2],
                "node_id": s["meta"][3],
                "kendall_tau": tau,
                "ndcg_at_5": nd,
                "ndcg_at_3": nd3,
                "reranked_nodes": [s["node_ids"][i] for i in order],
                "pred_scores": [pred[i] for i in order],
            }
        )

    mean_tau = sum(taus) / max(1, len(taus))
    mean_ndcg = sum(ndcgs) / max(1, len(ndcgs))
    mean_ndcg3 = sum(ndcgs3) / max(1, len(ndcgs3))
    return mean_tau, mean_ndcg, mean_ndcg3, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 reranking with Attention + MLP and pairwise ranking loss.")
    parser.add_argument("--step1_json", type=str, default="demo/tiage-1/step1_candidates_top10.json")
    parser.add_argument("--stage2_json", type=str, default="demo/tiage-1/stage2_shift_features_selected.json")
    parser.add_argument("--centrality_dir", type=str, default="demo/DGCN3/Centrality/alpha_1.5")
    parser.add_argument("--processed_dir", type=str, default="demo/DGCN3/datasets/processed_data/tiage")
    parser.add_argument("--stage1_model_path", type=str, default="demo/DGCN3/model_registry/node_importance_tiage.pkl")

    parser.add_argument("--hidden_dim_stage1", type=int, default=256)
    parser.add_argument("--output_dim_stage1", type=int, default=256)
    parser.add_argument("--stage3_hidden_dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--metrics_json", type=str, default="demo/tiage-1/stage3_test_metrics.json")
    parser.add_argument("--pred_json", type=str, default="demo/tiage-1/stage3_test_predictions.json")
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(".").resolve()
    step1_json = (root / args.step1_json).resolve()
    stage2_json = (root / args.stage2_json).resolve()
    centrality_dir = (root / args.centrality_dir).resolve()
    processed_dir = (root / args.processed_dir).resolve()
    stage1_model_path = (root / args.stage1_model_path).resolve()
    metrics_json = (root / args.metrics_json).resolve()
    pred_json = (root / args.pred_json).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    node_slice = load_node_slice_map(centrality_dir)
    snapshots = load_snapshots(processed_dir, device)
    if not snapshots:
        raise RuntimeError(f"No snapshots found in {processed_dir}")

    first_graph = snapshots[min(snapshots.keys())]
    input_dim = int(first_graph.x.size(1))
    stage1 = NodeImportanceModel(input_dim, args.hidden_dim_stage1, args.output_dim_stage1).to(device)
    stage1.load_state_dict(torch.load(stage1_model_path, map_location=device))
    stage1.eval()

    # Cache node embeddings per slice from stage-1 encoder.
    slice_embeds: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for sid, g in snapshots.items():
            z = stage1.encoder(g.x, g.edge_index, normalize=True).detach()
            slice_embeds[sid] = z

    stage2_map = build_stage2_map(stage2_json)
    samples = build_samples(step1_json, stage2_map, node_slice, slice_embeds, device)
    print(
        f"[INFO] samples train={len(samples['train'])} dev={len(samples['dev'])} test={len(samples['test'])}"
    )

    model = Stage3Reranker(node_dim=args.output_dim_stage1, ctx_dim=768, hidden_dim=args.stage3_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1e9
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(samples["train"])
        total_loss = 0.0
        for s in samples["train"]:
            pred = model(s["node_repr"], s["base_c"], s["h_t"], s["p_shift"])
            loss = pairwise_ranking_loss(pred, s["target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        dev_tau, dev_ndcg, dev_ndcg3, _ = evaluate(model, samples["dev"])
        score = dev_tau + dev_ndcg + dev_ndcg3
        print(
            f"[EPOCH {epoch}] train_loss={total_loss/max(1,len(samples['train'])):.6f} "
            f"dev_tau={dev_tau:.4f} dev_ndcg5={dev_ndcg:.4f} dev_ndcg3={dev_ndcg3:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[INFO] Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[INFO] best_epoch={best_epoch} best_dev_score={best_score:.6f}")

    test_tau, test_ndcg, test_ndcg3, test_rows = evaluate(model, samples["test"])
    print(
        f"[RESULT] test_kendall_tau={test_tau:.6f} "
        f"test_ndcg@5={test_ndcg:.6f} test_ndcg@3={test_ndcg3:.6f}"
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_dev_score": best_score,
        "test_kendall_tau": test_tau,
        "test_ndcg_at_5": test_ndcg,
        "test_ndcg_at_3": test_ndcg3,
        "num_test_samples": len(samples["test"]),
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with pred_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": test_rows}, f, ensure_ascii=False)

    print(f"[SAVED] {metrics_json}")
    print(f"[SAVED] {pred_json}")


if __name__ == "__main__":
    main()
