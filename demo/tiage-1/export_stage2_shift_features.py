import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel


@dataclass
class Sample:
    split: str
    dialog_id: int
    turn_id: int
    node_id: int
    text: str
    label: int
    centrality: float


def parse_range(spec: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", spec)
    if not m:
        raise ValueError(f"Invalid range: {spec}. Expected like 101-200")
    start, end = int(m.group(1)), int(m.group(2))
    if start < 1 or end < start:
        raise ValueError(f"Invalid range values: {spec}")
    return start, end


def parse_nodes(nodes_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with nodes_csv.open("r", encoding="utf-8", newline="") as f:
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


def load_centrality(centrality_dir: Path, num_slices: int) -> Dict[int, float]:
    centrality: Dict[int, float] = {}
    for sid in range(num_slices):
        path = centrality_dir / f"tiage_{sid}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                node_str, val_str = line.split(",", 1)
                centrality[int(node_str)] = float(val_str)
    return centrality


def build_samples(rows: List[dict], centrality: Dict[int, float], context_turns: int) -> Dict[str, List[Sample]]:
    by_dialog: Dict[Tuple[str, int], List[dict]] = {}
    for r in rows:
        by_dialog.setdefault((r["split"], r["dialog_id"]), []).append(r)

    result: Dict[str, List[Sample]] = {"train": [], "dev": [], "test": []}
    for (split, _dialog_id), turns in by_dialog.items():
        turns = sorted(turns, key=lambda x: x["turn_id"])
        for i, cur in enumerate(turns):
            label = cur["shift_label"]
            if label not in (0, 1):
                continue

            hist = turns[:i]
            if context_turns > 0:
                hist = hist[-context_turns:]

            ctx_parts: List[str] = []
            for h in hist:
                h_c = centrality.get(h["node_id"], 0.0)
                ctx_parts.append(f"{h['text']} [CEN={h_c:.4f}]")

            cur_c = centrality.get(cur["node_id"], 0.0)
            context_text = " </s> ".join(ctx_parts) if ctx_parts else "<no_context>"
            text = (
                f"task: topic shift detection\n"
                f"context: {context_text}\n"
                f"response: {cur['text']} [CEN={cur_c:.4f}]"
            )

            s = Sample(
                split=split,
                dialog_id=cur["dialog_id"],
                turn_id=cur["turn_id"],
                node_id=cur["node_id"],
                text=text,
                label=label,
                centrality=cur_c,
            )
            if split in result:
                result[split].append(s)
    return result


class TiageDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Sample]) -> dict:
        texts = [x.text for x in batch]
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([x.label for x in batch], dtype=torch.long)
        feats = torch.tensor([[x.centrality] for x in batch], dtype=torch.float)
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels,
            "extra_features": feats,
            "meta": batch,
        }


class RWKVChannelMix(nn.Module):
    def __init__(self, hidden_size: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_size = hidden_size * hidden_mult
        self.ln = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, inner_size, bias=False)
        self.value = nn.Linear(inner_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.ln(x)
        k = torch.relu(self.key(xx)) ** 2
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xx))
        return x + self.dropout(r * kv)


class T5Baseline(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-base",
        dropout: float = 0.2,
        channelmix_layers: int = 1,
        channelmix_hidden_mult: int = 4,
        channelmix_dropout: float = 0.1,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = self.encoder.config.d_model
        self.channelmix_stack = nn.ModuleList(
            [RWKVChannelMix(hidden, hidden_mult=channelmix_hidden_mult, dropout=channelmix_dropout) for _ in range(max(0, channelmix_layers))]
        )
        self.attn_pool = nn.Linear(hidden, 1)
        self.feature_norm = nn.LayerNorm(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def encode_semantic(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        for block in self.channelmix_stack:
            hs = block(hs)
        attn_logits = self.attn_pool(hs).squeeze(-1)
        attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        attn_weight = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        pooled = (hs * attn_weight).sum(dim=1)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, extra_features: torch.Tensor):
        pooled = self.encode_semantic(input_ids, attention_mask)
        feats = self.feature_norm(extra_features)
        logits = self.classifier(torch.cat([pooled, feats], dim=-1))
        return pooled, logits


@torch.no_grad()
def extract_features(model: T5Baseline, loader: DataLoader, device: torch.device, threshold: float) -> List[dict]:
    model.eval()
    rows: List[dict] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        pooled, logits = model(input_ids, attention_mask, extra_features)
        probs = torch.softmax(logits, dim=-1)
        p_shift = probs[:, 1]
        y_hat = (p_shift >= threshold).long()

        pooled_np = pooled.detach().cpu().tolist()
        p_shift_np = p_shift.detach().cpu().tolist()
        y_hat_np = y_hat.detach().cpu().tolist()
        for m, h_t, p_s, y_p in zip(batch["meta"], pooled_np, p_shift_np, y_hat_np):
            rows.append(
                {
                    "split": m.split,
                    "dialog_id": m.dialog_id,
                    "turn_id": m.turn_id,
                    "node_id": m.node_id,
                    "label": m.label,
                    "h_t": h_t,
                    "p_shift": float(p_s),
                    "y_hat_shift": int(y_p),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stage-2 semantic representation and shift outputs.")
    parser.add_argument("--nodes_csv", type=str, default="demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv")
    parser.add_argument("--centrality_dir", type=str, default="demo/DGCN3/Centrality/alpha_1.5")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--model_ckpt", type=str, default="demo/tiage-1/results_t5_base_channelmix_best_baseline/best_model.pt")
    parser.add_argument("--output_json", type=str, default="demo/tiage-1/stage2_shift_features_selected.json")

    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--context_turns", type=int, default=16)
    parser.add_argument("--num_slices", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--channelmix_layers", type=int, default=1)
    parser.add_argument("--channelmix_hidden_mult", type=int, default=4)
    parser.add_argument("--channelmix_dropout", type=float, default=0.1)

    # parser.add_argument("--train_range", type=str, default="101-200")
    # parser.add_argument("--dev_range", type=str, default="1-50")
    # parser.add_argument("--test_range", type=str, default="1-50")
    parser.add_argument("--train_range", type=str, default="1-300")
    parser.add_argument("--dev_range", type=str, default="1-100")
    parser.add_argument("--test_range", type=str, default="1-100")
    args = parser.parse_args()

    root = Path(".").resolve()
    nodes_csv = (root / args.nodes_csv).resolve()
    centrality_dir = (root / args.centrality_dir).resolve()
    model_ckpt = (root / args.model_ckpt).resolve()
    output_json = (root / args.output_json).resolve()

    rows = parse_nodes(nodes_csv)
    centrality = load_centrality(centrality_dir, args.num_slices)
    split_samples = build_samples(rows, centrality, args.context_turns)

    train_start, train_end = parse_range(args.train_range)
    dev_start, dev_end = parse_range(args.dev_range)
    test_start, test_end = parse_range(args.test_range)
    selected = {
        "train": split_samples["train"][train_start - 1 : min(train_end, len(split_samples["train"]))],
        "dev": split_samples["dev"][dev_start - 1 : min(dev_end, len(split_samples["dev"]))],
        "test": split_samples["test"][test_start - 1 : min(test_end, len(split_samples["test"]))],
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = T5Baseline(
        model_name=args.model_name,
        dropout=args.classifier_dropout,
        channelmix_layers=args.channelmix_layers,
        channelmix_hidden_mult=args.channelmix_hidden_mult,
        channelmix_dropout=args.channelmix_dropout,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))

    collator = Collator(tokenizer, args.max_length)
    loaders = {
        k: DataLoader(TiageDataset(v), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        for k, v in selected.items()
    }

    output = {
        "config": {
            "nodes_csv": str(nodes_csv),
            "centrality_dir": str(centrality_dir),
            "model_ckpt": str(model_ckpt),
            "model_name": args.model_name,
            "context_turns": args.context_turns,
            "max_length": args.max_length,
            "threshold": args.threshold,
            "ranges": {"train": args.train_range, "dev": args.dev_range, "test": args.test_range},
        },
        "splits": {
            "train": extract_features(model, loaders["train"], device, args.threshold),
            "dev": extract_features(model, loaders["dev"], device, args.threshold),
            "test": extract_features(model, loaders["test"], device, args.threshold),
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
