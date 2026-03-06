import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel, get_linear_schedule_with_warmup


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


@dataclass
class Sample:
    split: str
    dialog_id: int
    turn_id: int
    node_id: int
    text: str
    label: int
    centrality: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

            sample = Sample(
                split=split,
                dialog_id=cur["dialog_id"],
                turn_id=cur["turn_id"],
                node_id=cur["node_id"],
                text=text,
                label=label,
                centrality=cur_c,
            )
            if split in result:
                result[split].append(sample)
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
    ):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden = self.encoder.config.d_model
        self.channelmix_stack = nn.ModuleList(
            [RWKVChannelMix(hidden, hidden_mult=channelmix_hidden_mult, dropout=channelmix_dropout) for _ in range(max(0, channelmix_layers))]
        )
        self.attn_pool = nn.Linear(hidden, 1)
        self.feature_norm = nn.LayerNorm(1)
        self.feature_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.loss_fn = FocalLoss(gamma=2.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, extra_features: torch.Tensor, labels: torch.Tensor = None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        for block in self.channelmix_stack:
            hs = block(hs)
        attn_logits = self.attn_pool(hs).squeeze(-1)
        attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        attn_weight = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        pooled = (hs * attn_weight).sum(dim=1)
        feats = self.feature_norm(extra_features)
        g = self.feature_gate(feats)
        pooled_gated = pooled * (1.0 + g)
        logits = self.classifier(torch.cat([pooled_gated, feats], dim=-1))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return loss, logits


def precision_recall_macro_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    p_list: List[float] = []
    r_list: List[float] = []
    f_list: List[float] = []
    for cls in (0, 1):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    return sum(p_list) / 2.0, sum(r_list) / 2.0, sum(f_list) / 2.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    pred_rows: List[dict] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        extra_features = batch["extra_features"].to(device)
        _, logits = model(input_ids, attention_mask, extra_features, labels)
        probs = torch.softmax(logits, dim=-1)
        pred = (probs[:, 1] >= threshold).long()

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        probs1 = probs[:, 1].cpu().tolist()
        for m, t, p, p1 in zip(batch["meta"], labels.cpu().tolist(), pred.cpu().tolist(), probs1):
            pred_rows.append(
                {
                    "split": m.split,
                    "dialog_id": m.dialog_id,
                    "turn_id": m.turn_id,
                    "node_id": m.node_id,
                    "label": t,
                    "pred": p,
                    "prob_1": float(p1),
                    "centrality": float(m.centrality),
                }
            )

    precision, recall, macro_f1 = precision_recall_macro_f1(y_true, y_pred)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    return {"precision": precision, "recall": recall, "macro_f1": macro_f1, "accuracy": acc, "size": len(y_true)}, pred_rows


def save_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Best baseline: T5-base + centrality + ChannelMix (no oversample, no z-score)")
    parser.add_argument("--nodes_csv", type=str, default="demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv")
    parser.add_argument("--centrality_dir", type=str, default="demo/DGCN3/Centrality/alpha_1.5")
    parser.add_argument("--output_dir", type=str, default="demo/tiage-1/results_t5_base_channelmix_best_baseline")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--context_turns", type=int, default=16)
    parser.add_argument("--num_slices", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--channelmix_layers", type=int, default=1)
    parser.add_argument("--channelmix_hidden_mult", type=int, default=4)
    parser.add_argument("--channelmix_dropout", type=float, default=0.1)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(".").resolve()
    nodes_csv = (root / args.nodes_csv).resolve()
    centrality_dir = (root / args.centrality_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] nodes_csv={nodes_csv}")
    print(f"[INFO] centrality_dir={centrality_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] model={args.model_name} context_turns={args.context_turns} batch_size={args.batch_size}")

    rows = parse_nodes(nodes_csv)
    centrality = load_centrality(centrality_dir, args.num_slices)
    split_samples = build_samples(rows, centrality, args.context_turns)
    for split in ("train", "dev", "test"):
        print(f"[INFO] {split} samples={len(split_samples[split])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5Baseline(
        model_name=args.model_name,
        dropout=args.classifier_dropout,
        channelmix_layers=args.channelmix_layers,
        channelmix_hidden_mult=args.channelmix_hidden_mult,
        channelmix_dropout=args.channelmix_dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] device={device}")

    collator = Collator(tokenizer, args.max_length)
    train_loader = DataLoader(TiageDataset(split_samples["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    train_eval_loader = DataLoader(TiageDataset(split_samples["train"]), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    dev_loader = DataLoader(TiageDataset(split_samples["dev"]), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(TiageDataset(split_samples["test"]), batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=max(1, total_steps),
    )

    best_dev = -1.0
    best_score = -1.0
    best_epoch = 0
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            extra_features = batch["extra_features"].to(device)

            loss, _ = model(input_ids, attention_mask, extra_features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        dev_metrics, _ = evaluate(model, dev_loader, device, threshold=args.fixed_threshold)
        avg_loss = total_loss / max(1, len(train_loader))
        print(
            f"[EPOCH {epoch}] train_loss={avg_loss:.4f} "
            f"dev_P={dev_metrics['precision']:.4f} dev_R={dev_metrics['recall']:.4f} "
            f"dev_MacroF1={dev_metrics['macro_f1']:.4f}"
        )
        current_score = dev_metrics["precision"] + dev_metrics["recall"] + dev_metrics["macro_f1"]
        if current_score > best_score:
            best_score = current_score
            best_dev = dev_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    print(f"[INFO] best dev Macro-F1={best_dev:.4f} at epoch={best_epoch} (prf_sum={best_score:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"[INFO] fixed_threshold={args.fixed_threshold:.2f}")

    train_metrics, train_rows = evaluate(model, train_eval_loader, device, threshold=args.fixed_threshold)
    dev_metrics, dev_rows = evaluate(model, dev_loader, device, threshold=args.fixed_threshold)
    test_metrics, test_rows = evaluate(model, test_loader, device, threshold=args.fixed_threshold)

    metrics_rows = [
        {"split": "train", "best_epoch": best_epoch, "best_threshold": args.fixed_threshold, "model_name": args.model_name, "use_channelmix": 1, **train_metrics},
        {"split": "dev", "best_epoch": best_epoch, "best_threshold": args.fixed_threshold, "model_name": args.model_name, "use_channelmix": 1, **dev_metrics},
        {"split": "test", "best_epoch": best_epoch, "best_threshold": args.fixed_threshold, "model_name": args.model_name, "use_channelmix": 1, **test_metrics},
    ]
    save_csv(
        output_dir / "metrics.csv",
        metrics_rows,
        ["split", "best_epoch", "best_threshold", "model_name", "use_channelmix", "precision", "recall", "macro_f1", "accuracy", "size"],
    )
    save_csv(
        output_dir / "predictions.csv",
        train_rows + dev_rows + test_rows,
        ["split", "dialog_id", "turn_id", "node_id", "label", "pred", "prob_1", "centrality"],
    )

    print("[RESULT] Test metrics")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"[SAVED] {output_dir / 'metrics.csv'}")
    print(f"[SAVED] {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
