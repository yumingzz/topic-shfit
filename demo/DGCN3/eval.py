from typing import List
from torch_geometric.data import Data
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import kendalltau
# def evaluate(
#         test_probs: List[torch.Tensor],
#         test_timesteps: List[int],
#         dataset: List[Data]
# ) -> pd.DataFrame:
#     def map_at_k(y_true: torch.Tensor, y_pred: torch.Tensor, k: int) -> float:
#         k = min(k, len(y_true))
#         pred_topk = torch.topk(y_pred, k).indices
#         true_topk = torch.topk(y_true, k).indices
#         hits = 0
#         avg_precisions = []
#         for i, idx in enumerate(pred_topk):
#             if idx in true_topk:
#                 hits += 1
#                 avg_precisions.append(hits / (i + 1))
#         return sum(avg_precisions) / k if avg_precisions else 0.0
#
#     results = {
#         'Timestep': [],
#         'KendallTau': [],
#         'MAP@10': [],
#         'MAP@20': []
#     }
#
#     for pred, t in zip(test_probs, test_timesteps):
#         y_true = dataset[t].y
#         mask = dataset[t].node_mask
#         y_true = y_true[mask]
#         y_pred = pred[mask]
#
#         tau, _ = kendalltau(y_true.cpu().numpy(), y_pred.cpu().numpy())
#         map_10 = map_at_k(y_true, y_pred, 10)
#         map_20 = map_at_k(y_true, y_pred, 20)
#
#         results["Timestep"].append(t)
#         results["KendallTau"].append(tau)
#         results["MAP@10"].append(map_10)
#         results["MAP@20"].append(map_20)
#
#     return pd.DataFrame(results)
#
def evaluate(
        test_probs: List[torch.Tensor],
        test_timesteps: List[int],
        dataset: List[Data]
) -> pd.DataFrame:
    from scipy.stats import kendalltau
    results = {
        'Timestep': [],
        # 'MAE': [],
        # 'MSE': [],
        # 'R2': [],
        'KendallTau': [],
        'MAP': []
    }

    for pred, t in zip(test_probs, test_timesteps):
        y_true = dataset[t].y
        mask = dataset[t].node_mask
        y_true = y_true[mask]
        y_pred = pred[mask]

        # mae = F.l1_loss(y_pred, y_true).item()
        # mse = F.mse_loss(y_pred, y_true).item()
        # r2 = 1 - ((y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()).item()
        tau, _ = kendalltau(y_true.cpu().numpy(), y_pred.cpu().numpy())

        # ✅ 计算 MAP（将 SIR 值转为二分类标签以表示“重要/不重要”）
        # 你可以按需自定义阈值，这里默认将 top 10% 作为正例
        top_k = max(1, int(0.1 * len(y_true)))
        y_true_bin = torch.zeros_like(y_true)
        top_indices = torch.topk(y_true, top_k).indices
        y_true_bin[top_indices] = 1

        map_score = average_precision_score(y_true_bin.cpu().numpy(), y_pred.cpu().numpy())

        results["Timestep"].append(t)
        # results["MAE"].append(mae)
        # results["MSE"].append(mse)
        # results["R2"].append(r2)
        results["KendallTau"].append(tau)
        results["MAP"].append(map_score)

    return pd.DataFrame(results)

