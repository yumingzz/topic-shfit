import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import List

# 计算成对排序损失，用于节点重要性预测任务
def pairwise_ranking_loss(pred_scores, true_scores, mask):
    pred_scores = pred_scores[mask]
    true_scores = true_scores[mask]

    # 计算分数差异
    diff_true = true_scores.unsqueeze(1) - true_scores.unsqueeze(0)
    diff_pred = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)

    # # 计算成对排序的 MAE 损失
    # # MAE 损失计算预测差异和真实差异之间的绝对误差
    # loss = F.l1_loss(diff_pred, diff_true)

    # 计算成对排序损失
    S_ij = (diff_true > 0).float()
    loss = F.binary_cross_entropy_with_logits(diff_pred, S_ij)
    return loss

class MPNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(MPNN, self).__init__()
        # 定义三层 GAT
        self.mp1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
        self.mp2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.mp3 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False)

        # 批量归一化层
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)

        # self.mp1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        # self.mp2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, concat=True)
        # self.mp3 = GATConv(hidden_dim * 8, output_dim, heads=1, concat=False)
        #
        # # 批量归一化层
        # self.bn1 = nn.BatchNorm1d(num_features=hidden_dim * 8)
        # self.bn2 = nn.BatchNorm1d(num_features=hidden_dim * 8)
        # self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)

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
            z = F.normalize(z, p=2., dim=-1)
        z = F.dropout(z, p=0.1, training=self.training)
        return z


class NodeImportanceModel(nn.Module):
    """
    NodeImportanceModel: 静态图节点重要性预测模型。
    该模型结合了图神经网络（GNN）和对比学习机制，用于学习图中节点的表示，并预测节点的重要性。
    """
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ):
        """
        初始化 NodeImportanceModel 模型。
        参数:
            input_dim (int): 输入特征的维度。
            hidden_dim (int): 隐藏层的维度。
            output_dim (int): 输出特征的维度。
        """
        super(NodeImportanceModel, self).__init__()

        # 图自编码器，使用 MPNN 作为编码器
        self.encoder = MPNN(input_dim, hidden_dim, output_dim)

        # 全连接层，用于预测节点重要性
        self.importance_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 局部和全局对比学习编码器
        self.local_predictive_encoder = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.global_predictive_encoder = nn.Linear(output_dim * 2, output_dim)

        # 用于计算对比学习损失
        # self.mse_loss = nn.MSELoss()

    def forward(self, graph: Data, normalize: bool = False) -> torch.Tensor:
        """
        前向传播，用于预测节点重要性。
        参数:
            graph (Data): 图数据，包含特征 x 和边 index。
            normalize (bool): 是否对节点嵌入进行归一化。
        返回:
            torch.Tensor: 预测的节点重要性。
        """
        x = graph.x
        edge_index = graph.edge_index
        # 打印x和edge_index的类型和形状
        # 打印x和edge_index的类型和形状
        # print(f"x type: {type(x)}, x size: {x.size()}")
        # 使用 MPNN 编码图中的节点特征
        z = self.encoder(x, edge_index, normalize)

        # 使用节点表示预测节点重要性
        y_pred = self.importance_predictor(z).squeeze()

        return y_pred

    def compute_losses(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        基于节点嵌入和预测值的对比学习损失。
        参数:
            graph (Data): 包含图结构和标签。
            node_embeddings (torch.Tensor): 编码器输出的节点嵌入 (z)。
        返回:
            torch.Tensor: 对比学习损失。
        """
        # state_dec 是预测值，可以视作“目标向量”
        state_dec = self.importance_predictor(node_embeddings).detach()  # [N, 1]
        state_dec = state_dec.expand_as(node_embeddings)  # 广播为 [N, D]，使得能与 z 拼接

        # 构造正样本对 (z, importance) => concat
        pos_pair = torch.cat([state_dec, node_embeddings], dim=-1)  # [N, 2D]
        z_local = self.local_predictive_encoder(pos_pair)  # [N, D]
        z_global = self.global_predictive_encoder(pos_pair)  # [N, D]

        # 正样本相似度
        pos_sim = F.cosine_similarity(z_local, z_global, dim=-1)  # [N]

        # 构造负样本对（importance + 打乱的 z）
        neg_idx = torch.randperm(node_embeddings.size(0))
        neg_embeds = node_embeddings[neg_idx]
        neg_pair = torch.cat([state_dec, neg_embeds], dim=-1)  # [N, 2D]
        z_local_neg = self.local_predictive_encoder(neg_pair)  # [N, D]
        z_global_neg = self.global_predictive_encoder(neg_pair)  # [N, D]

        # 负样本相似度
        neg_sim = F.cosine_similarity(z_local_neg, z_global_neg, dim=-1)  # [N]

        # 对比损失（InfoNCE 形式）
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)  # [N, 2]
        labels = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=logits.device)  # 正样本在第0列

        loss = F.cross_entropy(logits, labels)
        return loss

    def compute_total_loss(self, graph: Data, normalize: bool = False) -> torch.Tensor:
        """
        计算总损失：主任务损失 + 对比学习损失
        参数:
            graph (Data): 图数据，包含特征 x 和边 index。
            normalize (bool): 是否对节点嵌入进行归一化。
        返回:
            torch.Tensor: 总损失
        """
        # 获取节点重要性预测
        x = graph.x
        edge_index = graph.edge_index
        z = self.encoder(x, edge_index, normalize=normalize)

        # 用节点嵌入做预测
        y_pred = self.importance_predictor(z).squeeze()

        y_true = graph.y
        mask = graph.node_mask

        # 计算节点重要性损失（成对排序损失）
        importance_loss = pairwise_ranking_loss(y_pred, y_true, mask)

        # 计算对比学习损失
        contrastive_loss = self.compute_losses(z)

        # 总损失：主任务 + 辅助任务
        total_loss = importance_loss + 0.5 * contrastive_loss
        # total_loss = importance_loss + 0.25 * contrastive_loss
        # total_loss = importance_loss + 0.75 * contrastive_loss
        # total_loss = importance_loss + contrastive_loss
        # print(importance_loss)
        # print(contrastive_loss)
        # total_loss = importance_loss
        return total_loss

    def predict(self, graph: Data, normalize: bool = False) -> torch.Tensor:
        """
        使用训练好的模型进行节点重要性预测。
        参数:
            graph (Data): 图数据，包含特征 x 和边 index。
            snapshot_sequence (Optional): 可选参数，可能是时间步的历史数据。
            normalize (bool): 是否对节点嵌入进行归一化。
        返回:
            torch.Tensor: 预测的节点重要性（SIR）。
        """
        y_pred = self(graph, normalize)
        return y_pred

