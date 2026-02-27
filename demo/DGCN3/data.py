import os
import pickle
from typing import Tuple, List
import pandas as pd
import numpy as np
import torch
# from torch_sparse import SparseTensor
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from typing import Set
from networkx.algorithms.community import greedy_modularity_communities
import community as community_louvain
# import leidenalg
# import igraph as ig
# import bayanpy
# 贪心模块度最大化算法
def compute_D_features(G: nx.Graph, num_nodes: int) -> torch.Tensor:
    D = torch.zeros(num_nodes, 3)  # D1, D2, D3

    # 计算 D1：节点的度
    degrees = dict(G.degree())
    for node in G.nodes():
        D[node, 0] = degrees.get(node, 0)

    # 计算 D2：节点的度加上其所有邻居节点的度
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        D2 = D[node, 0] + sum(D[n, 0] for n in neighbors)
        D[node, 1] = D2

    # 计算 D3：节点的 D2 加上其所有邻居节点的 D2
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        D3 = D[node, 1] + sum(D[n, 1] for n in neighbors)
        D[node, 2] = D3

    # 特征归一化
    scaler = MinMaxScaler()
    D = torch.tensor(scaler.fit_transform(D), dtype=torch.float)

    return D

def compute_community_D_features(G: nx.Graph, communities: List[Set[int]], num_nodes: int) -> torch.Tensor:
    comm_D = torch.zeros(num_nodes, 3)

    for comm in communities:
        comm = list(comm)
        subG = G.subgraph(comm)
        D = torch.zeros(len(comm), 3)

        # 计算 D1：节点的度
        degrees = dict(subG.degree())
        for idx, node in enumerate(comm):
            D[idx, 0] = degrees.get(node, 0)

        # 计算 D2：节点的度加上其所有邻居节点的度
        for idx, node in enumerate(comm):
            neighbors = list(subG.neighbors(node))
            D2 = D[idx, 0] + sum(D[comm.index(n), 0] for n in neighbors if n in comm)
            D[idx, 1] = D2

        # 计算 D3：节点的 D2 加上其所有邻居节点的 D2
        for idx, node in enumerate(comm):
            neighbors = list(subG.neighbors(node))
            D3 = D[idx, 1] + sum(D[comm.index(n), 1] for n in neighbors if n in comm)
            D[idx, 2] = D3

        # 特征归一化
        scaler = MinMaxScaler()
        D = torch.tensor(scaler.fit_transform(D), dtype=torch.float)

        for i, node in enumerate(comm):
            comm_D[node] = D[i]

    return comm_D
def compute_H_features(G: nx.Graph, num_nodes: int) -> torch.Tensor:
    H = torch.zeros(num_nodes, 3)  # H1, H2, H3

    # 计算 H1
    h1 = []
    for node in G.nodes:
        degrees = [G.degree(n) for n in G.neighbors(node)]
        degrees.sort(reverse=True)
        h = 0
        for i, deg in enumerate(degrees, 1):
            if deg >= i:
                h = i
            else:
                break
        h1.append(h)
        H[node, 0] = h

    # 计算 H2
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        H[node, 1] = sum([H[n, 0] for n in neighbors])

    # 计算 H3
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        second_neighbors = set()
        for n in neighbors:
            second_neighbors.update(G.neighbors(n))
        second_neighbors.discard(node)  # 移除自身
        H[node, 2] = sum([H[n, 1] for n in second_neighbors])

    # 归一化
    scaler = MinMaxScaler()
    H = torch.tensor(scaler.fit_transform(H), dtype=torch.float)
    return H

def compute_community_H_features(G: nx.Graph, communities: List[Set[int]], num_nodes: int) -> torch.Tensor:
    comm_H = torch.zeros(num_nodes, 3)
    for comm in communities:
        comm = list(comm)
        subG = G.subgraph(comm)
        H = torch.zeros(len(comm), 3)

        # 计算 H1
        for idx, node in enumerate(comm):
            degrees = [subG.degree(n) for n in subG.neighbors(node)]
            degrees.sort(reverse=True)
            h = 0
            for i, deg in enumerate(degrees, 1):
                if deg >= i:
                    h = i
                else:
                    break
            H[idx, 0] = h

        # 计算 H2
        for idx, node in enumerate(comm):
            neighbors = list(subG.neighbors(node))
            H[idx, 1] = sum([H[comm.index(n), 0] for n in neighbors if n in comm])

        # 计算 H3
        for idx, node in enumerate(comm):
            neighbors = list(subG.neighbors(node))
            second_neighbors = set()
            for n in neighbors:
                second_neighbors.update(subG.neighbors(n))
            second_neighbors.discard(node)
            H[idx, 2] = sum([H[comm.index(n), 1] for n in second_neighbors if n in comm])

        # 归一化
        scaler = MinMaxScaler()
        H = torch.tensor(scaler.fit_transform(H), dtype=torch.float)

        for i, node in enumerate(comm):
            comm_H[node] = H[i]

    return comm_H

def preprocess_raw_data(
        raw_dataset_dir: str,
        processed_dataset_dir: str,
        sir_results_base_dir: str,  # 只传入 SIR 结果的根目录
        a: float
) -> None:
    """
    Preprocess raw dataset files into a format suitable for use with PyTorch Geometric.

    Args:
        raw_dataset_dir (str): Directory containing raw dataset files.
        processed_dataset_dir (str): Directory where processed dataset files will be saved.
        sir_results_base_dir (str): Root directory of SIR values.

    Returns:
        None
    """
    raw_dataset_files = sorted(os.listdir(raw_dataset_dir),
                               key=lambda x: int(x.split('_')[1].split('.txt')[0]))

    # 初始化数据集结构
    nodes_set = set()  # 所有节点的集合
    timestamp_list = []  # 每个时间步的时间戳
    edge_index_list = []  # 每个时间步的边索引
    sir_values_dict = {}  # 每个时间步的 SIR 结果，使用字典存储

    # 获取数据集名称
    dataset_name = os.path.basename(raw_dataset_dir)

    # 遍历所有时间步
    for t, file in enumerate(raw_dataset_files):
        # if  t >= 3:
        #     continue
        file_path = os.path.join(raw_dataset_dir, file)
        timestamp_list.append(t)

        # 读取边数据
        with open(file_path) as f:
            print(f'[*] Reading the file {file_path}...')
            lines = f.readlines()

        edge_index = []
        for line in lines:
            # i, j = map(int, line.strip().split('\t'))
            i, j = map(int, line.strip().split())
            nodes_set.add(i)
            nodes_set.add(j)
            if i != j:
                edge_index.append([i, j])
        edge_index_list.append(edge_index)

        # 获取 SIR 结果路径
        # print(sir_results_base_dir)
        sir_results_dataset_dir = os.path.join(sir_results_base_dir, dataset_name)
        # print(sir_results_dataset_dir)
        sir_results_file = os.path.join(sir_results_dataset_dir, f"{dataset_name}_{t}_a[{a}].csv")
        # print(sir_results_file)

        if not os.path.exists(sir_results_file):
            raise FileNotFoundError(f"[*] SIR results file not found: {sir_results_file}")

        # 读取 SIR 结果文件
        sir_data = pd.read_csv(sir_results_file)  # 假设 CSV 文件有表头
        sir_values = sir_data.set_index('Node')['SIR'].to_dict()  # 将 SIR 值转换为字典
        sir_values_dict[t] = sir_values

    num_nodes = max(nodes_set) + 1

    # 构造静态图 `Data`
    for t, edge_index in enumerate(edge_index_list):
        print(f'[*] Constructing the static graph data object for timestep {t}...')
        edge_index = np.array(edge_index).T
        source_nodes = set(edge_index[0])
        target_nodes = set(edge_index[1])
        node_index = np.array(sorted(source_nodes.union(target_nodes)))
        node_mask = np.zeros(num_nodes, dtype=bool)
        node_mask[node_index] = True

        # 构造图 G
        G = nx.Graph()
        G.add_edges_from(edge_index.T.tolist())

        # 计算节点的结构特征
        deg_features = compute_D_features(G, num_nodes)
        h_features = compute_H_features(G, num_nodes)

        # 社团检测
        # # 将NetworkX图转换为igraph图
        # G_ig = ig.Graph.from_networkx(G)
        # # 运行Leiden算法
        # # 获取 igraph 节点编号与 NetworkX 节点编号的映射
        # # 在转换过程中，igraph 会自动为每个节点分配一个唯一的整数编号
        # # 我们需要手动保存这个映射关系
        # node_mapping = {i: node for i, node in enumerate(G.nodes())}
        # # 运行 Leiden 算法
        # partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
        # # 提取社团的节点列表
        # communities_ig = [list(comm) for comm in partition]
        # # 将 igraph 的节点编号重新映射回 NetworkX 的节点编号
        # communities = [[node_mapping[node] for node in comm] for comm in communities_ig]

        # # 运行 Louvain 算法
        partition = community_louvain.best_partition(G)
        # 将 Louvain 算法的结果转换为与 greedy_modularity_communities 相同的格式
        # 创建一个字典，键为社团编号，值为该社团的节点集合
        community_dict = {}
        for node, community_id in partition.items():
            if community_id not in community_dict:
                community_dict[community_id] = set()
            community_dict[community_id].add(node)
        # 将社团字典转换为列表，每个元素是一个集合
        communities = list(community_dict.values())
        # _,  _, communities, _, _ = bayanpy.bayan(G, threshold=0.001,time_allowed=60, resolution=1)

        # communities = list(greedy_modularity_communities(G))

        # 计算社团内的结构特征
        comm_deg_features = compute_community_D_features(G, communities, num_nodes)
        comm_h_features = compute_community_H_features(G, communities, num_nodes)

        # 原始单位特征
        # onehot_tensor = torch.eye(num_nodes)

        # 最终拼接特征
        # final_x = torch.cat([onehot_tensor, deg_features, comm_deg_features,h_features,comm_h_features], dim=1)
        final_x = torch.cat([ deg_features, comm_deg_features, h_features, comm_h_features], dim=1)

        # 获取当前时间步的 SIR 值
        sir_values = sir_values_dict[t]
        sir_tensor = torch.zeros(num_nodes, dtype=torch.float)
        for node, value in sir_values.items():
            sir_tensor[node] = value
        # #对SIR归一化
        # scaler = MinMaxScaler()
        # sir_tensor_np = sir_tensor.numpy().reshape(-1, 1)
        # sir_tensor_np = scaler.fit_transform(sir_tensor_np)
        # sir_tensor = torch.tensor(sir_tensor_np.flatten(), dtype=torch.float)

        static_graph = Data(
            x=final_x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_mask=torch.tensor(node_mask, dtype=torch.bool),
            edge_count=edge_index.shape[1],
            timestep=t,
            timestamp=timestamp_list[t],
            y=sir_tensor  # 添加 SIR 值作为标签
        )

        file_name = f'{static_graph.timestamp}.pickle'
        print(f"[*] Saving the processed graph data {os.path.join(processed_dataset_dir, file_name)}...")
        with open(os.path.join(processed_dataset_dir, file_name), 'wb') as handle:
            pickle.dump(static_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_data(
        dataset_name: str,
        train_test_ratio: float,
        device: torch.device,
        a: float
) -> Tuple[List[Data], List[int], List[int]]:
    """
       加载和预处理数据集，并划分为训练集和测试集。

       参数:
           dataset_name (str): 要加载和处理的数据集名称。
           train_test_ratio (float): 用于训练的数据集比例（其余部分用于测试）。
           device (torch.device): 张量应移动到的设备（CPU 或 GPU）。

       返回:
           Tuple[List[Data], List[int], List[int]]:
               - List[Data]: 每个时间步的 PyTorch Geometric Data 对象列表。
               - List[int]: 训练时间步列表。
               - List[int]: 测试时间步列表。
    """
    print(f"=========== Loading the dataset: {dataset_name} ===========")
    raw_dataset_dir = os.path.join("datasets", "raw_data", dataset_name)
    processed_dataset_dir = os.path.join("datasets", "processed_data", dataset_name)
    sir_results_base_dir = os.path.join("sir_results")  # 只传入根目录

    os.makedirs(processed_dataset_dir, exist_ok=True)

    # 预处理原始数据  原本的论文逻辑
    # if len(os.listdir(processed_dataset_dir)) != 0:
    #     print(f"[*] The {dataset_name} dataset's raw txt files are already processed.")
    # else:
    #     print(f"[*] Preprocessing {dataset_name} raw txt files...")
    #     preprocess_raw_data(
    #         raw_dataset_dir=raw_dataset_dir,
    #         processed_dataset_dir=processed_dataset_dir,
    #         sir_results_base_dir=sir_results_base_dir , # 传入 SIR 结果目录
    #         a = a
    #     )
    # print(f"[*] Preprocessing {dataset_name} raw txt files...")
    preprocess_raw_data(
        raw_dataset_dir=raw_dataset_dir,
        processed_dataset_dir=processed_dataset_dir,
        sir_results_base_dir=sir_results_base_dir , # 传入 SIR 结果目录
        a = a
    )

    # 读取已处理的 pickle 文件
    pickle_files = sorted(os.listdir(processed_dataset_dir), key=lambda x: int(x.split('.')[0]))
    dataset = []
    x_in_list = []
    edge_index_list = []

    for file in pickle_files:
        with open(os.path.join(processed_dataset_dir, file), 'rb') as handle:
            snapshot_graph = pickle.load(handle)
        # # 将节点特征转换为单位矩阵，并移动到指定设备
        # snapshot_graph.x = torch.eye(snapshot_graph.x.size(0)).to(device=device)
        # # 将节点特征转换为稀疏张量，并移动到指定设备
        # snapshot_graph.x = SparseTensor.from_dense(snapshot_graph.x).to(device=device)
        snapshot_graph.x = snapshot_graph.x.to(device=device)
        # 将边索引移动到指定设备
        snapshot_graph.edge_index = snapshot_graph.edge_index.to(device=device)
        # 确保 `y` 也移动到设备
        snapshot_graph.y = snapshot_graph.y.to(device=device)
        # 将节点特征和边索引添加到列表中
        x_in_list.append(snapshot_graph.x)
        edge_index_list.append(snapshot_graph.edge_index)
        # 将图数据添加到数据集中
        dataset.append(snapshot_graph)
    # 根据训练测试比例划分数据集
    # train_test_ratio = 2 / 3
    # train_test_ratio = 1 / 2
    len_test_dataset = int(train_test_ratio * len(dataset))
    train_dataset = dataset[:-len_test_dataset]
    test_dataset = dataset[-len_test_dataset:]
    train_timesteps = [data.timestep for data in train_dataset]
    test_timesteps = [data.timestep for data in test_dataset]

    print(f"Total dataset size: {len(dataset)}")
    print(f"Test dataset size: {len_test_dataset}")
    print(f"Train dataset size: {len(dataset) - len_test_dataset}")
    # 输出训练集和测试集的详细信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(f"Train timesteps: {train_timesteps}")
    print(f"Test timesteps: {test_timesteps}")

    return dataset, train_timesteps, test_timesteps
