import os
import pandas as pd
import networkx as nx
import numpy as np
def split_txt_by_timestamp(input_file_path: str, output_dir: str) -> None:
    """
    读取一个 .txt 文件，根据第三列（时间戳）将数据分割成多个子文件，
    删除第四列（权重）和时间戳列，并按时间戳命名输出文件。

    Args:
        input_file_path (str): 输入的 .txt 文件路径。
        output_dir (str): 输出文件保存的目录。
    """
    # 读取 .txt 文件
    df = pd.read_csv(input_file_path, header=None, names=["start", "end", "timestamp", "weight"])

    # 检查输出目录是否存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)

    # 按时间戳分组
    grouped = df.groupby("timestamp")

    # 遍历每个时间戳分组
    for timestamp, group in grouped:
        # 删除权重列和时间戳列
        group = group.drop(columns=["weight", "timestamp"])

        # 定义输出文件名
        output_file_name = f"Dynamic_PPIN_{timestamp}.txt"
        output_file_path = os.path.join(output_dir, output_file_name)

        # 保存到新的 .txt 文件
        group.to_csv(output_file_path, index=False, header=False, sep=",")
        print(f"Saved {output_file_path} with {len(group)} rows.")
def rename_and_move_txt_files(source_dir: str, target_dir: str, base_name: str = "Gnu_", start_index: int = 3) -> None:
    """
    重命名源目录中的所有 .txt 文件，并将它们移动到目标目录。

    参数:
        source_dir (str): 包含原始 .txt 文件的源目录路径。
        target_dir (str): 重命名后的文件将被移动到的目标目录路径。
        base_name (str): 新文件名的前缀，默认为 "Gnu_"。
        start_index (int): 文件编号的起始值，默认为 3。
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取源目录下的所有文件名
    files = os.listdir(source_dir)

    # 筛选出所有 .txt 文件
    txt_files = [file for file in files if file.endswith('.txt')]

    # 遍历所有 .txt 文件并重命名、移动
    for i, txt_file in enumerate(txt_files, start=start_index):
        # 构造新的文件名
        new_file_name = f"{base_name}{i}.txt"

        # 构造完整的旧文件路径和新文件路径
        old_file_path = os.path.join(source_dir, txt_file)
        new_file_path = os.path.join(target_dir, new_file_name)

        # 检查目标文件是否已存在
        if os.path.exists(new_file_path):
            print(f"File {new_file_name} already exists in {target_dir}. Skipping rename.")
            continue

        # 移动并重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{txt_file}' to '{new_file_name}' and moved to {target_dir}")


# 示例调用
# source_directory = r"D:\lesson\Postgraduate\EXcode\DGCN3\MRCNN"
# target_directory = r"D:\lesson\Postgraduate\EXcode\DGCN3\RenamedFiles"
# rename_and_move_txt_files(source_directory, target_directory)

def generate_ba_snapshots(n_stage1, n_total, m, output_dir):
    """
    生成两个 BA 网络时间快照：
    - 第一个快照包含 n_stage1 个节点
    - 第二个快照扩展到 n_total 个节点
    - 每个新节点连接 m 条边
    - 结果保存在 output_dir 中，文件名为 BA_0.txt 和 BA_1.txt

    参数：
    - n_stage1: 第一阶段的节点数量
    - n_total: 总节点数量（第二阶段）
    - m: 每个新节点连接的边数
    - output_dir: 保存文件的目录
    """

    os.makedirs(output_dir, exist_ok=True)

    # 第一阶段图
    G_t0 = nx.barabasi_albert_graph(n_stage1, m)

    # 第二阶段图，扩展自第一阶段
    G_t1 = G_t0.copy()
    for new_node in range(n_stage1, n_total):
        degrees = np.array([deg for _, deg in G_t1.degree()])
        total_deg = degrees.sum()
        probs = degrees / total_deg  # 计算每个节点被选择的概率
        targets = np.random.choice(list(G_t1.nodes()), size=m, replace=False, p=probs)  # 无放回采样

        # 将新节点与采样的目标节点连接
        for target in targets:
            G_t1.add_edge(new_node, target)

    # 保存边列表
    nx.write_edgelist(G_t0, os.path.join(output_dir, "BA_0.txt"), data=False)
    nx.write_edgelist(G_t1, os.path.join(output_dir, "BA_1.txt"), data=False)

    print(f"已保存快照到目录：{output_dir}")


import networkx as nx
import numpy as np
import os
import random


def generate_dynamic_ba_snapshots(num_snapshots, max_nodes, m, output_dir, min_change=20, max_change=100):
    """
    生成动态BA图，每个快照最多 max_nodes 个节点，节点数可上下浮动。

    参数：
        num_snapshots: 快照数量（如 10）
        max_nodes: 每个快照最大节点数（如 1000）
        m: 每个新节点连接的边数
        output_dir: 保存目录
        min_change, max_change: 每次变动的节点数范围
    """
    os.makedirs(output_dir, exist_ok=True)

    current_node_count = random.randint(int(max_nodes * 0.6), max_nodes)
    next_node_id = current_node_count
    G = nx.barabasi_albert_graph(current_node_count, m)

    for i in range(num_snapshots):
        snapshot_file = os.path.join(output_dir, f"BA_{i}.txt")

        # 随机决定本轮是增加节点还是删除部分节点
        change = random.randint(min_change, max_change)
        if random.random() < 0.5 and current_node_count + change <= max_nodes:
            # 增加节点
            for _ in range(change):
                probs = np.array([G.degree(n) for n in G.nodes()], dtype=np.float64)
                probs /= probs.sum()
                targets = list(np.random.choice(list(G.nodes()), size=m, replace=False, p=probs))
                G.add_node(next_node_id)
                for target in targets:
                    G.add_edge(next_node_id, target)
                next_node_id += 1
                current_node_count += 1
        else:
            # 删除节点（只删除孤立或随机的一部分）
            removable_nodes = list(G.nodes())
            random.shuffle(removable_nodes)
            removed = 0
            for node in removable_nodes:
                if current_node_count - removed <= m + 1:
                    break  # 保证能连 m 条边
                G.remove_node(node)
                removed += 1
                current_node_count -= 1
                if removed >= change:
                    break

        # 写入当前快照
        with open(snapshot_file, "w") as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")

        print(f"Snapshot BA_{i} saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# 示例调用：
# generate_ba_snapshots(2000, 8000, 5, "BA-8K")


