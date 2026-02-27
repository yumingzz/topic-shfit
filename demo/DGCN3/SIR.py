import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# 定义数据集目录
# datasets = {
#     "colab": "dataset/raw_data/colab/",
#     "enron": "dataset/raw_data/enron/",
#     "facebook": "dataset/raw_data/facebook/"
# }
#
# # 定义输出目录
# output_dir_base = "sir_results/"
# os.makedirs(output_dir_base, exist_ok=True)


# 获取时间步（获取目录下所有 txt 文件）
def get_time_steps(raw_data_dir):
    return sorted([f for f in os.listdir(raw_data_dir) if f.endswith(".txt")])


# 加载无向图
def load_graph_from_file(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                u, v = map(int, parts[:2])
                G.add_edge(u, v)
    return G


# SIR 传播模型
def SIR(G, infected, beta, miu=1, iterations=1000):
    """ 计算单个节点的 SIR 影响力 """
    re = 0
    for _ in range(iterations):
        inf = set(infected)  # 初始感染集合
        R = set()  # 恢复集合

        while inf:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    if random.uniform(0, 1) < beta and j not in inf and j not in R:
                        newInf.append(j)

                # 判断是否恢复
                if random.uniform(0, 1) > miu:
                    newInf.append(i)
                else:
                    R.add(i)

            inf = set(newInf)

        re += len(R) + len(inf)

    return re / iterations


# 计算整个网络所有节点的 SIR 影响力
def SIR_dict(G, beta=0.1, miu=1, real_beta=None, a=1.5):
    node_list = list(G.nodes())
    SIR_dic = {}

    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
        beta = a * (dc_list.mean() / (dc_list.var() + dc_list.mean()))

    print(f"✅ 计算传播概率 beta: {beta}")

    for node in tqdm(node_list):
        SIR_dic[node] = SIR(G, infected=[node], beta=beta, miu=miu)

    return SIR_dic


# 存储 SIR 计算结果
def save_sir_dict(dic, path):
    df = pd.DataFrame({'Node': list(dic.keys()), 'SIR': list(dic.values())})
    df.to_csv(path, index=False)


# 处理所有时间步的 SIR 计算
def process_all_time_steps(datasets, output_dir_base, a_list):
    for dataset_name, raw_data_dir in datasets.items():
        print(f"🚀 处理数据集: {dataset_name}")
        output_dir = os.path.join(output_dir_base, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        time_steps = get_time_steps(raw_data_dir)
        print(f"📂 发现 {len(time_steps)} 个文件: {time_steps}")

        for file_name in time_steps:
            file_path = os.path.join(raw_data_dir, file_name)
            G = load_graph_from_file(file_path)

            for idx, a in enumerate(a_list):
                print(f"📌 处理 {file_name}, 传播参数 a={a} ...")
                sir_dict = SIR_dict(G, real_beta=True, a=a)
                output_file = os.path.join(output_dir, f"{file_name.replace('.txt', '')}_a[{a}].csv")

                save_sir_dict(sir_dict, output_file)
                print(f"✅ 结果已保存至 {output_file}")



# 运行计算
# process_all_time_steps(datasets, output_dir_base)
# print("🎉 所有数据集的 SIR 计算完成！")