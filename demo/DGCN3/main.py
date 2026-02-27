import os
import random
import time
import argparse
import configparser
from typing import List, Dict, Union
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from data import get_data
from model import NodeImportanceModel  # 使用新的模型类
from eval import evaluate
from itertools import chain, repeat
from pre_data import split_txt_by_timestamp,rename_and_move_txt_files,generate_ba_snapshots,generate_dynamic_ba_snapshots
from SIR import process_all_time_steps
# Setting random seeds for reproducibility
random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
import csv
import networkx as nx
from data import compute_D_features
from data import compute_H_features
import community as community_louvain
from data import compute_community_D_features
from data import compute_community_H_features

def train(
        model: torch.nn.Module,
        train_dataset: List[Data],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        hparams: Dict[str, Union[int, float]],
        model_path: str
) -> torch.nn.Module:
    """
    使用提供的训练数据集训练给定的模型。

    参数:
        model (torch.nn.Module): 要训练的模型。
        train_dataset (List[Data]): 用于训练的 PyTorch Geometric Data 对象列表。
        optimizer (torch.optim.Optimizer): 训练中使用的优化器。
        scheduler (torch.optim.lr_scheduler.LRScheduler): 学习率调度器。
        hparams (Dict[str, Union[int, float]]): 训练的超参数，包括训练周期数、alpha、beta 等。
        model_path (str): 保存训练后模型的路径。

    返回:
        torch.nn.Module: 训练后的模型。
    """
    print(f"=========== Train ===========")
    best_train_loss = float('inf')
    model.train()
    for epoch in range(hparams["epochs"]):
        start_time = time.time()
        total_loss = 0
        for data in train_dataset:
            optimizer.zero_grad()
            loss = model.compute_total_loss(data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print("[*] epoch: {}, loss: {:.4f}, time: {:.1f}".format(epoch, avg_loss, time.time() - start_time))

        scheduler.step(avg_loss)

        # 如果当前周期的损失值低于最佳损失值，则更新最佳损失值并保存模型
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            print('[*] --> Best training loss {:.4f} reached at epoch {}.'.format(avg_loss, epoch))
            print(f"[*] --> Saving the model {model_path}")
            torch.save(model.state_dict(), model_path)

    print("[*] Training is completed.")
    return model


def load_hparams() -> Dict[str, Union[int, float]]:
    """
    从配置文件中加载超参数。

    从与脚本位于同一目录下的 "config.ini" 文件中读取超参数。
    超参数应位于文件的 "hyperparameters" 部分。

    返回:
        Dict[str, Union[int, float]]: 包含超参数的字典，键包括：
            - "epochs": 训练的周期数。
            - "train_test_ratio": 训练数据与测试数据的比例。
            - "hidden_dim": 隐藏层的维度。
            - "output_dim": 输出层的维度。
            - "alpha": 图重建损失的权重。
            - "beta": 对比预测编码损失的权重。
            - "learning_rate": 优化器的学习率。
            - "weight_decay": 优化器的权重衰减（L2正则化）。
            - "scheduler_patience": 学习率调度器在多少个周期内没有改进后会降低学习率。
            - "scheduler_factor": 学习率调度器降低学习率的因子。
            - "scheduler_min_lr": 学习率调度器允许的最小学习率。

    异常:
        configparser.NoSectionError: 如果配置文件中缺少 "hyperparameters" 部分。
        configparser.NoOptionError: 如果 "hyperparameters" 部分中缺少任何预期的选项。
    """
    hparams = configparser.ConfigParser()
    hparams.read("config.ini")

    hparams = {
        "epochs": hparams.getint("hyperparameters", "EPOCHS"),
        "train_test_ratio": hparams.getfloat("hyperparameters", "TRAIN_TEST_RATIO"),
        "hidden_dim": hparams.getint("hyperparameters", "HIDDEN_DIM"),
        "output_dim": hparams.getint("hyperparameters", "OUTPUT_DIM"),
        "learning_rate": hparams.getfloat("hyperparameters", "LEARNING_RATE"),
        "weight_decay": hparams.getfloat("hyperparameters", "WEIGHT_DECAY"),
        "scheduler_patience": hparams.getint("hyperparameters", "SCHEDULER_PATIENCE"),
        "scheduler_factor": hparams.getfloat("hyperparameters", "SCHEDULER_FACTOR"),
        "scheduler_min_lr": hparams.getfloat("hyperparameters", "SCHEDULER_MIN_LR"),
    }
    return hparams


def inference(
    model: torch.nn.Module,
    dataset: List[Data],
    test_timesteps: List[int],
    model_path: str,
    device: torch.device
) -> List[torch.Tensor]:
    """
    使用模型对每个时间步滑动窗口预测下一个时间步的节点重要性（SIR）。

    返回:
        List[torch.Tensor]: 每个测试时间步的预测张量列表。
    """
    print(f"[*] Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_preds = []
    for t in test_timesteps:
        if t == 0:
            continue  # t=0 没有历史，跳过

        print(f"[*] Predicting timestep {t} using history [0:{t}]...")
        with torch.no_grad():
            y_pred = model.predict(graph=dataset[t], normalize=True)
        test_preds.append(y_pred.cpu())
    return test_preds


def main():
    """
    主函数，运行训练和推理流程。

    1. 配置路径和超参数。
    2. 加载数据集。
    3. 初始化模型、优化器和学习率调度器。
    4. 训练模型。
    5. 对测试集进行推理。
    6. 评估测试结果。
    """
    # 配置数据集和模型路径
    parser = argparse.ArgumentParser(description='Process dataset name.')
    parser.add_argument(
        '--dataset_name',
        default='tiage',
        choices=['enron', 'facebook', 'colab','BA','DBLP5','reddit','birth','expansion','merge','DBLP3','tiage'],
        help='Specify the dataset name (options: enron, facebook, colab). Default is "enron".'
    )
    args = parser.parse_args()

    # 读取数据集名称参数
    dataset_name = args.dataset_name
    print(f'[*] Dataset name selected: {dataset_name}')
    model_dir = os.path.join("model_registry")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"node_importance_{dataset_name}.pkl")

    # 设置设备（优先使用 GPU，否则使用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[*] Device selected: {device}')

    # 加载超参数
    hparams = load_hparams()
    # 循环测试不同a下面的效果
    # a_list = [1.5, 1.1, 1.2, 1.3, 1.4, 1.0, 1.6, 1.7, 1.8, 1.9]
    a_list = [1.5, 1.1, 1.2, 1.3, 1.4, 1.0, 1.6, 1.7, 1.8,1.9]
    df = pd.DataFrame(columns=["Timestep", "KendallTau", "MAP@10", "MAP@20"])
    df = pd.DataFrame(columns=["Timestep", "KendallTau", "MAP"])
    for i in range(0, len(a_list)):
        # 加载数据集
        dataset, train_timesteps, test_timesteps = get_data(dataset_name=dataset_name,
                                                            train_test_ratio=hparams["train_test_ratio"],
                                                            device=device, a=a_list[i])
        # print(test_timesteps)
        train_dataset = [dataset[k] for k in train_timesteps]

        INPUT_DIM = dataset[0].x.size(1)

        # 初始化模型、优化器和学习率调度器
        model = NodeImportanceModel(input_dim=INPUT_DIM,
                                    hidden_dim=hparams["hidden_dim"],
                                    output_dim=hparams["output_dim"]).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hparams["learning_rate"],
                                     weight_decay=hparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=hparams["scheduler_patience"],
            factor=hparams["scheduler_factor"],
            min_lr=hparams["scheduler_min_lr"]
        )
        if i == 0:
            # 训练模型
            model = train(model=model,
                          train_dataset=train_dataset,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          hparams=hparams,
                          model_path=model_path)

        # 对测试集进行推理
        test_probs = inference(model=model,
                               dataset=dataset,
                               test_timesteps=test_timesteps,
                               model_path=model_path,
                               device=device)

        # 评估测试结果
        test_results = evaluate(test_probs=test_probs,
                                test_timesteps=test_timesteps,
                                dataset=dataset)
        print(test_results)
        df = pd.concat([df, test_results], ignore_index=True)
        print(df)
    # 使用 itertools.chain 和 itertools.repeat 重复每个元素三次
    t_values = list(chain.from_iterable(repeat(item, len(test_timesteps)) for item in a_list))
    # 添加新列 a
    df['a'] = t_values
    # 按照列 'a' 降序排列
    df_sorted = df.sort_values(by='a', ascending=True)
    # csv_file = 'tau/'  + dataset_name + '.csv'
    csv_file = '学习率0.05/' + dataset_name + '.csv'
    # csv_file = '参数0.25/' + dataset_name + '.csv'
    df_sorted.to_csv(csv_file, index=False)
    #
    # learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # df_all = pd.DataFrame(columns=["LearningRate", "Timestep", "KendallTau", "MAP", "a"])
    # for lr in learning_rates:
    #     print(f"\n=== 正在测试学习率: {lr} ===\n")
    #     for i in range(0, len(a_list)):
    #         # 加载数据集
    #         dataset, train_timesteps, test_timesteps = get_data(
    #             dataset_name=dataset_name,
    #             train_test_ratio=hparams["train_test_ratio"],
    #             device=device,
    #             a=a_list[i]
    #         )
    #         train_dataset = [dataset[k] for k in train_timesteps]
    #         INPUT_DIM = dataset[0].x.size(1)
    #
    #         # 初始化模型、优化器、调度器
    #         model = NodeImportanceModel(
    #             input_dim=INPUT_DIM,
    #             hidden_dim=hparams["hidden_dim"],
    #             output_dim=hparams["output_dim"]
    #         ).to(device=device)
    #
    #         optimizer = torch.optim.Adam(
    #             model.parameters(),
    #             lr=lr,  # 使用当前循环的学习率
    #             weight_decay=hparams["weight_decay"]
    #         )
    #
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer=optimizer,
    #             patience=hparams["scheduler_patience"],
    #             factor=hparams["scheduler_factor"],
    #             min_lr=hparams["scheduler_min_lr"]
    #         )
    #
    #         if i == 0:
    #             # 训练模型
    #             model = train(
    #                 model=model,
    #                 train_dataset=train_dataset,
    #                 optimizer=optimizer,
    #                 scheduler=scheduler,
    #                 hparams=hparams,
    #                 model_path=model_path
    #             )
    #
    #         # 推理
    #         test_probs = inference(
    #             model=model,
    #             dataset=dataset,
    #             test_timesteps=test_timesteps,
    #             model_path=model_path,
    #             device=device
    #         )
    #
    #         # 评估
    #         test_results = evaluate(
    #             test_probs=test_probs,
    #             test_timesteps=test_timesteps,
    #             dataset=dataset
    #         )
    #
    #         test_results['LearningRate'] = lr
    #         test_results['a'] = a_list[i]
    #         df_all = pd.concat([df_all, test_results], ignore_index=True)
    # save_path = f'学习率实验结果/{dataset_name}_lr_experiment.csv'
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # df_all.to_csv(save_path, index=False)
    # print(f"✅ 学习率实验结果已保存至 {save_path}")


def split_csv_to_txt_files(csv_file_path: str, output_dir: str, num_files: int = 10) -> None:
    """
    读取 CSV 文件，删除第一行，并将剩余内容平均划分为多个 .txt 文件。

    Args:
        csv_file_path (str): 输入的 CSV 文件路径。
        output_dir (str): 输出的 .txt 文件保存目录。
        num_files (int): 要划分的文件数量，默认为 10。
    """
    # 读取 CSV 文件，跳过第一行
    # df = pd.read_csv(csv_file_path, skiprows=1)
    #
    # # # 获取总行数
    # # total_rows = len(df)
    # # # 计算每个文件的行数
    # # rows_per_file = total_rows // num_files
    #
    # # 创建输出目录（如果不存在）
    # os.makedirs(output_dir, exist_ok=True)
    # output_file_path = os.path.join(output_dir, f"Gnu_1.txt")
    # df.to_csv(output_file_path, index=False, header=False, sep="\t")
    #
    # print(f"Saved {output_file_path} with {len(df)} rows.")

    # 分割数据并保存为 .txt 文件
    # for i in range(num_files):
    #     start_row = i * rows_per_file
    #     end_row = (i + 1) * rows_per_file if i < num_files - 1 else total_rows
    #
    #     # 获取当前文件的数据
    #     current_df = df.iloc[start_row:end_row]
    #
    #     # 保存为 .txt 文件
    #     output_file_path = os.path.join(output_dir, f"Gnutella_{i}.txt")
    #     current_df.to_csv(output_file_path, index=False, header=False, sep="\t")
    #
    #     print(f"Saved {output_file_path} with {len(current_df)} rows.")

def rename_files(directory, prefix="expansion"):
    """
    遍历指定目录下的所有文件，并按照指定的命名规则重命名文件。

    参数:
    directory (str): 要遍历的目录路径
    prefix (str): 文件名前缀，默认为 "birth_death_data"
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return

    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 按文件名排序（可选）
    files.sort()

    # 重命名文件
    for index, filename in enumerate(files):
        # 构造新的文件名
        new_filename = f"{prefix}_{index}.txt"
        new_file_path = os.path.join(directory, new_filename)

        # 获取原始文件的完整路径
        old_file_path = os.path.join(directory, filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_filename}")
def rename_files_in_directory(directory, old_prefix="merge_split_data", new_prefix="merge"):
    """
    遍历指定目录下的所有 .csv 文件，并将文件名中的 old_prefix 替换为 new_prefix。

    参数:
    directory (str): 要遍历的目录路径
    old_prefix (str): 原始文件名前缀，默认为 "birth_death_data"
    new_prefix (str): 新文件名前缀，默认为 "birth"
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return

    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # 重命名文件
    for filename in files:
        if filename.startswith(old_prefix):
            # 构造新的文件名
            new_filename = filename.replace(old_prefix, new_prefix)
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"文件 {filename} 已重命名为 {new_filename}")
if __name__ == "__main__":
    # 示例调用 数据预处理
    # csv_file_path = "datasets/raw_data/BA-8k.csv"  # 输入的 CSV 文件路径
    # output_dir = "datasets/raw_data/Gnu"  # 输出的 .txt 文件保存目录
    # split_csv_to_txt_files(csv_file_path, output_dir)
    # # 示例调用
    # split_txt_by_timestamp("datasets/raw_data/Dynamic_PPIN.txt", "datasets/raw_data/Dynamic_PPIN/")
    # 示例调用
    # source_directory = r"D:\lesson\Postgraduate\EXcode\DGCN3\MRCNN"
    # target_directory = r"D:\lesson\Postgraduate\EXcode\DGCN3\datasets\raw_data\Gnu"
    # rename_and_move_txt_files(source_directory, target_directory)

    # # 定义要遍历的目录
    # directory = 'datasets/raw_data/expansion'  # 替换为你的文件夹路径\
    # # 调用函数
    # rename_files(directory)

    # 定义要遍历的目录
    # directory = 'sir_results/merge'  # 替换为你的文件夹路径
    # # 调用函数
    # rename_files_in_directory(directory)

    # generate_ba_snapshots(6400, 8000, 3, "datasets/raw_data/BA")
    # generate_dynamic_ba_snapshots(
    #     num_snapshots=10,
    #     max_nodes=1000,
    #     m=3,
    #     output_dir="datasets/raw_data/BA"
    # )

    # 构造SIR
    # 定义数据集目录
    # datasets = {
    #     # "merge": "datasets/raw_data/merge",
    #     # "expansion":"datasets/raw_data/expansion",
    #     # "birth":"datasets/raw_data/birth"
    #     # "BA": "datasets/raw_data/BA"
    #     "tiage": "datasets/raw_data/tiage"
    # }
    #
    # # 定义输出目录
    # output_dir_base = "sir_results/"
    # os.makedirs(output_dir_base, exist_ok=True)
    # a_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    # # 运行计算
    # process_all_time_steps(datasets, output_dir_base, a_list= a_list)
    # print("🎉 所有数据集的 SIR 计算完成！")
    #
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(f"PyTorch Version: {torch.__version__}")  # 打印 PyTorch 版本号
    # print(f"CUDA Available: {torch.cuda.is_available()}")  # 检查是否有可用的 GPU 设备
    # if torch.cuda.is_available():
    #     print(f"CUDA Version: {torch.version.cuda}")  # 输出编译时使用的 CUDA 版本
    #     print(f"Device Name: {torch.cuda.get_device_name(0)}")  # 获取设备名称
    # else:
    #     print("No CUDA-enabled device found.")

    # 记录开始时间
    start_time = time.time()
    main()
    end_time = time.time()
    # 计算运行时间(秒)
    elapsed_time = end_time - start_time
    # 将运行时间转换为小时
    runtime_minutes = elapsed_time / 60
    # 打印运行时间（秒和小时）
    print(f"代码运行时间：{elapsed_time:.6f} 秒")
    print(f"代码运行时间：{runtime_minutes:.6f} 分钟")

    # # 定义边列表
    # edges = [
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (1, 3),
    #     (2, 3),
    #     (2, 5),
    #     (4, 5),
    #     (4, 6),
    #     (5, 6)
    # ]
    # # 创建图对象
    # G = nx.Graph()
    # G.add_edges_from(edges)
    # num_nodes =  7
    # # 可视化或打印信息（可选）
    # print("节点数:", G.number_of_nodes())
    # # print("边数:", G.number_of_edges())
    # # print("节点列表:", list(G.nodes))
    # # print("边列表:", list(G.edges))
    #
    # # 计算节点的结构特征
    # deg_features = compute_D_features(G, num_nodes)
    # h_features = compute_H_features(G, num_nodes)
    # partition = community_louvain.best_partition(G)
    # # 将 Louvain 算法的结果转换为与 greedy_modularity_communities 相同的格式
    # # 创建一个字典，键为社团编号，值为该社团的节点集合
    # community_dict = {}
    # for node, community_id in partition.items():
    #     if community_id not in community_dict:
    #         community_dict[community_id] = set()
    #     community_dict[community_id].add(node)
    # # 将社团字典转换为列表，每个元素是一个集合
    # communities = list(community_dict.values())
    # # _,  _, communities, _, _ = bayanpy.bayan(G, threshold=0.001,time_allowed=60, resolution=1)
    #
    # # communities = list(greedy_modularity_communities(G))
    #
    # # 计算社团内的结构特征
    # comm_deg_features = compute_community_D_features(G, communities, num_nodes)
    # comm_h_features = compute_community_H_features(G, communities, num_nodes)
    #
    # # 原始单位特征
    # # onehot_tensor = torch.eye(num_nodes)
    #
    # # 最终拼接特征
    # # final_x = torch.cat([onehot_tensor, deg_features, comm_deg_features,h_features,comm_h_features], dim=1)
    # final_x = torch.cat([deg_features, comm_deg_features, h_features, comm_h_features], dim=1)
    # print(final_x)



    # import os
    # import shutil
    # #
    # # # 定义映射关系
    # mapping = {
    #     "GrQ": "Gnu_3",
    #     "Hep": "Gnu_4",
    #     "Email": "Gnu_5",
    #     "Faa": "Gnu_6",
    #     "Facebook": "Gnu_7",
    #     "Figeys": "Gnu_8",
    #     "jazz": "Gnu_9",
    #     "LastFM": "Gnu_10",
    #     "NS": "Gnu_11",
    #     "powergrid": "Gnu_12",
    #     "Sex": "Gnu_13",
    #     "stelzl": "Gnu_14",
    #     "vidal": "Gnu_15"
    # }
    #
    # # 定义传播概率倍数列表
    # a_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    #
    # # 源目录和目标目录
    # source_dir = "MRCNN/SIR results"
    # target_dir = "sir_results/Gnu"
    #
    # # 创建目标目录
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)
    #
    # # 遍历源目录中的所有文件夹
    # for folder_name in os.listdir(source_dir):
    #     folder_path = os.path.join(source_dir, folder_name)
    #
    #     # 检查是否是目录
    #     if os.path.isdir(folder_path):
    #         # 获取映射后的名称
    #         if folder_name in mapping:
    #             mapped_name = mapping[folder_name]
    #         else:
    #             print(f"Warning: {folder_name} not found in mapping. Skipping.")
    #             continue
    #
    #         # 遍历文件夹中的所有文件
    #         for file_name in os.listdir(folder_path):
    #             file_path = os.path.join(folder_path, file_name)
    #
    #             # 检查是否是文件
    #             if os.path.isfile(file_path):
    #                 # 获取文件编号（从文件名中提取）
    #                 file_index = int(file_name.split('_')[1].split('.')[0])
    #
    #                 # 获取对应的传播概率倍数
    #                 a_value = a_list[file_index]
    #
    #                 # 构造新的文件名
    #                 new_file_name = f"{mapped_name}_a[{a_value}].csv"
    #                 new_file_path = os.path.join(target_dir, new_file_name)
    #
    #                 # 复制文件到目标目录并重命名
    #                 shutil.copyfile(file_path, new_file_path)
    #                 print(f"Renamed {file_name} to {new_file_name}")
    #
    # print("Renaming complete.")

