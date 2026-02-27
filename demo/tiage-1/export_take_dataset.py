#!/usr/bin/env python3
"""
tiage → TAKE 数据格式转换脚本

将 tiage_anno_nodes_all.csv 转换为 TAKE 模型所需的数据格式：
- tiage.answer
- tiage.query
- tiage.pool
- tiage.passage
- tiage.split
- ID_label.json
- node_id.json
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Tuple


def load_tiage_nodes(csv_path: str) -> pd.DataFrame:
    """加载 tiage 节点数据"""
    df = pd.read_csv(csv_path)
    # 确保必要的列存在
    required_cols = ['node_id', 'split', 'dialog_id', 'turn_id', 'text', 'shift_label']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")
    return df


def assign_slice_id(df: pd.DataFrame, num_slices: int = 10) -> pd.DataFrame:
    """分配时间片 ID"""
    lens = df.groupby('dialog_id')['turn_id'].max() + 1
    df = df.copy()
    df['dialog_len'] = df['dialog_id'].map(lens).astype(int)
    df['slice_id'] = (num_slices * df['turn_id'] / df['dialog_len']).astype(int)
    df['slice_id'] = df['slice_id'].clip(0, num_slices - 1)
    return df


def generate_query_id(row) -> str:
    """生成 query_id: dialog_id_turn_id"""
    return f"{row['dialog_id']}_{row['turn_id']}"


def export_take_dataset(
    df: pd.DataFrame,
    output_dir: str,
    train_slices: List[int] = None,
    test_slices: List[int] = None
) -> None:
    """导出 TAKE 格式数据集"""

    if train_slices is None:
        train_slices = list(range(7))  # slice 0-6
    if test_slices is None:
        test_slices = [7, 8, 9]  # slice 7-9

    os.makedirs(output_dir, exist_ok=True)

    # 分配 slice_id
    df = assign_slice_id(df)

    # 生成 query_id
    df['query_id'] = df.apply(generate_query_id, axis=1)

    # 按 dialog_id 和 turn_id 排序
    df = df.sort_values(['dialog_id', 'turn_id']).reset_index(drop=True)

    # 1. 生成 tiage.query (query_id\tquery_content)
    query_path = os.path.join(output_dir, 'tiage.query')
    with open(query_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['query_id']}\t{row['text']}\n")
    print(f"[OK] Generated {query_path}")

    # 2. 生成 tiage.passage (passage_id\tpassage_content)
    # 最小版本：每个节点的 passage 就是自己
    passage_path = os.path.join(output_dir, 'tiage.passage')
    with open(passage_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['query_id']}\t{row['text']}\n")
    print(f"[OK] Generated {passage_path}")

    # 3. 生成 tiage.pool (query_id Q0 passage_id rank score model_name)
    # 最小版本：每个 query 的 pool 只包含自己
    pool_path = os.path.join(output_dir, 'tiage.pool')
    with open(pool_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['query_id']} Q0 {row['query_id']} 1 1.0 tiage\n")
    print(f"[OK] Generated {pool_path}")

    # 4. 生成 tiage.answer (prev_ids\tcurrent_id\tpassage_ids\tresponse)
    # prev_ids: 同一对话内的历史 query_id（分号分隔）
    # passage_ids: 当前使用的知识（分号分隔）
    answer_path = os.path.join(output_dir, 'tiage.answer')
    with open(answer_path, 'w', encoding='utf-8') as f:
        for dialog_id, group in df.groupby('dialog_id'):
            group = group.sort_values('turn_id')
            prev_ids = []
            for _, row in group.iterrows():
                prev_str = ';'.join(prev_ids) if prev_ids else ''
                # 最小版本：passage_id 就是当前 query_id
                f.write(f"{prev_str}\t{row['query_id']}\t{row['query_id']}\t{row['text']}\n")
                prev_ids.append(row['query_id'])
    print(f"[OK] Generated {answer_path}")

    # 5. 生成 tiage.split (query_id\ttrain/test)
    split_path = os.path.join(output_dir, 'tiage.split')
    with open(split_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if row['slice_id'] in train_slices:
                split_label = 'train'
            elif row['slice_id'] in test_slices:
                split_label = 'test'
            else:
                continue  # 跳过 dev
            f.write(f"{row['query_id']}\t{split_label}\n")
    print(f"[OK] Generated {split_path}")

    # 6. 生成 ID_label.json (话题转移标签数组)
    # 按 query_id 顺序排列，-1 -> -1 (对话开头), 0 -> 0 (非转移), 1 -> 1 (转移)
    id_labels = []
    for _, row in df.iterrows():
        label = row['shift_label']
        if pd.isna(label):
            id_labels.append(-1)
        else:
            id_labels.append(int(label))

    id_label_path = os.path.join(output_dir, 'ID_label.json')
    with open(id_label_path, 'w', encoding='utf-8') as f:
        json.dump(id_labels, f)
    print(f"[OK] Generated {id_label_path}")

    # 7. 生成 node_id.json (query_id -> node_id 映射)
    node_id_map = {}
    for _, row in df.iterrows():
        node_id_map[row['query_id']] = int(row['node_id'])

    node_id_path = os.path.join(output_dir, 'node_id.json')
    with open(node_id_path, 'w', encoding='utf-8') as f:
        json.dump(node_id_map, f, indent=2)
    print(f"[OK] Generated {node_id_path}")

    # 打印统计信息
    train_count = len(df[df['slice_id'].isin(train_slices)])
    test_count = len(df[df['slice_id'].isin(test_slices)])
    print(f"\n[统计]")
    print(f"  总节点数: {len(df)}")
    print(f"  训练集: {train_count}")
    print(f"  测试集: {test_count}")
    print(f"  对话数: {df['dialog_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description='导出 tiage → TAKE 格式数据')
    parser.add_argument('--input', type=str,
                        default='outputs_nodes/tiage_anno_nodes_all.csv',
                        help='输入 CSV 文件路径')
    parser.add_argument('--out', type=str,
                        default='../../knowSelect/datasets/tiage',
                        help='输出目录')
    args = parser.parse_args()

    # 解析相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_dir = os.path.join(script_dir, args.out)

    print(f"[*] 输入文件: {input_path}")
    print(f"[*] 输出目录: {output_dir}")

    df = load_tiage_nodes(input_path)
    export_take_dataset(df, output_dir)

    print(f"\n[完成] TAKE 数据已导出到 {output_dir}")


if __name__ == '__main__':
    main()
