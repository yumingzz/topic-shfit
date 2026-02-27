import os
import json
import pandas as pd
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
# import faiss
import umap

def load_anno_json(json_path: str, split: str) -> pd.DataFrame:
    """
    读取 TIAGE anno_{split}.json，输出 turn-level 节点表：
    columns = [split, dialog_id, turn_id, text, shift_label]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = []
    # obj: dict, key = dialog_id(str), value = list of [utterance, label]
    # 例如: {"1": [["utt","-1"], ["utt2","0"], ...], "2": ...}
    for dialog_id_str, turns in obj.items():
        try:
            dialog_id = int(dialog_id_str)
        except ValueError:
            dialog_id = dialog_id_str  # 保底

        if not isinstance(turns, list):
            continue

        for turn_id, turn in enumerate(turns):
            if not isinstance(turn, list) or len(turn) < 1:
                continue
            text = turn[0]
            label = turn[1] if len(turn) > 1 else None

            # 尝试把 label 转成 int（失败就保留原字符串）
            shift_label = None
            if label is not None:
                try:
                    shift_label = int(label)
                except Exception:
                    shift_label = label

            rows.append(
                {
                    "split": split,
                    "dialog_id": dialog_id,
                    "turn_id": turn_id,
                    "text": text,
                    "shift_label": shift_label,
                }
            )

    return pd.DataFrame(rows)


def build_nodes_table(out_dir: str) -> pd.DataFrame:
    """
    out_dir: 输出目录
    """
    anno_dir = os.path.join("data", "personachat", "anno")

    parts = []
    for split in ["train", "dev", "test"]:
        json_path = os.path.join(anno_dir, split, f"anno_{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到: {json_path}")

        df = load_anno_json(json_path, split=split)
        parts.append(df)

        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"tiage_anno_nodes_{split}.csv"), index=False, encoding="utf-8")
        print(f"[OK] {split}: {len(df)} nodes -> tiage_anno_nodes_{split}.csv")

    nodes_df = pd.concat(parts, ignore_index=True)

    # 给每个节点一个全局 node_id（保证 0..N-1 连续，后面 kNN 连边会非常省事）
    nodes_df.insert(0, "node_id", range(len(nodes_df)))

    nodes_df.to_csv(os.path.join(out_dir, "tiage_anno_nodes_all.csv"), index=False, encoding="utf-8")
    print(f"[OK] all: {len(nodes_df)} nodes -> tiage_anno_nodes_all.csv")
    return nodes_df


# -----------------------------
# 1) 按 turn 比例切片（每个 dialog 单独映射到 0..num_slices-1）
# -----------------------------
def assign_slice_by_turn_ratio(nodes_df: pd.DataFrame, num_slices: int = 10) -> pd.DataFrame:
    # turn_id 默认从 0 开始；dialog_len = max(turn_id)+1
    lens = nodes_df.groupby("dialog_id")["turn_id"].max() + 1
    out = nodes_df.copy()
    out["dialog_len"] = out["dialog_id"].map(lens).astype(int)
    out["slice_id"] = (num_slices * out["turn_id"] / out["dialog_len"]).astype(int)
    out["slice_id"] = out["slice_id"].clip(0, num_slices - 1)
    return out


# -----------------------------
# 2) 句向量（推荐直接用句向量模型）
# -----------------------------
def embed_texts(texts, model_name="BAAI/bge-base-en-v1.5", batch_size=64):
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # cosine sim = dot
        show_progress_bar=True
    )
    return emb.astype("float32")


# -----------------------------
# 3) kNN 建边（FAISS）
# 返回的是 embedding 行号的边: (i_row, j_row)
# -----------------------------
# def build_knn_edges_from_emb(emb: np.ndarray, k: int = 20):
#     n, d = emb.shape
#     index = faiss.IndexFlatIP(d)  # normalized => cosine similarity
#     index.add(emb)
#
#     sims, nbrs = index.search(emb, k + 1)  # include self at rank 0
#     edges = []
#     for i in range(n):
#         for rank in range(1, k + 1):
#             j = int(nbrs[i, rank])
#             sim = float(sims[i, rank])
#             dist = float(1.0 - sim)
#             edges.append((i, j, {"sim": sim, "dist": dist, "type": "knn"}))
#     return edges
def build_knn_edges_from_emb(emb: np.ndarray, k: int = 20):
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="auto"
    ).fit(emb)

    distances, indices = nbrs.kneighbors(emb)

    edges = []
    for i in range(emb.shape[0]):
        for rank in range(1, k + 1):  # 跳过自己
            j = int(indices[i, rank])
            dist = float(distances[i, rank])
            sim = float(1.0 - dist)
            edges.append((i, j, {"sim": sim, "dist": dist, "type": "knn"}))
    return edges


# -----------------------------
# 4) MST 补边：保证全连通
# 注意：MST 在“距离 dist”上做最小生成树
# -----------------------------
def add_mst_edges(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # 先把现有边（kNN）作为候选边加入 H，并带权重
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, weight=float(data.get("dist", 1.0)))

    # 这里如果 H 本身不连通，minimum_spanning_tree 只会对每个连通分量生成树
    # 但因为我们后面会把 MST 边并入 G，所以建议先确保 H 是连通的：
    # ——最稳的做法是：在所有点上补一个“完全图”太贵
    # ——实践上：k 选得合理(>=10~20) 通常 H 已经很大概率连通
    # 若你发现不连通，直接把 k 调大即可
    mst = nx.minimum_spanning_tree(H, weight="weight")

    for u, v in mst.edges():
        if not G.has_edge(u, v):
            dist = float(H[u][v]["weight"])
            sim = float(1.0 - dist)
            G.add_edge(u, v, sim=sim, dist=dist, type="mst")
    return G


# -----------------------------
# 5) UMAP 坐标
# -----------------------------
def project_umap(emb: np.ndarray, n_neighbors=15, min_dist=0.05, random_state=42):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state
    )
    xy = reducer.fit_transform(emb)
    return xy


# -----------------------------
# 6) 主流程：支持从 anno 节点表构图
# - 可选过滤 split（train/dev/test/all）
# - 自动把“原始 node_id”映射为连续 row_id，避免 kNN 连错
# - 节点属性保留 split / shift_label（如果存在）
# -----------------------------
def build_spatial_network(
    nodes_df: pd.DataFrame,
    model_name: str = "BAAI/bge-base-en-v1.5",
    k: int = 20,
    num_slices: int = 10,
    split: str = "all",              # "train"/"dev"/"test"/"all"
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.05,
    random_state: int = 42,
):
    df = nodes_df.copy()

    # 0) 过滤 split（如果你传入的是 all）
    if split != "all":
        if "split" not in df.columns:
            raise ValueError("你指定了 split 过滤，但 nodes_df 没有 split 列。")
        df = df[df["split"] == split].reset_index(drop=True)

    # 1) 基本列检查
    required = {"node_id", "dialog_id", "turn_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"nodes_df 缺少必要列: {missing}")

    # 2) 切片
    df = assign_slice_by_turn_ratio(df, num_slices=num_slices)

    # 3) 为 kNN 建边做“行号映射”
    # row_id: 0..N-1 (embedding index)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=int)

    # 4) embedding
    texts = df["text"].astype(str).tolist()
    emb = embed_texts(texts, model_name=model_name)

    # 5) 坐标
    xy = project_umap(emb, n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=random_state)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    # 6) 建图：用 row_id 作为图节点 id（最安全），同时把原 node_id 存属性里
    G = nx.Graph()
    has_shift = "shift_label" in df.columns
    has_split = "split" in df.columns

    for _, row in df.iterrows():
        rid = int(row["row_id"])
        attrs = {
            "orig_node_id": int(row["node_id"]),
            "dialog_id": int(row["dialog_id"]),
            "turn_id": int(row["turn_id"]),
            "slice_id": int(row["slice_id"]),
            "text": row["text"],
            "x": float(row["x"]),
            "y": float(row["y"]),
        }
        if has_split:
            attrs["split"] = row["split"]
        if has_shift:
            attrs["shift_label"] = row["shift_label"]
        G.add_node(rid, **attrs)

    # 7) kNN 边（注意：u,v 是 row_id）
    edges = build_knn_edges_from_emb(emb, k=k)
    for u, v, data in edges:
        if u != v:
            G.add_edge(int(u), int(v), **data)

    # 8) MST 补边（保证连通尽量）
    G = add_mst_edges(G)

    # 9) 切片子图（诱导子图）
    slice_graphs = {}
    for s in range(num_slices):
        nodes_s = [n for n, d in G.nodes(data=True) if d["slice_id"] == s]
        slice_graphs[s] = G.subgraph(nodes_s).copy()

    return G, slice_graphs, df

def save_slice_graphs_as_txt(
    slice_graphs: dict,
    out_dir: str = "./slice_txt",
    prefix: str = "tiage"
):
    """
    将 slice_graphs 保存为 txt 边列表
    文件名: tiage_1.txt, tiage_2.txt, ..., tiage_10.txt
    每行: u v
    """
    os.makedirs(out_dir, exist_ok=True)

    for s, Gs in slice_graphs.items():
        # 文件编号从 0 开始
        fname = f"{prefix}_{s}.txt"
        fpath = os.path.join(out_dir, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            for u, v in Gs.edges():
                f.write(f"{u} {v}\n")

        print(f"[OK] slice {s} -> {fname} | nodes={Gs.number_of_nodes()} edges={Gs.number_of_edges()}")



if __name__ == "__main__":
    OUT_DIR = "./outputs_nodes"
    nodes_df = build_nodes_table(OUT_DIR)
    # 简单 sanity check
    print(nodes_df.head())
    print(nodes_df.groupby("split")["dialog_id"].nunique())

    nodes_df = pd.read_csv("outputs_nodes/tiage_anno_nodes_all.csv")

    G_all, slice_graphs, df_used = build_spatial_network(
        nodes_df,
        model_name="BAAI/bge-base-en-v1.5",
        k=20,
        split="all",  # 或 "train"
        num_slices=10
    )

    print(G_all.number_of_nodes(), G_all.number_of_edges())
    print({s: slice_graphs[s].number_of_nodes() for s in slice_graphs})
    save_slice_graphs_as_txt(
        slice_graphs,
        out_dir="./tiage_slices_txt",
        prefix="tiage"
    )

