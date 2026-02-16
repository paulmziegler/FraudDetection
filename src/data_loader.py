import os

import dgl
import pandas as pd
import torch


def load_data(data_path, dataset_name="test", train_split_step=35, val_split_step=42):
    """
    Loads data and constructs a DGL graph. Supports the 'elliptic' dataset and a 'test' dataset.

    Args:
        data_path (str): Path to the directory containing the data.
        dataset_name (str): The name of the dataset to load ('elliptic' or 'test').
        train_split_step (int): For Elliptic, the time step to end training.
        val_split_step (int): For Elliptic, the time step to end validation.

    Returns:
        dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    """
    if dataset_name == "elliptic":
        return _load_elliptic_data(data_path, train_split_step, val_split_step)
    elif dataset_name == "test":
        return _load_test_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_elliptic_data(data_path, train_split_step, val_split_step):
    """Loads the Elliptic dataset."""
    raw_path = os.path.join(data_path, "raw")
    classes_df = pd.read_csv(os.path.join(raw_path, "elliptic_txs_classes.csv"))
    edgelist_df = pd.read_csv(os.path.join(raw_path, "elliptic_txs_edgelist.csv"))
    features_df = pd.read_csv(
        os.path.join(raw_path, "elliptic_txs_features.csv"), header=None
    )

    # Pre-process features and labels
    features_df.columns = (
        ["txId", "Time step"]
        + [f"local_feat_{i}" for i in range(93)]
        + [f"agg_feat_{i}" for i in range(72)]
    )

    # Merge classes and features
    nodes_df = pd.merge(features_df, classes_df, on="txId")

    # Filter out unknown classes and map to binary
    nodes_df = nodes_df[nodes_df["class"] != "unknown"].copy()
    nodes_df["class"] = nodes_df["class"].map({"1": 1, "2": 0})  # Illicit=1, Licit=0

    # Node IDs must be contiguous from 0 to N-1 for DGL. We create a map.
    node_ids = nodes_df["txId"].unique()
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
    nodes_df["nodeId"] = nodes_df["txId"].map(node_id_map)
    edgelist_df["src"] = edgelist_df["txId1"].map(node_id_map)
    edgelist_df["dst"] = edgelist_df["txId2"].map(node_id_map)

    # Drop rows with NaN in src/dst (edges pointing to unknown nodes)
    edgelist_df = edgelist_df.dropna()

    # Create graph
    g = dgl.graph((edgelist_df["src"].values, edgelist_df["dst"].values))

    # Align nodes_df with the graph's node IDs
    nodes_df = nodes_df.sort_values("nodeId").reset_index(drop=True)

    # Extract features, labels, and timestamps
    feature_cols = [c for c in nodes_df.columns if "feat" in c]
    features = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)
    labels = torch.tensor(nodes_df["class"].values, dtype=torch.long)
    timestamps = torch.tensor(nodes_df["Time step"].values, dtype=torch.long)

    # Temporal Split
    train_mask = (timestamps < train_split_step)
    val_mask = (timestamps >= train_split_step) & (timestamps < val_split_step)
    test_mask = (timestamps >= val_split_step)

    g.ndata["feat"] = features
    g.ndata["label"] = labels
    return g, features, labels, train_mask, val_mask, test_mask


def _load_test_data(data_path):
    """Loads the synthetic test data."""
    nodes_df = pd.read_csv(os.path.join(data_path, "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(data_path, "edges.csv"))
    nodes_df = nodes_df.sort_values("node_id").reset_index(drop=True)
    g = dgl.graph((edges_df["src"].values, edges_df["dst"].values))
    feature_cols = [
        c for c in nodes_df.columns if c not in ["node_id", "timestamp", "label"]
    ]
    features = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)
    labels = torch.tensor(nodes_df["label"].values, dtype=torch.long)

    # Dummy masks for test data
    num_nodes = g.number_of_nodes()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes) : int(0.8 * num_nodes)] = True
    test_mask[int(0.8 * num_nodes) :] = True

    return g, features, labels, train_mask, val_mask, test_mask
