# DGA-GNN Model (`src/models/dga_gnn.py`)

## Core Functionality
The `dga_gnn.py` file defines the **DGA-GNN (Dynamic Grouping Aggregation Graph Neural Network)** architecture. It is a PyTorch `nn.Module`.

This is the high-level container that:
1.  Initializes the specific layers (`DGAGNNLayer`).
2.  Defines the flow of data (`forward` pass).
3.  Orchestrates the multi-layer message passing and final classification.

## ELI5 Description
This is the brain of the operation. It takes the map of connections (graph) and the "groups" (what we currently think the nodes are, e.g., "suspicious" or "safe") and passes them through two special filters (layers).

After filtering and refining the information twice, it passes the result to a final decision-maker (classifier) that shouts out "Fraud!" or "Safe!" for every single dot.

## Classes

### `DGAGNN`
-   **Inherits**: `torch.nn.Module`
-   **Init**:
    -   `in_feats`: Number of input features per node.
    -   `h_feats`: Hidden dimension size (width of the layers).
    -   `num_classes`: Output size (usually 2 for Binary Classification).
    -   `num_groups`: Number of dynamic groups (e.g., 2).
-   **Methods**:
    -   `forward(g, h, group_labels)`:
        -   **Inputs**: The graph `g`, node features `h`, and the dynamic `group_labels` (predictions from the *previous* epoch).
        -   **Logic**: `Input -> Layer1 -> ReLU -> Layer2 -> ReLU -> Classifier -> Output`.
        -   **Key Detail**: Note that `group_labels` are passed into every layer, allowing the aggregation to remain dynamic throughout the network.
