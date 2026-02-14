# Topology (`src/feature_eng/topology.py`)

## Core Functionality
The `topology.py` module extracts **graph-based features** from the network structure. Unlike transaction features (amount, time), these features describe a node's *position* and *relationships* within the graph.

Currently, it supports extracting centrality metrics, which help identify key hubs or influential nodes in the laundering network.

## ELI5 Description
This module is like a popularity contest judge. It looks at the tangled web of dots (people) and lines (transactions) and calculates:
1.  **Degree Centrality**: Who has the most friends? (Who sends/receives the most transfers?)
2.  **PageRank**: Who is the most influential VIP? (Who is connected to other important people?)
3.  **Community Detection**: Who hangs out in the same clique? (Used to find groups of fraudsters working together).

## Functions

### `compute_centrality`
-   **Signature**: `compute_centrality(G, metric='degree')`
-   **Description**:
    -   Accepts a NetworkX graph `G`.
    -   Calculates the requested `metric` ('degree' or 'pagerank').
    -   Wraps the result in a Pandas Series for easy merging with other features.
-   **Returns**: A Pandas Series where the index is the node ID and the value is the score.

### `extract_community_features`
-   **Signature**: `extract_community_features(G)`
-   **Description**:
    -   *Placeholder*. Intended to implement algorithms like Louvain Modularity or Label Propagation to assign a "Community ID" to each node.
    -   Currently returns `None` or `pass`.
