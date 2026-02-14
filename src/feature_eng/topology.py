import networkx as nx
import pandas as pd

def compute_centrality(G, metric='degree'):
    """
    Compute centrality measures for nodes in the graph.
    Args:
        G (nx.Graph): NetworkX graph.
        metric (str): 'degree', 'pagerank'.
    Returns:
        pd.Series: Centrality scores indexed by node ID.
    """
    if metric == 'degree':
        centrality = nx.degree_centrality(G)
    elif metric == 'pagerank':
        centrality = nx.pagerank(G)
    else:
        raise ValueError(f"Unknown centrality metric: {metric}")
        
    return pd.Series(centrality, name=f'{metric}_centrality')

def extract_community_features(G):
    """
    Extract community detection features (e.g., Louvain modularity).
    Note: This is a placeholder for future implementation.
    """
    # Placeholder for community detection logic
    # communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    # community_map = {node: i for i, community in enumerate(communities) for node in community}
    # return pd.Series(community_map, name='community_id')
    pass
