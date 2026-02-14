import pytest
import networkx as nx
import pandas as pd
from src.feature_eng.topology import compute_centrality

@pytest.fixture
def star_graph():
    """Creates a 5-node star graph with node 0 as the center."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    return G

def test_degree_centrality(star_graph):
    """
    On a star graph, the central node (0) must have the highest degree centrality.
    """
    centrality = compute_centrality(star_graph, metric='degree')
    
    assert isinstance(centrality, pd.Series)
    assert centrality.idxmax() == 0 # Node 0 should be the max
    assert centrality[0] > centrality[1]
    
def test_pagerank_centrality(star_graph):
    """
    On a star graph, the central node (0) must have the highest PageRank.
    """
    centrality = compute_centrality(star_graph, metric='pagerank')
    
    assert isinstance(centrality, pd.Series)
    assert centrality.idxmax() == 0 # Node 0 should be the max
    assert centrality[0] > centrality[1]

def test_invalid_metric(star_graph):
    """Ensure an unknown metric raises a ValueError."""
    with pytest.raises(ValueError):
        compute_centrality(star_graph, metric='invalid_metric')
