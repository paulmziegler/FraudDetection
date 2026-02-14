import pytest
import torch
import dgl
from src.models.dga_gnn import DGAGNN

@pytest.fixture
def sample_graph_and_data():
    """
    Creates a simple graph with random features for shape testing.
    """
    g = dgl.graph(([0, 1, 2], [1, 2, 3])) # A simple 4-node graph
    features = torch.randn(4, 10) # 4 nodes, 10 features each
    group_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return g, features, group_labels

def test_dga_gnn_forward_pass(sample_graph_and_data):
    """
    Verify that the full DGAGNN model can execute a forward pass
    and that the output tensor has the correct shape.
    """
    g, features, group_labels = sample_graph_and_data
    
    in_feats = 10
    h_feats = 16
    num_classes = 2 # e.g., 'fraud' vs 'benign'
    
    model = DGAGNN(in_feats, h_feats, num_classes)
    
    # Perform a forward pass
    output = model(g, features, group_labels)
    
    # Check the output shape
    assert output.shape[0] == g.number_of_nodes(), "Output should have a row for each node"
    assert output.shape[1] == num_classes, "Output should have a column for each class"
