import pytest
import torch
import dgl
from src.models.dga_layers import DGAGNNLayer

@pytest.fixture
def sample_graph():
    """
    Creates a 3-node graph: 0 -> 1, 2 -> 1.
    Node 1 is the target node for testing aggregation.
    Node 0 is 'benign' (group 0).
    Node 2 is 'fraud' (group 1).
    """
    g = dgl.graph(([0, 2], [1, 1]))
    features = torch.tensor([[1.0], [0.0], [10.0]], dtype=torch.float32) # Node 1 feature is 0
    group_labels = torch.tensor([0, 0, 1], dtype=torch.long) # Node 0 -> group 0, Node 2 -> group 1
    return g, features, group_labels

def test_dga_gnn_layer(sample_graph):
    """
    Verify that the DGAGNNLayer applies different transformations for different neighbor groups.
    """
    g, features, group_labels = sample_graph
    in_feats = 1
    out_feats = 2
    num_groups = 2
    
    layer = DGAGNNLayer(in_feats, out_feats, num_groups)
    
    # Manually set the weights for predictability in the test
    # Group 0 (benign) transform: weight = [[1, 1]]
    # Group 1 (fraud) transform: weight = [[10, 10]]
    # Self transform: weight = [[0, 0]] (to isolate neighbor effect)
    with torch.no_grad():
        layer.neighbor_transforms[0].weight.copy_(torch.tensor([[1.], [1.]]))
        layer.neighbor_transforms[1].weight.copy_(torch.tensor([[10.], [10.]]))
        layer.self_transform.weight.copy_(torch.tensor([[0.], [0.]]))
        
    output = layer(g, features, group_labels)
    
    # Node 1 is the only node with neighbors, so we check its output
    target_node_output = output[1]
    
    # Expected calculation for Node 1:
    # From Node 0 (group 0, feature 1.0): 1.0 * [1, 1] = [1, 1]
    # From Node 2 (group 1, feature 10.0): 10.0 * [10, 10] = [100, 100]
    # Sum of neighbors: [1, 1] + [100, 100] = [101, 101]
    # Self contribution is 0 due to weights.
    expected_output = torch.tensor([101., 101.])
    
    assert torch.allclose(target_node_output, expected_output), f"Expected {expected_output}, but got {target_node_output}"
