import pytest
import os
import torch
from src.data_loader import load_data

def test_temporal_split():
    """
    Verify that nodes are correctly partitioned by time without leakage.
    Train: < 2024-01-01 (Nodes 0, 1)
    Val: 2024-01-01 to 2024-03-01 (Nodes 2, 3)
    Test: >= 2024-03-01 (Node 4)
    """
    data_path = os.path.join("tests", "data")
    train_date = "2024-01-01"
    val_date = "2024-03-01"
    
    g, features, labels, train_mask, val_mask, test_mask = load_data(
        data_path, 
        train_end_date=train_date, 
        val_end_date=val_date
    )
    
    # Check total counts
    assert g.number_of_nodes() == 5
    
    # Check masks (indices)
    assert train_mask[0] == True
    assert train_mask[1] == True
    assert train_mask[2] == False # Should be in Val
    
    assert val_mask[2] == True
    assert val_mask[3] == True
    
    assert test_mask[4] == True
    
    # Ensure no overlap
    overlap = (train_mask & val_mask).any() or (val_mask & test_mask).any() or (train_mask & test_mask).any()
    assert overlap == False, "Masks should be mutually exclusive"

def test_feature_loading():
    """Verify features and labels are loaded correctly."""
    data_path = os.path.join("tests", "data")
    _, features, labels, _, _, _ = load_data(data_path)
    
    assert features.shape == (5, 1)
    assert labels.shape == (5,)
    assert labels[2] == 1 # Node 2 is fraud in our dummy data
