import pytest
import os
from src.trainer import train

def test_feedback_loop_updates_groups():
    """
    Verify that the group_labels tensor is updated between epochs,
    proving the feedback mechanism is working.
    """
    # Config for a minimal training run on our synthetic test data
    config = {
        'data_path': os.path.join("tests", "data"),
        'epochs': 2,
        'lr': 0.5, # Increased learning rate for more significant updates
        'hidden_dim': 4
    }
    
    # Run training for a few epochs
    model, group_labels_history = train(config)
    
    # Check that we have history for initial state + each epoch
    assert len(group_labels_history) == config['epochs'] + 1
    
    # The initial group_labels should be all zeros
    initial_groups = group_labels_history[0]
    assert (initial_groups == 0).all()
    
    # The group_labels after the first epoch should be different from the initial ones.
    # Because the model initializes with random weights, it's highly unlikely
    # that all predictions will be 0.
    groups_after_epoch_1 = group_labels_history[1]
    assert not torch.equal(initial_groups, groups_after_epoch_1)
        
    # The group labels should also be different between epoch 1 and 2
    groups_after_epoch_2 = group_labels_history[2]
    assert not torch.equal(groups_after_epoch_1, groups_after_epoch_2)

# To run this test, we need torch available
import torch
