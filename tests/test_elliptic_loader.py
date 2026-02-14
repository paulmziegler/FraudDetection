import os

import pytest
import torch

from src.data_loader import load_data


@pytest.mark.elliptic
def test_elliptic_data_loading():
    """
    Tests the loading and preprocessing of the real Elliptic dataset.
    This test will only run if the raw data is available.
    """
    data_path = "data"
    raw_path = os.path.join(data_path, "raw")

    # Skip test if data is not present
    if not all(
        os.path.exists(os.path.join(raw_path, f))
        for f in [
            "elliptic_txs_classes.csv",
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_features.csv",
        ]
    ):
        pytest.skip("Elliptic dataset not found in data/raw")

    g, features, labels, train_mask, val_mask, test_mask = load_data(
        data_path, dataset_name="elliptic", train_split_step=35, val_split_step=42
    )

    # Basic sanity checks
    assert g.number_of_nodes() > 0
    assert features.shape[0] == g.number_of_nodes()
    assert labels.shape[0] == g.number_of_nodes()

    # Check label mapping (should only be 0s and 1s)
    assert torch.all((labels == 0) | (labels == 1))

    # Check mask exclusivity
    overlap = (
        (train_mask & val_mask).any()
        or (val_mask & test_mask).any()
        or (train_mask & test_mask).any()
    )
    assert not overlap, "Masks should be mutually exclusive"

    # Check that some nodes exist in each set
    assert train_mask.sum() > 0
    assert val_mask.sum() > 0
    assert test_mask.sum() > 0
