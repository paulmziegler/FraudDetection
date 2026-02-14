import pytest
import pandas as pd
import numpy as np
from src.feature_eng.binning import find_optimal_bins, encode_bins

@pytest.fixture
def sample_data():
    """
    Creates a simple dataset where a clear split point exists for fraud.
    - feature 'amount': Fraud occurs for amounts >= 100
    - feature 'age': No clear fraud signal, should result in fewer bins.
    """
    X = pd.DataFrame({
        'amount': [10, 20, 50, 100, 150, 200],
        'age':    [25, 30, 22, 40, 50, 35]
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return X, y

def test_find_optimal_bins(sample_data):
    """
    Verify the decision tree finds the correct split point for 'amount'.
    """
    X, y = sample_data
    bins = find_optimal_bins(X, y, max_bins=2)
    
    assert 'amount' in bins
    # The optimal split should be between 50 and 100. The tree finds a value like 75.
    assert len(bins['amount']) == 1
    assert 50 < bins['amount'][0] < 100

def test_encode_bins(sample_data):
    """
    Verify that data is correctly encoded into bins based on thresholds.
    """
    X, y = sample_data
    # Manually define bins for a predictable test
    manual_bins = {'amount': [75.0], 'age': [37.5]}
    
    X_binned = encode_bins(X, manual_bins)
    
    assert 'amount_binned' in X_binned.columns
    # Amounts < 75 should be in bin 0, >= 75 in bin 1
    expected_amount_bins = np.array([0, 0, 0, 1, 1, 1])
    assert np.array_equal(X_binned['amount_binned'].values, expected_amount_bins)
    
    # Ages < 37.5 in bin 0, >= 37.5 in bin 1
    expected_age_bins = np.array([0, 0, 0, 1, 1, 0])
    assert np.array_equal(X_binned['age_binned'].values, expected_age_bins)
