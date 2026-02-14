import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def find_optimal_bins(X_train, y_train, max_bins=5):
    """
    Finds optimal binning thresholds for each feature using a Decision Tree.
    
    Args:
        X_train (pd.DataFrame): DataFrame of continuous features.
        y_train (pd.Series): Series of binary labels (0 or 1).
        max_bins (int): The maximum number of bins to create (corresponds to max_leaf_nodes).
        
    Returns:
        dict: A dictionary mapping feature names to a sorted list of thresholds.
    """
    bins = {}
    for feature in X_train.columns:
        # Reshape feature for DecisionTree
        feature_data = X_train[[feature]]
        
        # Train a shallow decision tree to find splits
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, criterion='entropy')
        tree.fit(feature_data, y_train)
        
        # Extract thresholds where splits occur
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]  # -2 indicates a leaf node
        thresholds = np.unique(thresholds)
        thresholds.sort()
        bins[feature] = thresholds
            
    return bins

def encode_bins(X, bins):
    """
    Encodes continuous data into discrete bins based on provided thresholds.
    
    Args:
        X (pd.DataFrame): DataFrame of continuous features.
        bins (dict): Dictionary mapping feature names to thresholds.
        
    Returns:
        pd.DataFrame: A new DataFrame with features encoded as bin labels (integers).
    """
    X_binned = pd.DataFrame(index=X.index)
    for feature, thresholds in bins.items():
        # Use pandas.cut to apply the binning
        bin_edges = [-np.inf] + list(thresholds) + [np.inf]
        X_binned[f"{feature}_binned"] = pd.cut(
            X[feature], 
            bins=bin_edges, 
            labels=False, # Return integer labels
            right=False # Intervals are [left, right)
        )
    
    return X_binned
