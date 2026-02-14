# Binning (`src/feature_eng/binning.py`)

## Core Functionality
The `binning.py` module implements **discretization** for continuous features. In fraud detection, the raw value of a transaction (e.g., $50.00 vs $50.01) often matters less than the *range* it falls into (e.g., "Small", "Large", "Huge").

This module uses a **Decision Tree** approach to find the optimal cut-points for these ranges. By training a shallow tree to predict fraud using only one feature, the tree naturally splits the feature at points that maximize information gain (separation between fraud and legitimate).

## ELI5 Description
Imagine you are sorting Lego bricks. Instead of measuring every single brick's width with a micrometer (which is too much detail), you want to put them into buckets: "Small", "Medium", and "Large".

This file uses a smart robot (the Decision Tree) to look at the bricks and decide *exactly* where the cutoff size between "Small" and "Medium" should be so that the "Small" bucket has mostly red bricks (fraud) and the "Medium" bucket has mostly blue bricks (safe).

## Functions

### `find_optimal_bins`
-   **Signature**: `find_optimal_bins(X_train, y_train, max_bins=5)`
-   **Description**:
    -   Iterates through every column in `X_train`.
    -   Trains a `DecisionTreeClassifier` with `max_leaf_nodes=max_bins` on that single feature against the target `y_train`.
    -   Extracts the `threshold` values from the tree's internal nodes.
    -   These thresholds represent the optimal split points.
-   **Returns**: A dictionary `{feature_name: [threshold1, threshold2, ...]}`.

### `encode_bins`
-   **Signature**: `encode_bins(X, bins)`
-   **Description**:
    -   Takes the raw dataframe `X` and the dictionary of `bins` (from the function above).
    -   Uses `pandas.cut` to digitize the continuous values into integer bin IDs (0, 1, 2...).
-   **Returns**: A new DataFrame where values are integer categories.
