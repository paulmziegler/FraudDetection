# Data Loader (`src/data_loader.py`)

## Core Functionality
The `data_loader.py` module is responsible for ingesting raw transaction data and converting it into a structured **Deep Graph Library (DGL)** graph. It handles:
1.  **Data Ingestion**: Reading CSVs for nodes (features, classes) and edges.
2.  **Graph Construction**: Mapping raw IDs to contiguous DGL node IDs and creating the graph structure.
3.  **Feature Assignment**: Attaching features and labels to graph nodes.
4.  **Temporal Splitting**: Crucially, it creates boolean masks (`train_mask`, `val_mask`, `test_mask`) based on time steps to strictly enforce temporal validity (training on past, testing on future).

## ELI5 Description
Imagine a giant coloring book of dots (accounts) and lines (money transfers). This file is the librarian. It opens the book, assigns every dot a specific number, and checks the "timestamp" on every page. It then marks the early pages with a green sticker ("Train" - allowed to study), the middle pages with a yellow sticker ("Validate" - for practice tests), and the final pages with a red sticker ("Test" - strictly for the final exam).

## Functions

### `load_data`
-   **Signature**: `load_data(data_path, dataset_name="test", train_split_step=35, val_split_step=42)`
-   **Description**: The main public interface. It acts as a factory, calling the appropriate specific loader (`_load_elliptic_data` or `_load_test_data`) based on the `dataset_name` argument.
-   **Returns**: A tuple containing the graph `g`, features, labels, and the three masks (train, val, test).

### `_load_elliptic_data`
-   **Signature**: `_load_elliptic_data(data_path, train_split_step, val_split_step)`
-   **Description**: specifically handles the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
    -   Merges classes and features CSVs.
    -   Maps "Illicit" to 1 and "Licit" to 0.
    -   Removes "Unknown" class nodes (standard practice for this dataset).
    -   Constructs the graph and creates time-based masks.

### `_load_test_data`
-   **Signature**: `_load_test_data(data_path)`
-   **Description**: Loads a synthetic/dummy dataset from `nodes.csv` and `edges.csv`. Useful for unit testing or quick debugging without the full Elliptic dataset. Uses a simple 60/20/20 split.
