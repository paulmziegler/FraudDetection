# Fraud Detection GNN (DGA-GNN)

This project implements a Dynamic Grouping Aggregation GNN (DGA-GNN) for fraud detection, inspired by the AAAI 2024 framework. It leverages non-additive attribute handling and feedback dynamic grouping to improve detection accuracy.

## Project Structure

```
fraud_gnn_project/
├── configs/                # Hydra or YAML configs
├── data/                   # Raw and processed transaction data
├── src/
│   ├── data_loader.py      # Ingestion, temporal splitting, and DGL graph construction
│   ├── feature_eng/
│   │   ├── binning.py      # Decision Tree Binning (Toad/Scikit-learn)
│   │   └── topology.py     # NetworkX centrality/community features
│   ├── models/
│   │   ├── dga_layers.py   # Bin-aware aggregation layers
│   │   └── dga_gnn.py      # Main architecture with Feedback Grouping
│   └── trainer.py          # Training loop with recursive group feedback
├── notebooks/              # Exploratory Data Analysis (EDA)
├── run.py                  # Main entry point
└── requirements.txt        # Dependencies
```

## Key Features

1.  **Decision Tree Binning Encoding**: Handles continuous features like "Account Age" by finding optimal thresholds using Decision Trees (or Toad) and encoding them as bin vectors.
2.  **Feedback Dynamic Grouping**: A feedback loop where node predictions from one epoch are used to dynamically group neighbors (e.g., "Potential Fraud" vs "Potential Benign") for the next epoch's aggregation.
3.  **Strict Temporal Validation**: Data is split by time (e.g., train on Jan-Jun, test on July) to simulate real-world conditions and avoid data leakage.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Experiments**
    ```bash
    python run.py --config configs/default.yaml
    ```

3.  **Docker**
    ```bash
    docker-compose up --build
    ```

## Development

-   **Linting**: `python manage.py lint` (or `ruff check src`)
-   **Testing**: `python manage.py test` (or `pytest tests`)

## Documentation
Documentation is located in `docs/` and is designed to be compatible with Obsidian.
