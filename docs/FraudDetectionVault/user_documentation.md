# User Documentation: Fraud Detection Project

## Introduction
This project simulates a **Dynamic Grouping Aggregation GNN (DGA-GNN)** for fraud detection. It leverages machine learning, graph theory, and advanced feature engineering to identify fraudulent transactions.

## Installation

1.  **Clone/Download the Repository**
    -   Ensure you have access to the `D:\Data\FraudDetection` directory.

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU Support (Optional)**
    To enable GPU training (NVIDIA CUDA), install the specific versions of PyTorch and DGL:
    ```powershell
    # Windows PowerShell
    .\install_gpu_deps.ps1
    ```
    Or manually:
    ```bash
    pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
    ```

## Usage

### Running Experiments
The main entry point for running experiments is `run.py`.
```bash
python run.py --config configs/default.yaml
```

**Parameters**:
-   `--config`: Path to the YAML configuration file (default: `configs/default.yaml`).
-   `--epochs`: Number of training epochs (default: `100`).
-   `--batch_size`: Batch size for training (default: `32`).
-   `--data_path`: Path to the directory containing processed data (default: `data/processed`).

### Running Tests
Execute unit tests using the management script:
```bash
python manage.py test
# Or directly with pytest
pytest tests/
```

### Running Linting
Check code quality:
```bash
python manage.py lint
# Or directly with ruff
ruff check src/
```

### Hyperparameter Tuning
To perform a grid search for optimal hyperparameters:
```bash
python tune.py --config configs/elliptic.yaml
```
This script iterates through combinations of learning rates, hidden dimensions, and epochs, printing the validation accuracy for each.

## Docker (GPU Supported)

The project includes a Docker configuration optimized for NVIDIA GPUs (CUDA 12.1).

**Recommended Method (Windows):**
Use the helper script to build and run the container:
```powershell
.\build_and_run_docker.ps1
```

**Manual Method:**
1.  **Build:**
    ```bash
    docker-compose build
    ```
2.  **Run:**
    ```bash
    docker-compose up
    ```

**Note:** Ensure you have the NVIDIA Container Toolkit installed and Docker configured to use the `nvidia` runtime.

Access the container shell:
```bash
docker-compose run app /bin/bash
```

## Configuration (`configs/`)
Configuration files are stored in the `configs/` directory. Create new YAML files for different datasets (e.g., Elliptic, Amazon) or hyperparameter settings.

Example `default.yaml`:
```yaml
model:
  hidden_dim: 64
  num_layers: 2
training:
  device: "cuda" # or "cpu"
data:
  train_split: "2024-01-01"
  val_split: "2024-03-01"
```

## Troubleshooting
-   **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed.
-   **Data Not Found**: Verify that `data/processed` contains the necessary CSV/Parquet files.
-   **Docker Issues**: Ensure Docker Desktop is running and you have sufficient permissions.
