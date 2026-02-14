# Run Script (`run.py`)

## Core Functionality
The `run.py` script is the **entry point** for the application. It handles argument parsing, configuration loading, and initiates the training process.

It serves as the glue between the user (command line) and the internal logic (`trainer.py`).

## ELI5 Description
This is the "Start" button. It reads the instruction manual (config file) you give it, sets up the workspace, and tells the coach (trainer) to start the practice session.

## Functions

### `main`
-   **Description**:
    1.  Sets up `argparse` to accept `--config`.
    2.  Reads the YAML configuration file (e.g., `configs/elliptic.yaml`).
    3.  Flattens the nested YAML config into a single dictionary (`train_config`) suitable for the `train` function.
    4.  Calls `train(train_config)`.

## Usage
```bash
python run.py --config configs/elliptic.yaml
```
