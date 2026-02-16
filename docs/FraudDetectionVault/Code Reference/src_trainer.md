# Trainer (`src/trainer.py`)

## Core Functionality
The `trainer.py` module implements the training loop. While standard PyTorch loops are common, this one implements the unique **Feedback Loop** required by DGA-GNN.

**The Feedback Mechanism:**
In a standard GNN, the graph structure is static. In DGA-GNN, the *interpretation* of the structure changes.
1.  **Epoch T**: The model predicts labels for all nodes.
2.  **Feedback**: These predictions (`preds.detach()`) are saved as `group_labels`.
3.  **Epoch T+1**: These `group_labels` are fed back into the model input. The model now knows what it *thought* the neighbors were in the previous step.

## ELI5 Description
This is the coach. It runs the practice sessions (epochs).

1.  It tells the model to make guesses.
2.  It checks how wrong the guesses were (Loss) and teaches the model (Optimizer step).
3.  **Crucially**: It takes the guesses the model made today, writes them on sticky notes, and sticks them on the players' foreheads for *tomorrow's* practice. "Yesterday I thought you were a fraud, so today your neighbors will treat you like one."

## Functions

### `train`
-   **Signature**: `train(config)`
-   **Logic**:
    1.  Loads data (`load_data`).
    2.  Sets up the compute device (`cuda` or `cpu`) based on configuration and availability.
    3.  Moves the graph, features, masks, and model to the target device.
    4.  Initializes the `DGAGNN` model and `Adam` optimizer.
    5.  Initializes `group_labels` to all zeros (cold start) on the device.
    6.  **Loop (Epochs)**:
        -   Forward pass: `model(g, features, group_labels)`.
        -   Calculate Loss (CrossEntropy) on **only** the `train_mask` nodes.
        -   Backpropagate and Update Weights.
        -   **Update Groups**: `group_labels = logits.argmax(dim=1)`. The predictions become the new groups.
        -   Log validation accuracy.
-   **Returns**: The trained model and the history of group labels.
