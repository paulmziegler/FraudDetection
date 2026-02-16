import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .data_loader import load_data
from .models.dga_gnn import DGAGNN


def train(config):
    """
    Main training loop for DGA-GNN.
    Implements Feedback Dynamic Grouping.

    Args:
        config (dict): A dictionary containing all necessary configuration.
    """
    # Load data using parameters from config
    g, features, labels, train_mask, val_mask, test_mask = load_data(
        config["data_path"],
        dataset_name=config["dataset_name"],
        train_split_step=config["train_split_step"],
        val_split_step=config["val_split_step"],
    )

    # Device handling
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Move data to device
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Initialize Model
    model = DGAGNN(config["in_feats"], config["hidden_dim"], config["num_classes"])
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Initial Group Labels
    group_labels = torch.zeros(g.number_of_nodes(), dtype=torch.long, device=device)

    group_labels_history = [group_labels.clone().cpu()]

    pbar = tqdm(range(config["epochs"]), desc="Training")
    for epoch in pbar:
        model.train()

        logits = model(g, features, group_labels)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)

        group_labels = preds.detach()
        group_labels_history.append(group_labels.clone().cpu())

        # Basic evaluation for logging
        acc = (logits[val_mask].argmax(dim=1) == labels[val_mask]).float().mean()
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Val Acc": f"{acc.item():.4f}"})

    print("Training complete.")
    return model, group_labels_history
