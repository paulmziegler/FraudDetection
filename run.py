import argparse

import yaml

from src.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Run DGA-GNN Experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (e.g., configs/elliptic.yaml)",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Starting experiment with config: {args.config}")

    # Prepare a dictionary for the trainer function
    # This combines data, model, and training settings
    train_config = {
        "data_path": config["data"]["path"],
        "dataset_name": config["data"]["dataset_name"],
        "train_split_step": config["data"]["train_split_step"],
        "val_split_step": config["data"]["val_split_step"],
        "in_feats": config["model"]["in_feats"],
        "hidden_dim": config["model"]["hidden_dim"],
        "num_classes": config["model"]["num_classes"],
        "lr": config["training"]["lr"],
        "epochs": config["training"]["epochs"],
    }

    # Pass the combined config to the train function
    train(train_config)


if __name__ == "__main__":
    main()
