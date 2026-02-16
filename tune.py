import argparse
import itertools
import yaml
import torch
from src.trainer import train

def load_base_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_tuning(config_path):
    base_config = load_base_config(config_path)
    
    # Define hyperparameter grid
    param_grid = {
        "lr": [0.01, 0.001, 0.0001],
        "hidden_dim": [64, 128],
        "epochs": [30] # Reduced epochs for tuning speed
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting tuning with {len(combinations)} combinations...")
    
    for i, params in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)}: {params} ---")
        
        # Construct config based on base config + params
        train_config = {
            "data_path": base_config["data"]["path"],
            "dataset_name": base_config["data"]["dataset_name"],
            "train_split_step": base_config["data"]["train_split_step"],
            "val_split_step": base_config["data"]["val_split_step"],
            "in_feats": base_config["model"]["in_feats"],
            "hidden_dim": params.get("hidden_dim", base_config["model"]["hidden_dim"]),
            "num_classes": base_config["model"]["num_classes"],
            "lr": params.get("lr", base_config["training"]["lr"]),
            "epochs": params.get("epochs", base_config["training"]["epochs"]),
            "device": base_config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        }
        
        # Run training
        train(train_config)
        
    print("\nTuning Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning")
    parser.add_argument("--config", type=str, default="configs/elliptic.yaml", help="Path to base config")
    args = parser.parse_args()
    
    run_tuning(args.config)
