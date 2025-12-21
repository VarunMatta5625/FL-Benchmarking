import argparse
import json
import torch
from pathlib import Path
from utils.model_utils import get_model
from utils.logger_utils import init_csv_logger
from utils.sent140_loader import load_sent140_clients
from utils.emnist_loader import load_emnist_clients

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name (e.g., dp_fedavg)")
    parser.add_argument("--dataset", type=str, required=True, choices=["sent140", "emnist"])
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    # Determine device
    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])

    # Load data first (so tokenizer builds vocab)
    if args.dataset == "sent140":
        train_data = load_sent140_clients(config)
        config["vocab_size"] = train_data["vocab_size"]  # ✅ ensure model matches tokenizer
    elif args.dataset == "emnist":
        train_data = load_emnist_clients(config)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load model after vocab size is known
    model = get_model(args.dataset, config).to(device)
    

    # Load data
    if args.dataset == "sent140":
        train_data = load_sent140_clients(config)
    elif args.dataset == "emnist":
        train_data = load_emnist_clients(config)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Setup logger
    run_name = f"{args.algo}_{args.dataset}"
    log_path = Path("logs") / f"{run_name}.csv"
    init_csv_logger(log_path)

    # Import the appropriate trainer dynamically
    trainer_module = f"training.train_{args.algo}"
    try:
        trainer = __import__(trainer_module, fromlist=["train"])
    except ImportError as e:
        raise ImportError(f"Trainer not found: {trainer_module}") from e

    # Run training
    trainer.train(config=config, model=model, train_data=train_data, device=device, run_name=run_name)

if __name__ == "__main__":
    main()