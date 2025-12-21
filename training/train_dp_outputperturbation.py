import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils.model_utils import get_model
from utils.dp_utils import add_noise_to_model
from utils.logger_utils import append_log
from utils.eval_utils import evaluate_model

def train(config, model, train_data, device, run_name="dp_outputperturbation_sent140"):
    rounds = config["rounds"]
    clients_per_round = config["clients_per_round"]
    learning_rate = config["learning_rate"]
    noise_stddev = config["noise_stddev"]
    batch_size = config["batch_size"]

    vocab = train_data["vocab"]
    test_loader = train_data["loaders"]["test"]
    client_loaders = train_data["loaders"]

    log_path = f"logs/{run_name}.csv"
    os.makedirs("logs", exist_ok=True)

    for rnd in range(1, rounds + 1):
        print(f"\n--- Round {rnd} ---")
        global_state = model.state_dict()
        client_ids = list(client_loaders.keys())
        client_ids.remove("test")
        selected_clients = random.sample(client_ids, clients_per_round)

        agg_weights = None

        for cid in selected_clients:
            local_model = get_model(dataset="sent140", config=config).to(device)
            local_model.load_state_dict(global_state)
            optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            loader = client_loaders[cid]

            local_model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = local_model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            # Output Perturbation
            add_noise_to_model(local_model, noise_stddev)

            # Aggregate models
            local_state = local_model.state_dict()
            if agg_weights is None:
                agg_weights = {k: v.clone() for k, v in local_state.items()}
            else:
                for k in agg_weights:
                    agg_weights[k] += local_state[k]

        # Federated average
        for k in agg_weights:
            agg_weights[k] /= clients_per_round

        model.load_state_dict(agg_weights)

        # Evaluate global model
        acc, loss = evaluate_model(model, test_loader, device)
        epsilon = rnd * noise_stddev

        # Log
        append_log(log_path, {
            "round": rnd,
            "accuracy": round(acc, 4),
            "train_loss": round(loss, 4),
            "epsilon": round(epsilon, 2),
            "noise_stddev": noise_stddev,
            "clip_threshold": "N/A",
            "pre_clip_grad_norm": "N/A",
            "post_clip_grad_norm": "N/A",
            "clients_sampled": clients_per_round,
            "communication_MB": 0  # Optional placeholder
        })