import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader

from utils.model_utils import get_model
from utils.logger_utils import append_log, init_csv_logger
from utils.dp_utils import clip_gradients, add_noise
from utils.eval_utils import evaluate_model


def train(config, model, train_data, device, run_name="dp_fedsyn_sent140"):
    rounds = config["rounds"]
    clients_per_round = config["clients_per_round"]
    learning_rate = config["learning_rate"]
    noise_stddev = config["noise_stddev"]
    batch_size = config["batch_size"]
    clip_threshold = config["clip_threshold"]

    client_data = train_data["clients"]
    test_loader = train_data["loaders"]["test"]

    log_path = f"logs/{run_name}.csv"
    init_csv_logger(log_path)

    for rnd in range(1, rounds + 1):
        print(f"\n--- Round {rnd} ---")

        global_grads = None
        selected_clients = random.sample(list(client_data.values()), clients_per_round)

        for client_idx, dataset in enumerate(selected_clients):
            # ✅ Use the full config to create a matching local model
            local_model = get_model("sent140", config).to(device)
            local_model.load_state_dict(model.state_dict())
            local_model.train()

            optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_data["collate_fn"])
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = local_model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            grads = [p1.data - p2.data for p1, p2 in zip(local_model.parameters(), model.parameters())]
            clipped = clip_gradients(grads, clip_threshold)
            noised = add_noise(clipped, noise_stddev)

            if global_grads is None:
                global_grads = noised
            else:
                for i in range(len(global_grads)):
                    global_grads[i] += noised[i]

        # Average gradients
        for i in range(len(global_grads)):
            global_grads[i] /= clients_per_round

        # Apply to global model
        with torch.no_grad():
            for param, grad in zip(model.parameters(), global_grads):
                param.data += grad

        # Evaluate global model
        acc, loss = evaluate_model(model, test_loader, device)
        epsilon = rnd * noise_stddev

        # Log metrics
        append_log(log_path, {
            "round": rnd,
            "accuracy": round(acc, 4),
            "train_loss": round(loss, 4),
            "epsilon": round(epsilon, 2),
            "noise_stddev": noise_stddev,
            "clip_threshold": clip_threshold,
            "pre_clip_grad_norm": "N/A",
            "post_clip_grad_norm": "N/A",
            "clients_sampled": clients_per_round,
            "communication_MB": 0
        })