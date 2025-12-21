import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dp_utils import clip_gradients, add_noise
from utils.logger_utils import append_log
from utils.sent140_loader import TweetDataset

def train_one_client(model, data, config, vocab, device):
    model.train()
    loader = DataLoader(TweetDataset(data, vocab, config["sequence_length"]), batch_size=config["batch_size"], shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    grads = []
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        num_batches += 1
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                grads.append(p.grad.clone())
        model.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    return grads, avg_loss

def evaluate(model, client_loaders, config, vocab, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for client_id in random.sample(list(client_loaders.keys()), 10):
            loader = client_loaders[client_id]
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
    return correct / total if total else 0.0

def train(config, model, train_data, device, run_name="dp_ftrl_sent140"):
    rounds = config["rounds"]
    clients_per_round = config["clients_per_round"]
    clip = config["clip_threshold"]
    sigma = config["noise_stddev"]
    log_path = f"logs/{run_name}.csv"

    vocab = train_data["vocab"]
    client_data = train_data["clients"]

    accumulated_grads = None

    for rnd in range(1, rounds + 1):
        print(f"\n--- Round {rnd} ---")
        selected_clients = random.sample(list(client_data.values()), clients_per_round)
        client_grads = []
        client_losses = []

        for data in selected_clients:
            local_model = type(model)(**model.init_args).to(device)
            local_model.load_state_dict(model.state_dict())
            grads, loss = train_one_client(local_model, data, config, vocab, device)
            clipped = clip_gradients(grads, clip)
            noisy = add_noise(clipped, sigma)
            client_grads.append(noisy)
            client_losses.append(loss)

        # Ensure gradients are same length across all clients
        client_grads = [g for g in client_grads if len(g) > 0]
        min_len = min(len(g) for g in client_grads) if client_grads else 0
        client_grads = [g for g in client_grads if len(g) == min_len]

        if not client_grads or min_len == 0:
            print(f"⚠️ Skipping round {rnd} due to inconsistent or empty gradients.")
            continue

        # Federated average gradient
        avg_grad = []
        for p in range(min_len):
            stacked = [grad[p] for grad in client_grads]
            avg_grad.append(sum(stacked) / len(stacked))

        # Accumulate for FTRL
        if accumulated_grads is None:
            accumulated_grads = avg_grad
        else:
            accumulated_grads = [accumulated_grads[i] + avg_grad[i] for i in range(len(avg_grad))]

        # Apply accumulated gradient to global model
        with torch.no_grad():
            for p_index, p in enumerate(model.parameters()):
                if p.requires_grad:
                    p -= config["learning_rate"] * accumulated_grads[p_index]

        # Metrics
        acc = evaluate(model, train_data["loaders"], config, vocab, device)
        avg_loss = sum(client_losses) / len(client_losses)
        epsilon = rnd * sigma

        append_log(log_path, {
            "round": rnd,
            "accuracy": round(acc, 4),
            "train_loss": round(avg_loss, 4),
            "epsilon": round(epsilon, 2),
            "noise_stddev": sigma,
            "clip_threshold": clip,
            "pre_clip_grad_norm": "N/A",
            "post_clip_grad_norm": "N/A",
            "clients_sampled": clients_per_round,
            "communication_MB": 0
        })