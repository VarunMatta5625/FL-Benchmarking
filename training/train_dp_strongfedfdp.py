import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os, csv
import pickle
from tqdm import tqdm
from utils.logger_utils import append_log, init_csv_logger

class StrongFDPLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TFIDFDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx].toarray(), dtype=torch.float32).squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def clip_and_fdp_noise(grads, clip, sigma, tau):
    norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    scale = min(1.0, clip / (norm + 1e-6))
    clipped = [g * scale for g in grads]
    noised = [g + torch.normal(0, sigma * tau, size=g.shape).to(g.device) for g in clipped]
    return noised

def train(config, model, train_data, device, run_name="dp_strongfedfdp_sent140"):
    rounds = config["rounds"]
    lr = config["learning_rate"]
    clip = config["clip_threshold"]
    sigma = config["noise_stddev"]
    tau = config["tau"]
    batch_size = config["batch_size"]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    log_path = f"logs/{run_name}.csv"
    init_csv_logger(log_path)

    for rnd in range(1, rounds + 1):
        model.train()
        total_loss = 0

        loader = DataLoader(train_data["train_dataset"], batch_size=batch_size, shuffle=True)
        for x, y in tqdm(loader, desc=f"[Round {rnd}]"):
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            grads = [p.grad for p in model.parameters()]
            clipped_noised = clip_and_fdp_noise(grads, clip, sigma, tau)
            for p, g in zip(model.parameters(), clipped_noised):
                p.grad = g
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in DataLoader(train_data["test_dataset"], batch_size=batch_size):
                x, y = x.to(device), y.to(device)
                x = x.unsqueeze(1)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        epsilon = rnd * sigma * tau

        append_log(log_path, {
            "round": rnd,
            "accuracy": round(acc, 4),
            "train_loss": round(total_loss, 4),
            "epsilon": round(epsilon, 2),
            "noise_stddev": sigma,
            "clip_threshold": clip,
            "pre_clip_grad_norm": "N/A",
            "post_clip_grad_norm": "N/A",
            "clients_sampled": "Centralized",
            "communication_MB": 0
        })