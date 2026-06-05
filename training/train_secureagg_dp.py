# train_pldp_fl.py

import os
import csv
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====================== Model ======================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ====================== Config ======================

MAX_FEATURES = 300
EPOCHS = 1
BATCH_SIZE = 32
CLIP = 1.0
SIGMA = 1.0
DELTA = 1e-5
LEARNING_RATE = 0.01
TOTAL_ROUNDS = 30
CLIENTS_PER_ROUND = 50
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ====================== Dataset ======================

class Sent140Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0), torch.tensor(self.y[idx], dtype=torch.long)

df = pd.read_csv("data/sent140/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ["label", "text"]
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X = vectorizer.fit_transform(df["text"])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_dataset = Sent140Dataset(X_train, y_train)
test_dataset = Sent140Dataset(X_test, y_test)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ====================== Privacy Ops ======================

def clip_grads(grads, max_norm=CLIP):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * clip_coef for g in grads]

def add_noise(grads, sigma=SIGMA):
    return [g + torch.normal(0, sigma, size=g.shape).to(g.device) for g in grads]

# ====================== Train Loop ======================

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

indices = list(range(len(train_dataset)))
client_indices = np.array_split(indices, CLIENTS_PER_ROUND)

metrics = []

for rnd in range(1, TOTAL_ROUNDS + 1):
    aggregated_grads = None
    for c in range(CLIENTS_PER_ROUND):
        model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        subset = Subset(train_dataset, client_indices[c])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        clipped = clip_grads(grads)
        noisy = add_noise(clipped)

        if aggregated_grads is None:
            aggregated_grads = noisy
        else:
            aggregated_grads = [agg + n for agg, n in zip(aggregated_grads, noisy)]

    # Update global model
    global_model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
    for p, g in zip(global_model.parameters(), aggregated_grads):
        p.grad = g / CLIENTS_PER_ROUND
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)
    optimizer.step()

    # Evaluate
    global_model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            out = global_model(x)
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    epsilon = rnd * SIGMA
    metrics.append((rnd, acc, epsilon))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {epsilon:.2f}")

# ====================== Logging ======================

os.makedirs("logs", exist_ok=True)
with open("logs/secureagg_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, epsilons = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("SecureAgg+DP: Accuracy vs Rounds")
plt.savefig("logs/secureagg_privacy_vs_rounds.png")
plt.clf()

plt.plot(epsilons, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("SecureAgg+DP: Accuracy vs Privacy Loss")
plt.savefig("logs/secureagg_privacy_vs_accuracy.png")