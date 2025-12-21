# train_fedsyn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ========== CONFIG ==========
MAX_FEATURES = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.01
TOTAL_ROUNDS = 30
NUM_CLIENTS = 50
NOISE_MULTIPLIER = 1.0
CLIP = 1.0

# ========== MODEL ==========
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ========== DP UTILS ==========
def clip_grads(grads, max_norm):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * coef for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# ========== DATASET ==========
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

client_data_size = X_train.shape[0] // NUM_CLIENTS
clients = [
    Sent140Dataset(X_train[i * client_data_size:(i + 1) * client_data_size],
                   y_train[i * client_data_size:(i + 1) * client_data_size])
    for i in range(NUM_CLIENTS)
]
test_dataset = Sent140Dataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)

# ========== TRAINING ==========
metrics = []
for rnd in range(1, TOTAL_ROUNDS + 1):
    model.train()
    synthetic_grads = []
    for client_data in clients:
        loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
        local_model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
        local_model.load_state_dict(model.state_dict())
        opt = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            opt.zero_grad()
            out = local_model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            opt.step()

        grads = [p1.data - p2.data for p1, p2 in zip(local_model.parameters(), model.parameters())]
        clipped = clip_grads(grads, CLIP)
        noised = add_noise(clipped, NOISE_MULTIPLIER, device)
        synthetic_grads.append(noised)

    # Average gradients
    avg_grads = [sum(g[i] for g in synthetic_grads) / NUM_CLIENTS for i in range(len(synthetic_grads[0]))]
    for param, grad in zip(model.parameters(), avg_grads):
        param.data += grad

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    eps = rnd * NOISE_MULTIPLIER
    metrics.append((rnd, acc, eps))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {eps:.2f}")

# ========== LOGGING ==========
os.makedirs("logs", exist_ok=True)
with open("logs/fedsyn_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, epsilons = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("FedSyn + DP: Accuracy vs Rounds")
plt.savefig("logs/fedsyn_accuracy_vs_rounds.png")
plt.clf()

plt.plot(epsilons, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("FedSyn + DP: Accuracy vs Privacy")
plt.savefig("logs/fedsyn_accuracy_vs_privacy.png")