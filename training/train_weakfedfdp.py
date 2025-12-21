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
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== MODEL ==========
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ========== CONFIG ==========
MAX_FEATURES = 300
EPOCHS = 1
BATCH_SIZE = 32
CLIP = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5
LEARNING_RATE = 0.01
TOTAL_ROUNDS = 30
NUM_CLIENTS = 50

# ========== DATA ==========
class Sent140Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]  # ✅ Fix: use shape[0] instead of len()

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

# ========== DP UTILS ==========
def clip_grads(grads, max_norm):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * coef for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# ========== TRAINING ==========
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = []
epsilon_list = []

for rnd in range(1, TOTAL_ROUNDS + 1):
    model.train()
    all_grads = []
    for client_data in clients:
        loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
        local_model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
        local_model.load_state_dict(model.state_dict())
        opt = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            opt.zero_grad()
            output = local_model(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()

        grads = [p1.data - p2.data for p1, p2 in zip(local_model.parameters(), model.parameters())]
        clipped = clip_grads(grads, CLIP)
        noised = add_noise(clipped, NOISE_MULTIPLIER, device)
        all_grads.append(noised)

    # Weak Fed f-DP: average noised grads
    avg_grad = [sum(g[p] for g in all_grads) / NUM_CLIENTS for p in range(len(all_grads[0]))]
    for p, g in zip(model.parameters(), avg_grad):
        p.data += g

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
    epsilon_list.append(eps)
    metrics.append((rnd, acc, eps))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {eps:.2f}")

# ========== LOGGING ==========
os.makedirs("logs", exist_ok=True)
with open("logs/weakfedfdp_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, epsilons = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Weak Federated f-DP: Accuracy vs Rounds")
plt.savefig("logs/weakfedfdp_privacy_vs_rounds.png")
plt.clf()

plt.plot(epsilons, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("Weak Federated f-DP: Accuracy vs Privacy")
plt.savefig("logs/weakfedfdp_privacy_vs_accuracy.png")