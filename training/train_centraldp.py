
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os

# ======================= MODEL DEFINITION =======================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ======================= CONFIGS =======================

MAX_FEATURES = 300
EPOCHS = 1
BATCH_SIZE = 32
CLIP = 1.0
NOISE_MULTIPLIER = 1.1
DELTA = 1e-5
LEARNING_RATE = 0.01
TOTAL_ROUNDS = 30

# ======================= DATA LOADING =======================

class Sent140Dataset(Dataset):
    def __init__(self, texts, labels):
        self.X = texts.toarray()  # Convert sparse to dense here
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================= DP TRAINING =======================

def add_noise(grads, sensitivity=1.0, sigma=NOISE_MULTIPLIER):
    return [g + torch.normal(0, sigma * sensitivity, size=g.shape).to(g.device) for g in grads]

def clip_grads(grads, max_norm=CLIP):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * clip_coef for g in grads]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = []
epsilon_list = []

for round in range(1, TOTAL_ROUNDS + 1):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Round {round}"):
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)  # [batch, seq_len, input_size]

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Clip and add noise to gradients
        grads = [p.grad for p in model.parameters()]
        clipped_grads = clip_grads(grads)
        noisy_grads = add_noise(clipped_grads)
        for p, g in zip(model.parameters(), noisy_grads):
            p.grad = g

        optimizer.step()
        total_loss += loss.item()

    # Privacy Accounting (simple linear approximation)
    epsilon = round * NOISE_MULTIPLIER
    epsilon_list.append(epsilon)

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            output = model(x)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    metrics.append((round, acc, epsilon))
    print(f"✅ Round {round}: Accuracy = {acc:.4f}, ε ≈ {epsilon:.2f}")

# ======================= LOGGING =======================

os.makedirs("logs", exist_ok=True)
with open("logs/centraldp_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, epsilons = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Central DP FL: Accuracy vs Rounds")
plt.savefig("logs/centraldp_privacy_vs_rounds.png")
plt.clf()

plt.plot(epsilons, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("Central DP FL: Accuracy vs Privacy Loss")
plt.savefig("logs/centraldp_privacy_vs_accuracy.png")
