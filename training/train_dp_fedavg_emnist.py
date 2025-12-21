import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import random
import csv

# ========== MODEL ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 62),  # 62 classes in EMNIST Balanced
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========== CONFIG ==========
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01
CLIP = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ========== CLIENT WRAPPER ==========
class ClientDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

# ========== LOAD CLIENTS ==========
clients = []
client_files = sorted([f"./data/emnist/clients/{f}" for f in os.listdir("./data/emnist/clients") if f.endswith(".pt")])

for file in client_files:
    subset = torch.load(file, weights_only=False)
    clients.append(ClientDataset(subset))

# ========== LOAD TEST SET ==========
test_data = torch.load("./data/emnist/emnist_test.pt", weights_only=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# ========== DP UTILS ==========
def clip_grads(grads, max_norm):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * coef for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# ========== TRAINING ==========
model = SimpleCNN().to(device)
metrics = []
epsilon_list = []

for rnd in range(1, TOTAL_ROUNDS + 1):
    model.train()
    selected_clients = random.sample(clients, 10)
    all_grads = []

    for client_data in selected_clients:
        loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
        local_model = SimpleCNN().to(device)
        local_model.load_state_dict(model.state_dict())
        opt = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                output = local_model(x)
                loss = criterion(output, y)
                loss.backward()
                opt.step()

        grads = [p1.data - p2.data for p1, p2 in zip(local_model.parameters(), model.parameters())]
        clipped = clip_grads(grads, CLIP)
        noised = add_noise(clipped, NOISE_MULTIPLIER, device)
        all_grads.append(noised)

    avg_grad = [sum(g[p] for g in all_grads) / len(all_grads) for p in range(len(all_grads[0]))]
    for p, g in zip(model.parameters(), avg_grad):
        p.data += g

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
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
os.makedirs("logs/EMNIST", exist_ok=True)
with open("logs/EMNIST/dp_fedavg_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, epsilons = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("DP-FedAvg on EMNIST: Accuracy vs Rounds")
plt.savefig("logs/EMNIST/dp_fedavg_privacy_vs_rounds.png")
plt.clf()

plt.plot(epsilons, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("DP-FedAvg on EMNIST: Accuracy vs Privacy")
plt.savefig("logs/EMNIST/dp_fedavg_privacy_vs_accuracy.png")