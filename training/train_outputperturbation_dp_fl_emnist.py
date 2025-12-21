import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
from torch.utils.data.dataset import Subset
import torchvision.datasets.mnist  # 👈 Required to access EMNIST class
import csv

# ✅ Safe globals for PyTorch 2.6+
add_safe_globals([
    Subset,
    torchvision.datasets.mnist.EMNIST
])

# ========== CONFIG ==========
NUM_CLIENTS = 50
NUM_CLASSES = 47  # ✅ EMNIST Balanced
BATCH_SIZE = 32
EPOCHS = 5
ROUNDS = 10
EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
DATA_DIR = "data/emnist/clients"
LOG_DIR = "logs/EMNIST"
os.makedirs(LOG_DIR, exist_ok=True)

# ========== SEED ==========
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========== DATASET WRAPPER ==========
class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = torch.stack([x[0] for x in data]).view(-1, 28*28).float()
        self.y = torch.tensor([x[1] for x in data], dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ========== LOAD CLIENT DATA ==========
clients = []
for fname in sorted(os.listdir(DATA_DIR))[:NUM_CLIENTS]:
    if fname.endswith(".pt"):
        data = torch.load(os.path.join(DATA_DIR, fname), weights_only=False)  # 👈 Required
        clients.append(ClientDataset(data))

# ========== MODEL ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )
    def forward(self, x):
        return self.net(x)

# ========== TRAIN + EVALUATE ==========
results = []

for eps in EPSILON_VALUES:
    print(f"\n[ε = {eps}] Training with Output Perturbation...")

    global_model = MLP()
    criterion = nn.CrossEntropyLoss()

    for rnd in range(ROUNDS):
        local_weights = []

        for client in clients:
            local_model = MLP()
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=0.01)

            loader = DataLoader(client, batch_size=BATCH_SIZE, shuffle=True)
            local_model.train()
            for _ in range(EPOCHS):
                for x, y in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(x), y)
                    loss.backward()
                    optimizer.step()

            # Output Perturbation
            noisy_state = {}
            for name, param in local_model.state_dict().items():
                noise = torch.normal(0, 1/eps, size=param.shape)
                noisy_state[name] = param + noise
            local_weights.append(noisy_state)

        # Aggregation (FedAvg)
        new_state = {}
        for key in global_model.state_dict().keys():
            new_state[key] = sum([weights[key] for weights in local_weights]) / len(local_weights)
        global_model.load_state_dict(new_state)

        print(f"  ✅ Round {rnd+1}/{ROUNDS} complete")

    # ========== Evaluation ==========
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for client in clients:
            loader = DataLoader(client, batch_size=BATCH_SIZE)
            for x, y in loader:
                out = global_model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
    acc = correct / total
    results.append((eps, acc))
    print(f"[ε = {eps}] ✅ Accuracy: {acc:.4f}")

# ========== PLOT ==========
epsilons, accuracies = zip(*results)
plt.plot(epsilons, accuracies, marker='o')
plt.xlabel("Epsilon (ε)")
plt.ylabel("Accuracy")
plt.title("Vanilla Output Perturbation DP-FL (EMNIST)")
plt.grid(True)
plt.savefig(os.path.join(LOG_DIR, "outputperturbation_dp_fl_emnist.png"))
plt.show()

# ========== SAVE CSV ==========
with open(os.path.join(LOG_DIR, "outputperturbation_dp_fl_emnist.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "Accuracy"])
    writer.writerows(results)