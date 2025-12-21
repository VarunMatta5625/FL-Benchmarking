import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from torch.serialization import add_safe_globals
from torch.utils.data.dataset import Subset
from torchvision.datasets.mnist import EMNIST

# ✅ Patch PyTorch 2.6+ loading
add_safe_globals([Subset, EMNIST])

# ========== CONFIG ==========
NUM_CLIENTS = 50
BATCH_SIZE = 32
EPOCHS = 5
ROUNDS = 10
DATA_DIR = "data/emnist/clients"
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# ========== SEED ==========
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========== DATASET WRAPPER ==========
class ClientDataset(Dataset):
    def __init__(self, data):
        self.x = torch.stack([x[0] for x in data]).view(-1, 28*28).float()
        self.y = torch.tensor([x[1] for x in data], dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ========== MODEL ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 47)  # EMNIST Balanced: 47 classes
        )
    def forward(self, x):
        return self.net(x)

# ========== LOAD CLIENTS ==========
clients = []
for fname in sorted(os.listdir(DATA_DIR))[:NUM_CLIENTS]:
    if fname.endswith(".pt"):
        data = torch.load(os.path.join(DATA_DIR, fname), weights_only=False)
        clients.append(ClientDataset(data))

# ========== FEDSYN TRAINING ==========
print("\n🧪 Starting FedSyn Training...")
global_model = MLP()
criterion = nn.CrossEntropyLoss()
acc_log = []

for rnd in range(ROUNDS):
    print(f"\n🔄 Round {rnd+1}/{ROUNDS}")
    synthetic_gradients = []

    for client in clients:
        local_model = MLP()
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=0.01)

        loader = DataLoader(client, batch_size=BATCH_SIZE, shuffle=True)
        local_model.train()
        for _ in range(EPOCHS):
            for x, y in loader:
                optimizer.zero_grad()
                output = local_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # Compute synthetic gradient
        synthetic_grads = {}
        for name, param in local_model.named_parameters():
            with torch.no_grad():
                synthetic_grads[name] = param - global_model.state_dict()[name]
        synthetic_gradients.append(synthetic_grads)

    # Aggregate synthetic gradients
    new_state = {}
    for name in global_model.state_dict().keys():
        avg_update = sum([grads[name] for grads in synthetic_gradients]) / len(synthetic_gradients)
        new_state[name] = global_model.state_dict()[name] + avg_update

    global_model.load_state_dict(new_state)
    print("✅ Synthetic update applied.")

    # Evaluation
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
    acc_log.append((rnd+1, acc))
    print(f"📊 Round {rnd+1} Accuracy: {acc:.4f}")

# ========== PLOT ==========
rounds, accuracies = zip(*acc_log)
plt.plot(rounds, accuracies, marker='o')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("FedSyn Accuracy over Rounds (EMNIST)")
plt.grid(True)
plt.savefig(os.path.join(LOG_PATH, "fedsyn_emnist_accuracy_vs_rounds.png"))
plt.show()

# ========== SAVE CSV ==========
with open(os.path.join(LOG_PATH, "fedsyn_emnist_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy"])
    writer.writerows(acc_log)