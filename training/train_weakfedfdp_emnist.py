import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os, csv
import matplotlib.pyplot as plt

# ==== CONFIG ====
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05
CLIP = 1.0
NOISE_MULTIPLIER = 0.8
DELTA = 1e-5
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# ==== DATA ====
class ClientDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.float(), torch.tensor(y).long()

clients = []
for i in range(NUM_CLIENTS):
    data = torch.load(f"data/emnist/clients/client_{i}.pt", weights_only=False)
    clients.append(ClientDataset(data))

test_data = torch.load("data/emnist/emnist_test.pt", weights_only=False)
test_loader = DataLoader(ClientDataset(test_data), batch_size=BATCH_SIZE)

# ==== MODEL ====
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 62)
        )

    def forward(self, x):
        return self.net(x)

# ==== DP UTILS ====
def clip_grads(grads, clip):
    norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    scale = min(1.0, clip / (norm + 1e-6))
    return [g * scale for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# ==== TRAINING ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
metrics = []

for rnd in range(1, TOTAL_ROUNDS + 1):
    all_grads = []
    for client in clients:
        local_model = MLP().to(device)
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(client, batch_size=BATCH_SIZE, shuffle=True)

        local_model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = local_model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        grads = [p.data - g.data for p, g in zip(local_model.parameters(), global_model.parameters())]
        clipped = clip_grads(grads, CLIP)
        noised = add_noise(clipped, NOISE_MULTIPLIER, device)
        all_grads.append(noised)

    # Average gradients and update global model
    avg_grads = [sum(grads[i] for grads in all_grads) / NUM_CLIENTS for i in range(len(all_grads[0]))]
    for param, g in zip(global_model.parameters(), avg_grads):
        param.data += g

    # Evaluate
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = global_model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    eps = rnd * NOISE_MULTIPLIER
    metrics.append((rnd, acc, eps))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {eps:.2f}")

# ==== LOGGING ====
with open(f"{LOG_PATH}/weakfedfdp_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, eps = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Weak Fed f-DP: Accuracy vs Rounds")
plt.savefig(f"{LOG_PATH}/weakfedfdp_privacy_vs_rounds.png")
plt.clf()

plt.plot(eps, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("Weak Fed f-DP: Accuracy vs Privacy")
plt.savefig(f"{LOG_PATH}/weakfedfdp_privacy_vs_accuracy.png")