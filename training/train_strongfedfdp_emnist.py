import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os, csv

# ==== CONFIG ====
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# ==== DATASET WRAPPER ====
class ClientDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x.float(), torch.tensor(y, dtype=torch.long)

# ==== LOAD CLIENTS ====
clients = []
for i in range(NUM_CLIENTS):
    data = torch.load(f"data/emnist/clients/client_{i}.pt", weights_only=False)
    clients.append(ClientDataset(data))

# ==== LOAD TEST DATA ====
test_data = torch.load("data/emnist/emnist_test.pt", weights_only=False)
test_loader = DataLoader(ClientDataset(test_data), batch_size=BATCH_SIZE)

# ==== MODEL ====
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 62)
        )

    def forward(self, x):
        return self.net(x)

# ==== DP UTILS ====
def clip_gradients(grads, max_norm):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    scale = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * scale for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# ==== TRAINING ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
metrics = []

for rnd in range(1, TOTAL_ROUNDS + 1):
    agg_grads = []

    for client in clients:
        model = MLP().to(device)
        model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(client, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Gradient diff
        grads = [p.data - g.data for p, g in zip(model.parameters(), global_model.parameters())]
        clipped = clip_gradients(grads, CLIP_NORM)
        noised = add_noise(clipped, NOISE_MULTIPLIER, device)
        agg_grads.append(noised)

    # Average + Update
    avg_grad = [sum(g[i] for g in agg_grads) / NUM_CLIENTS for i in range(len(agg_grads[0]))]
    for param, g in zip(global_model.parameters(), avg_grad):
        param.data += g

    # Evaluation
    global_model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = global_model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    eps = rnd * NOISE_MULTIPLIER
    metrics.append((rnd, acc, eps))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {eps:.2f}")

# ==== LOGGING ====
with open(f"{LOG_PATH}/strongfedfdp_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

# ==== PLOTTING ====
import matplotlib.pyplot as plt

rounds, accs, eps = zip(*metrics)

plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Strong Fed f-DP: Accuracy vs Rounds")
plt.savefig(f"{LOG_PATH}/strongfedfdp_privacy_vs_rounds.png")
plt.clf()

plt.plot(eps, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("Strong Fed f-DP: Accuracy vs Privacy")
plt.savefig(f"{LOG_PATH}/strongfedfdp_privacy_vs_accuracy.png")