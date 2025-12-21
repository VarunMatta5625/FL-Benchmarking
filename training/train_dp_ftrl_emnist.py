import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os, csv, matplotlib.pyplot as plt

# === Required for loading Subset and EMNIST from .pt ===
import torch.utils.data.dataset
import torchvision.datasets.mnist
torch.serialization.add_safe_globals([
    torch.utils.data.dataset.Subset,
    torchvision.datasets.mnist.EMNIST
])

# === CONFIG ===
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.1
CLIP = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# === DATA LOADING ===
class ClientDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

clients = []
for i in range(NUM_CLIENTS):
    path = f"data/emnist/clients/client_{i}.pt"
    subset = torch.load(path, weights_only=False)
    clients.append(ClientDataset(subset))

test_data = torch.load("data/emnist/emnist_test.pt", weights_only=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# === MODEL ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 62)  # EMNIST has 62 classes
        )

    def forward(self, x):
        return self.net(x)

# === DP UTILS ===
def clip_gradients(grads, max_norm):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * coef for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# === TRAINING ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
metrics = []

accumulator = [torch.zeros_like(p.data) for p in global_model.parameters()]

for rnd in range(1, TOTAL_ROUNDS + 1):
    all_grads = []
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

        grads = [p.data - g.data for p, g in zip(model.parameters(), global_model.parameters())]
        clipped = clip_gradients(grads, CLIP)
        noised = add_noise(clipped, NOISE_MULTIPLIER, device)
        all_grads.append(noised)

    avg_grad = [sum(grads[p] for grads in all_grads) / NUM_CLIENTS for p in range(len(all_grads[0]))]
    accumulator = [a + g for a, g in zip(accumulator, avg_grad)]
    for param, acc in zip(global_model.parameters(), accumulator):
        param.data += acc

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

# === LOGGING ===
with open(f"{LOG_PATH}/dp_ftrl_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, eps = zip(*metrics)

plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("DP-FTRL: Accuracy vs Rounds")
plt.savefig(f"{LOG_PATH}/dp_ftrl_privacy_vs_rounds.png")
plt.clf()

plt.plot(eps, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("DP-FTRL: Accuracy vs Privacy")
plt.savefig(f"{LOG_PATH}/dp_ftrl_privacy_vs_accuracy.png")