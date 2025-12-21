import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, random, csv
import matplotlib.pyplot as plt

# ==== CONFIG ====
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05
CLIP = 1.0
DELTA = 1e-5
BASE_NOISE_MULTIPLIER = 0.8  # Base noise multiplier
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# ==== DATA ====
class ClientDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        x = torch.tensor(self.data["x"][idx], dtype=torch.float32)
        y = torch.tensor(self.data["y"][idx], dtype=torch.long)
        return x, y

clients = []
for i in range(NUM_CLIENTS):
    subset = torch.load(f"data/emnist/clients/client_{i}.pt", weights_only=False)
    x_data = [subset.dataset[i][0] for i in subset.indices]
    y_data = [subset.dataset[i][1] for i in subset.indices]
    clients.append(ClientDataset({"x": x_data, "y": y_data}))   

test_subset = torch.load("data/emnist/emnist_test.pt", weights_only=False)
x_test = [test_subset[i][0] for i in range(len(test_subset))]
y_test = [test_subset[i][1] for i in range(len(test_subset))]
test_loader = DataLoader(ClientDataset({"x": x_test, "y": y_test}), batch_size=BATCH_SIZE)

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
    factor = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * factor for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

# ==== TRAINING ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
metrics = []

# Random personalized noise levels per client
client_epsilons = [BASE_NOISE_MULTIPLIER + random.uniform(0.0, 0.8) for _ in range(NUM_CLIENTS)]

for rnd in range(1, TOTAL_ROUNDS + 1):
    all_grads = []

    for idx, client in enumerate(clients):
        model = MLP().to(device)
        model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(client, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        grads = [p.data - g.data for p, g in zip(model.parameters(), global_model.parameters())]
        clipped = clip_gradients(grads, CLIP)
        noised = add_noise(clipped, sigma=client_epsilons[idx], device=device)
        all_grads.append(noised)

    avg_grad = [sum(grads[p] for grads in all_grads) / NUM_CLIENTS for p in range(len(all_grads[0]))]
    for param, grad in zip(global_model.parameters(), avg_grad):
        param.data += grad

    # Eval
    global_model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = global_model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    avg_epsilon = sum(client_epsilons) / NUM_CLIENTS
    metrics.append((rnd, acc, avg_epsilon))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, Avg ε ≈ {avg_epsilon:.2f}")

# ==== LOGGING ====
with open(f"{LOG_PATH}/pldp_fl_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "AvgEpsilon"])
    writer.writerows(metrics)

rounds, accs, eps = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("PLDP-FL: Accuracy vs Rounds")
plt.savefig(f"{LOG_PATH}/pldp_fl_accuracy_vs_rounds.png")
plt.clf()

plt.plot(eps, accs)
plt.xlabel("Avg Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("PLDP-FL: Accuracy vs Privacy")
plt.savefig(f"{LOG_PATH}/pldp_fl_accuracy_vs_privacy.png")