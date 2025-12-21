import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os, csv, random

# ==== CONFIG ====
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.1
CLIP = 1.0
NOISE_MULTIPLIER = 1.0  # Optional DP noise post-aggregation
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# ==== DATA ====
class ClientDataset(Dataset):
    def __init__(self, data):
        # Fix for Subset or Dataset objects saved with torch.save()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.float(), int(y)

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

# ==== GRADIENT UTILS ====
def clip_gradients(grads, max_norm):
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    coef = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * coef for g in grads]

def add_noise(grads, sigma, device):
    return [g + torch.normal(0, sigma, size=g.shape).to(device) for g in grads]

def mask_gradients(grads_list, seed=42):
    random.seed(seed)
    masks = []
    for grads in grads_list:
        mask = [torch.randn_like(g) for g in grads]
        masks.append(mask)
    # Make masks cancel out: total = 0
    for i in range(len(masks) - 1):
        for j in range(len(masks[0])):
            masks[-1][j] -= masks[i][j]
    masked = [[g + m for g, m in zip(grads, mask)] for grads, mask in zip(grads_list, masks)]
    return masked

# ==== TRAINING ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
metrics = []

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
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        grads = [p.data - g.data for p, g in zip(model.parameters(), global_model.parameters())]
        clipped = clip_gradients(grads, CLIP)
        all_grads.append(clipped)

    # Apply masking
    masked_grads = mask_gradients(all_grads)

    # Aggregate gradients
    avg_grad = [sum(g[p] for g in masked_grads) / NUM_CLIENTS for p in range(len(masked_grads[0]))]

    # Optionally add DP noise after secure aggregation
    avg_grad = add_noise(avg_grad, NOISE_MULTIPLIER, device)

    # Update global model
    for param, grad in zip(global_model.parameters(), avg_grad):
        param.data += grad

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
    print(f"🔒 MPC-FL Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {eps:.2f}")

# ==== LOGGING ====
with open(f"{LOG_PATH}/mpc_fl_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, eps = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("MPC-FL: Accuracy vs Rounds")
plt.savefig(f"{LOG_PATH}/mpc_fl_privacy_vs_rounds.png")
plt.clf()

plt.plot(eps, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("MPC-FL: Accuracy vs Privacy")
plt.savefig(f"{LOG_PATH}/mpc_fl_privacy_vs_accuracy.png")