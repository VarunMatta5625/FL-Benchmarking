import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, csv, random
import matplotlib.pyplot as plt

# ==== CONFIG ====
NUM_CLIENTS = 50
TOTAL_ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05
LOG_PATH = "logs/EMNIST"
os.makedirs(LOG_PATH, exist_ok=True)

# ==== DATA ====
class ClientDataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

clients = []
for i in range(NUM_CLIENTS):
    client_data = torch.load(f"data/emnist/clients/client_{i}.pt", weights_only=False)
    clients.append(ClientDataset(client_data))

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

# ==== RAPPOR NOISE SIMULATION ====
def rappor_encode(labels, num_classes=62, prob=0.75):
    noisy_labels = []
    for label in labels:
        if random.random() < prob:
            noisy_labels.append(label)
        else:
            noisy_labels.append(random.randint(0, num_classes - 1))
    return torch.tensor(noisy_labels)

# ==== TRAINING ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
metrics = []

for rnd in range(1, TOTAL_ROUNDS + 1):
    global_state = global_model.state_dict()
    local_weights = []

    for client in clients:
        model = MLP().to(device)
        model.load_state_dict(global_state)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(client, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for x, y in loader:
            x = x.to(device)
            y = rappor_encode(y).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        local_weights.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

    # Aggregate
    new_state = {}
    for k in global_state.keys():
        new_state[k] = sum([w[k] for w in local_weights]) / NUM_CLIENTS
    global_model.load_state_dict(new_state)

    # Eval
    global_model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = global_model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    eps = rnd  # placeholder for symbolic ε per round
    metrics.append((rnd, acc, eps))
    print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {eps:.2f}")

# ==== LOGGING ====
with open(f"{LOG_PATH}/rappor_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Accuracy", "Epsilon"])
    writer.writerows(metrics)

rounds, accs, epsilons = zip(*metrics)
plt.plot(rounds, accs)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("RAPPOR-FL: Accuracy vs Rounds")
plt.savefig(f"{LOG_PATH}/rappor_accuracy_vs_rounds.png")
plt.clf()

plt.plot(epsilons, accs)
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("RAPPOR-FL: Accuracy vs Privacy")
plt.savefig(f"{LOG_PATH}/rappor_privacy_vs_accuracy.png")