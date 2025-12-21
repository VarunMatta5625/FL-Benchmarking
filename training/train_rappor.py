import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os

# ======================= MODEL DEFINITION =======================

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ======================= CONFIGS =======================

MAX_FEATURES = 300
BATCH_SIZE = 32
EPOCHS = 1
SIGMA = 0.4    # Noise parameter for randomized response
TOTAL_ROUNDS = 30
LEARNING_RATE = 0.01

# ======================= RAPPOR-Like Mechanism =======================

def randomized_response(vector, p=0.75, q=0.25):
    rr_vector = np.copy(vector)
    flip_mask = np.random.rand(*vector.shape) > p
    random_bits = np.random.rand(*vector.shape) < q
    rr_vector[flip_mask] = random_bits[flip_mask]
    return rr_vector

# ======================= DATA PREP =======================

class Sent140Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# ======================= LOAD AND PROCESS =======================

if __name__ == "__main__":
    df = pd.read_csv("data/sent140/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(df["text"]).toarray()
    X_noised = np.array([randomized_response(x, p=1 - SIGMA, q=SIGMA) for x in X])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X_noised, y, test_size=0.2)

    train_dataset = Sent140Dataset(X_train, y_train)
    test_dataset = Sent140Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ======================= TRAINING =======================

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleNN(input_dim=MAX_FEATURES, output_dim=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics = []

    for round in range(1, TOTAL_ROUNDS + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Round {round}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        epsilon = round * SIGMA  # Simplified tracking for illustration
        metrics.append((round, acc, epsilon))
        print(f"✅ Round {round}: Accuracy = {acc:.4f}, ε ≈ {epsilon:.2f}")

    # ======================= LOGGING =======================

    os.makedirs("logs", exist_ok=True)
    with open("logs/rappor_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy", "Epsilon"])
        writer.writerows(metrics)

    rounds, accs, epsilons = zip(*metrics)
    plt.plot(rounds, accs)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("RAPPOR (Local DP): Accuracy vs Rounds")
    plt.savefig("logs/rappor_privacy_vs_rounds.png")
    plt.clf()

    plt.plot(epsilons, accs)
    plt.xlabel("Privacy Loss (ε)")
    plt.ylabel("Accuracy")
    plt.title("RAPPOR (Local DP): Accuracy vs Privacy Loss")
    plt.savefig("logs/rappor_privacy_vs_accuracy.png")