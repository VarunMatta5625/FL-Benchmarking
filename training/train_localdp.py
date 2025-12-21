import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv

# ======================= MODEL =======================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ======================= DATASET =======================
class Sent140Dataset(Dataset):
    def __init__(self, texts, labels):
        self.X = texts
        self.y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0), torch.tensor(self.y[idx], dtype=torch.long)

# ======================= DP UTILS =======================
def add_noise(tensor, sigma):
    noise = torch.normal(0, sigma, size=tensor.shape).to(tensor.device)
    return tensor + noise

if __name__ == "__main__":
    # ======================= CONFIGS =======================
    MAX_FEATURES = 300
    BATCH_SIZE = 32
    EPOCHS = 1
    CLIP = 1.0
    SIGMA = 1.1  # Noise multiplier
    TOTAL_ROUNDS = 30
    LEARNING_RATE = 0.01

    # ======================= DATA PREP =======================
    df = pd.read_csv("data/sent140/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['label', 'text']
    df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_dataset = Sent140Dataset(X_train, y_train)
    test_dataset = Sent140Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=11)

    # ======================= INIT =======================
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMClassifier(input_dim=MAX_FEATURES, hidden_dim=128, output_dim=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    # ======================= TRAIN =======================
    for rnd in range(1, TOTAL_ROUNDS + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Round {rnd}"):
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # Add local noise to gradients before update
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.clamp(param.grad, -CLIP, CLIP)
                    param.grad = add_noise(param.grad, sigma=SIGMA)

            optimizer.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = x.unsqueeze(1)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        epsilon = rnd * SIGMA
        history.append((rnd, acc, epsilon))
        print(f"✅ Round {rnd}: Accuracy = {acc:.4f}, ε ≈ {epsilon:.2f}")

    # ======================= LOG =======================
    os.makedirs("logs", exist_ok=True)
    with open("logs/localdp_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy", "Epsilon"])
        writer.writerows(history)

    rounds, accs, epsilons = zip(*history)
    plt.plot(rounds, accs)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Local DP FL: Accuracy vs Rounds")
    plt.savefig("logs/localdp_privacy_vs_rounds.png")
    plt.clf()

    plt.plot(epsilons, accs)
    plt.xlabel("Privacy Loss (ε)")
    plt.ylabel("Accuracy")
    plt.title("Local DP FL: Accuracy vs Privacy")
    plt.savefig("logs/localdp_privacy_vs_accuracy.png")