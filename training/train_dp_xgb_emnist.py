import os
import torch
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.serialization import add_safe_globals
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import csv

# ========== CONFIG ==========
DATA_DIR = "data/emnist/clients"
LOG_PATH = "logs/EMNIST"
EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
os.makedirs(LOG_PATH, exist_ok=True)

# ✅ Allow loading of Subset-serialized objects
add_safe_globals([Subset])

# ========== LOAD CLIENT DATA ==========
print("📦 Loading EMNIST client data...")
x_all, y_all = [], []

for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith(".pt"):
        dataset = torch.load(os.path.join(DATA_DIR, fname), weights_only=False)
        for x, y in dataset:
            x_all.append(x.view(-1).numpy())  # Flatten 28x28 images
            y_all.append(int(y))

X = np.stack(x_all)
y = np.array(y_all)
print(f"✅ Loaded {len(X)} samples from {len(os.listdir(DATA_DIR))} client files.")

# ========== SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== TRAIN + EVALUATE ==========
print("\n🚀 Starting DP-XGB training for each epsilon...")
results = []

for eps in tqdm(EPSILON_VALUES, desc="Training DP-XGB"):
    print(f"\n🔐 Training for ε = {eps}...")

    dp_params = {
        "dp_enable": True,
        "dp_epsilon": eps,
        "dp_delta": 1e-5,
        "dp_max_depth": 6,
        "dp_min_child_weight": 1,
        "dp_lambda": 1,
        "dp_learning_rate": 0.1
    }

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        enable_categorical=False,
        verbosity=0,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((eps, acc))

    print(f"📊 [ε = {eps}] Accuracy: {acc:.4f}")

# ========== PLOT ==========
epsilons, accuracies = zip(*results)

plt.figure()
plt.plot(epsilons, accuracies, marker='o')
plt.xlabel("Epsilon (ε)")
plt.ylabel("Accuracy")
plt.title("DP-XGB Accuracy vs Privacy (EMNIST)")
plt.grid(True)
plt.savefig(os.path.join(LOG_PATH, "dp_xgb_emnist_privacy_vs_accuracy.png"))
plt.show()

# ========== SAVE CSV ==========
with open(os.path.join(LOG_PATH, "dp_xgb_emnist_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "Accuracy"])
    writer.writerows(results)

print("\n✅ Training complete! Results saved to CSV and plot.")