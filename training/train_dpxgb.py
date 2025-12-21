import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== CONFIG ==========
MAX_FEATURES = 300
EPSILON_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]  # can be extended
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
MAX_DEPTH = 5

# ========== DATA ==========
df = pd.read_csv("data/sent140/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ["label", "text"]
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ========== TRAINING + LOGGING ==========
results = []

for eps in EPSILON_VALUES:
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        verbosity=0,
        tree_method="hist",
        enable_categorical=False,
        use_label_encoder=False
    )

    # Enable DP with XGBoost
    model.set_params(**{
        "dp_epsilon": eps,
        "dp_enabled": True,
        "dp_delta": 1e-5,
        "dp_max_depth": MAX_DEPTH
    })

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ DP-XGB ε = {eps:.2f}: Accuracy = {acc:.4f}")
    results.append((eps, acc))

# ========== SAVE LOG ==========
os.makedirs("logs", exist_ok=True)
with open("logs/dpxgb_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "Accuracy"])
    writer.writerows(results)

# ========== PLOTS ==========
epsilons, accs = zip(*results)
plt.plot(epsilons, accs, marker='o')
plt.xlabel("Privacy Loss (ε)")
plt.ylabel("Accuracy")
plt.title("DP-XGB: Accuracy vs Privacy (ε)")
plt.grid(True)
plt.savefig("logs/dpxgb_privacy_vs_accuracy.png")
plt.close()