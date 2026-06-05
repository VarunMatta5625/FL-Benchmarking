"""Build the EMNIST client splits expected by the training/*_emnist.py scripts.

Downloads EMNIST (Balanced split, 47 classes) via torchvision, partitions the
training set IID across NUM_CLIENTS clients, and writes:

    data/emnist/clients/client_0.pt ... client_{NUM_CLIENTS-1}.pt
    data/emnist/emnist_test.pt

Each client file is a torch.utils.data.Subset backed by a small per-client
TensorDataset (so files stay small while still exposing the .dataset /
.indices attributes some scripts rely on). The test file is a TensorDataset.

Usage (from the repo root):
    python scripts/prepare_emnist.py
"""

import os

import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import EMNIST

NUM_CLIENTS = 50
SEED = 42
DATA_ROOT = "data"
OUT_DIR = "data/emnist"


def to_tensor_dataset(dataset, indices):
    """Materialise a list of dataset indices into a compact TensorDataset."""
    xs = torch.stack([dataset[i][0] for i in indices])
    ys = torch.tensor([dataset[i][1] for i in indices], dtype=torch.long)
    return TensorDataset(xs, ys)


def main():
    os.makedirs(os.path.join(OUT_DIR, "clients"), exist_ok=True)
    tfm = transforms.ToTensor()

    print("Downloading / loading EMNIST (balanced split)...")
    train_set = EMNIST(root=DATA_ROOT, split="balanced", train=True, download=True, transform=tfm)
    test_set = EMNIST(root=DATA_ROOT, split="balanced", train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(train_set), generator=g).tolist()
    shard = len(train_set) // NUM_CLIENTS

    print(f"Writing {NUM_CLIENTS} IID client shards ({shard} samples each)...")
    for i in range(NUM_CLIENTS):
        idx = perm[i * shard:(i + 1) * shard]
        client_ds = to_tensor_dataset(train_set, idx)
        # Wrap in a Subset: some scripts access .dataset / .indices directly.
        subset = Subset(client_ds, list(range(len(client_ds))))
        torch.save(subset, os.path.join(OUT_DIR, "clients", f"client_{i}.pt"))

    print("Writing test set...")
    test_ds = to_tensor_dataset(test_set, list(range(len(test_set))))
    torch.save(test_ds, os.path.join(OUT_DIR, "emnist_test.pt"))

    print(f"Done. {NUM_CLIENTS} client files + emnist_test.pt under {OUT_DIR}/")


if __name__ == "__main__":
    main()
