# Data setup

The `data/` directory is git-ignored. Recreate it as follows before running any
training script. All scripts are run from the repo root and use these relative
paths.

## EMNIST (all `training/*_emnist.py` scripts)

Run the prep script once — it downloads EMNIST (Balanced split, 47 classes) via
torchvision and builds the 50 IID client shards the scripts expect:

```bash
python scripts/prepare_emnist.py
```

This produces:

```
data/emnist/clients/client_0.pt ... client_49.pt   # one Subset per client
data/emnist/emnist_test.pt                          # held-out test set
```

## Sent140 (all Sent140 scripts)

Download the Sentiment140 dataset (1.6M labelled tweets) and place the training
CSV at:

```
data/sent140/training.1600000.processed.noemoticon.csv
```

Sources:
- Kaggle: https://www.kaggle.com/datasets/kazanova/sentiment140
- Stanford (original): http://help.sentiment140.com/for-students

The scripts read columns 0 (label: 0 = negative, 4 = positive) and 5 (tweet
text) and vectorise the text with TF-IDF.
