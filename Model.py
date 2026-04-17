# ==========================================
# INSTALL IF NEEDED
# pip install kagglehub torch pandas numpy scikit-learn
# ==========================================

import os
import ast
import numpy as np
import pandas as pd
import kagglehub
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. DOWNLOAD DATASET
# ==========================================
dataset_root = kagglehub.dataset_download(
    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
)

print("Downloaded dataset root:", dataset_root)

# ==========================================
# 2. FIND CORRECT DATASET PATH
# ==========================================
def find_dataset_path(root_path):
    for root, dirs, files in os.walk(root_path):
        if "labeled_anomalies.csv" in files:
            return root
    return None

dataset_path = find_dataset_path(dataset_root)

if dataset_path is None:
    raise FileNotFoundError("Could not locate labeled_anomalies.csv")

print("Actual dataset path:", dataset_path)

# ==========================================
# 3. LOAD DATASET
# ==========================================
def load_nasa_dataset(base_path, spacecraft="SMAP"):
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    label_path = os.path.join(base_path, "labeled_anomalies.csv")

    labels_df = pd.read_csv(label_path)
    labels_df["spacecraft"] = labels_df["spacecraft"].str.upper()
    labels_df = labels_df[labels_df["spacecraft"] == spacecraft.upper()]

    X_train_list = []
    X_test_list = []
    y_test_list = []

    for _, row in labels_df.iterrows():
        chan_id = str(row["chan_id"]).strip()

        train_file = os.path.join(train_path, chan_id + ".npy")
        test_file = os.path.join(test_path, chan_id + ".npy")

        if not os.path.exists(train_file):
            continue
        if not os.path.exists(test_file):
            continue

        train_data = np.load(train_file)
        test_data = np.load(test_file)

        labels = np.zeros(len(test_data))
        anomaly_ranges = ast.literal_eval(row["anomaly_sequences"])

        for start, end in anomaly_ranges:
            labels[start:end+1] = 1

        X_train_list.append(train_data)
        X_test_list.append(test_data)
        y_test_list.append(labels)

    if len(X_train_list) == 0:
        raise ValueError("No telemetry files loaded.")

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_test


# ==========================================
# 4. CREATE WINDOWS
# ==========================================
def create_windows(data, labels=None, window_size=10):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])

        if labels is not None:
            y.append(np.max(labels[i:i+window_size]))

    if labels is None:
        return np.array(X)

    return np.array(X), np.array(y)


# ==========================================
# 5. MODEL DEFINITIONS
# ==========================================
class USAD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder1(z), self.decoder2(z)


class TranAD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.fc(self.encoder(x))


class GDN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, input_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ==========================================
# 6. TRAINING
# ==========================================
def train_usad(model, loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        for batch in loader:
            x = batch[0].to(device)
            x = x.view(x.size(0), -1)

            w1, w2 = model(x)
            loss = criterion(w1, x) + criterion(w2, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_model(model, loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        for batch in loader:
            x = batch[0].to(device)

            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ==========================================
# 7. EVALUATION
# ==========================================
def get_metrics(y_true, scores):
    threshold = np.percentile(scores, 95)
    preds = (scores > threshold).astype(int)

    return {
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1": f1_score(y_true, preds),
        "ROC_AUC": roc_auc_score(y_true, scores)
    }


def evaluate_usad(model, loader, y_true):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            x = x.view(x.size(0), -1)

            output, _ = model(x)
            err = torch.mean((x - output) ** 2, dim=1)

            scores.extend(err.cpu().numpy())

    return get_metrics(y_true, scores)


def evaluate_model(model, loader, y_true):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)

            output = model(x)
            err = torch.mean((x - output) ** 2, dim=(1, 2))

            scores.extend(err.cpu().numpy())

    return get_metrics(y_true, scores)


# ==========================================
# 8. MAIN EXECUTION
# ==========================================
X_train, X_test, y_test = load_nasa_dataset(dataset_path, "SMAP")

X_train_seq = create_windows(X_train, window_size=10)
X_test_seq, y_test_seq = create_windows(X_test, y_test, window_size=10)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32)),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32)),
    batch_size=64,
    shuffle=False
)

results = {}

# USAD
usad = USAD(X_train_seq.shape[1] * X_train_seq.shape[2]).to(device)
train_usad(usad, train_loader)
results["USAD"] = evaluate_usad(usad, test_loader, y_test_seq)

# TranAD
tranad = TranAD(X_train_seq.shape[2]).to(device)
train_model(tranad, train_loader)
results["TranAD"] = evaluate_model(tranad, test_loader, y_test_seq)

# GDN
gdn = GDN(X_train_seq.shape[2]).to(device)
train_model(gdn, train_loader)
results["GDN"] = evaluate_model(gdn, test_loader, y_test_seq)

# ==========================================
# 9. SHOW RESULTS
# ==========================================
print("\nMODEL COMPARISON RESULTS\n")

for model_name, metrics in results.items():
    print(model_name)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 30)
