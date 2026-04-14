import pickle
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================

path = "C:/Users/lefor/Desktop/OneDriveBackupFiles/Documentos/Q4 MUEI/TFM/GitHub/NN/"

BSL = pd.read_parquet(path + "BSL_allChan.parquet")
SENS = pd.read_parquet(path + "SENS_allChan.parquet")
DELAY = pd.read_parquet(path + "DELAY_allChan.parquet")

# =========================
# 2. RESHAPE PER SUBJECT
# =========================

def reshape_per_subject(df, expected_channels=64):
    eeg_cols = [c for c in df.columns if c.startswith("EEG.V")]
    data = {}

    grouped = df.groupby(["subjectID", "trialID"], sort=False)

    for (subj, trial), group in grouped:
        group = group.sort_values("channel")
        if len(group) != expected_channels:
            continue

        trial_data = group[eeg_cols].to_numpy(dtype=np.float32)  # (channels, time)

        # ✅ NORMALIZATION (per channel, per trial)
        mean = trial_data.mean(axis=1, keepdims=True)
        std = trial_data.std(axis=1, keepdims=True) + 1e-5
        trial_data = (trial_data - mean) / std

        label = group["y"].iloc[0]

        if subj not in data:
            data[subj] = {"X": [], "y": []}

        data[subj]["X"].append(trial_data)
        data[subj]["y"].append(label)

    # Convert to numpy
    for subj in data:
        data[subj]["X"] = np.stack(data[subj]["X"])  # (trials, channels, time)
        data[subj]["y"] = np.array(data[subj]["y"])

    return data

bsl_data = reshape_per_subject(BSL)
sens_data = reshape_per_subject(SENS)
delay_data = reshape_per_subject(DELAY)

# =========================
# 3. MODEL (CNN)
# =========================

class CNN(nn.Module):
    def __init__(self, n_channels=64, n_classes=3, dropout=0.6):
        super().__init__()

        # Temporal filtering (interpretable)
        self.temporal = nn.Conv1d(n_channels, 16, kernel_size=25,
            padding=12, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        # Spatial filtering (channel mixing)
        self.spatial = nn.Conv1d(16, 32,
            kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(32)

        # Feature extraction
        self.conv = nn.Conv1d(32, 64,
            kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.temporal(x)))
        x = F.relu(self.bn2(self.spatial(x)))
        x = F.relu(self.bn3(self.conv(x)))

        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)

        return self.fc(x)

# =========================
# 4. TRAIN FUNCTION
# =========================

def train_model(model, train_loader, val_loader, device, epochs=50):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_loss = float("inf")
    patience = 10
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            xb = xb + 0.01 * torch.randn_like(xb)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0
        model.eval()

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    return np.array(all_preds), np.array(all_labels), train_losses, val_losses

# =========================
# 5. CROSS-VALIDATION
# =========================

def run_cv(X, y, n_splits=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_classes = len(np.unique(y))
    accuracies = []
    conf_matrix_total = np.zeros((n_classes, n_classes))

    all_train_losses = []
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                               torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        model = CNN(n_channels=64, n_classes=n_classes).to(device)

        preds, labels, train_losses, val_losses = train_model(
            model, train_loader, val_loader, device
        )

        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        accuracies.append(acc)
        conf_matrix_total += cm

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    return np.mean(accuracies), conf_matrix_total, all_train_losses, all_val_losses
# =========================
# 6. RUN PER SUBJECT
# =========================

def run_all_subjects(data, name):
    print(f"\n===== {name} =====")

    results = {}

    with open(f"NN/results/results_{name}.txt", "w") as f:

        f.write(f"===== {name} =====\n")

        for subj, d in data.items():
            print(f"\nSubject {subj}")
            f.write(f"\nSubject {subj}\n")

            X = d["X"]

            le = LabelEncoder()
            y = le.fit_transform(d["y"])

#            print("Unique labels:", np.unique(y)) 
            f.write(f"Unique labels: {np.unique(y)}\n")

            acc, cm, train_losses, val_losses = run_cv(X, y)

            print(f"Mean Accuracy: {acc:.4f}")
            print("Confusion Matrix:\n", cm)

            f.write(f"Mean Accuracy: {acc:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

            results[subj] = {
                "accuracy": acc,
                "conf_matrix": cm,
                "train_losses": train_losses,
                "val_losses": val_losses}

    return results

# =========================
# 7. EXECUTION
# =========================

results_bsl = run_all_subjects(bsl_data, "BSL")
results_sens = run_all_subjects(sens_data, "SENS")
results_delay = run_all_subjects(delay_data, "DELAY")

# =========================
# 8. RESULTS
# =========================

def summarize_results(results, name):
    accs = [res["accuracy"] for res in results.values()]
    print(f"\n{name} Mean Accuracy across subjects: {np.mean(accs):.4f}")

results_bsl_summary = summarize_results(results_bsl, "BSL")
results_sens_summary = summarize_results(results_sens, "SENS")
results_delay_summary = summarize_results(results_delay, "DELAY")

with open("NN/results/results_bsl.pkl", "wb") as f:
    pickle.dump(results_bsl, f)

with open("NN/results/results_sens.pkl", "wb") as f:
    pickle.dump(results_sens, f)

with open("NN/results/results_delay.pkl", "wb") as f:
    pickle.dump(results_delay, f)