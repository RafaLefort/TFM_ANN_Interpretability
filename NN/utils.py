"""
utils.py
========
Shared pipeline utilities for EEG working memory classification.

Functions
---------
load_subject_csv   : Load a subject CSV and return (X, y) arrays.
frequency_mixup    : Frequency-domain mixup between two EEG trials.
augment_eeg        : Online augmentation for EEGNet (noise + masking).
train_model        : Full training loop with early stopping.
run_cv             : 10-fold stratified cross-validation.
run_all_subjects   : Iterate over all subject CSVs in a folder.
summarize_results  : Print mean accuracy per experimental period.
save_results       : Save results dict to JSON.

Classes
-------

EEGDataset : PyTorch Dataset with optional frequency mixup augmentation.
"""

import os
import gc
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix)


# =========================================================
# LOAD SUBJECT CSV
# =========================================================

def load_subject_csv(
    file_path,
    expected_channels=64):
    """
    Load one subject CSV and return raw (X, y) arrays.

    Each row in the CSV corresponds to one EEG channel within one trial.
    Trials with fewer than expected_channels rows are discarded.

    Normalization is intentionally NOT applied here: it is computed
    inside each CV fold from training data only, to avoid data leakage
    from validation samples into the normalization statistics.

    Parameters
    ----------
    file_path         : str  — path to the subject CSV file.
    expected_channels : int  — trials with != expected_channels rows
                               are discarded (default: 64).

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_channels, n_samples), float32
    y : np.ndarray, shape (n_trials,), raw integer labels
    """

    df = pd.read_csv(file_path)

    eeg_cols = [
        c for c in df.columns
        if c.startswith("EEG.V")]

    X = []
    y = []

    for _, group in df.groupby("trialID", sort=False):

        group = group.sort_values("channel")

        if len(group) != expected_channels:
            continue

        trial = group[eeg_cols].to_numpy(dtype=np.float32)
        label = group["y"].iloc[0]

        X.append(trial)
        y.append(label)

    X = np.stack(X)
    y = np.array(y)

    del df
    gc.collect()

    return X, y


# =========================================================
# FREQUENCY MIXUP
# =========================================================

def frequency_mixup(x1, x2, alpha=0.4):
    """
    Frequency-domain mixup between two EEG trials.

    Interpolates the rfft spectra of x1 and x2 with ratio alpha,
    then returns the irfft of the mixed spectrum.

    Parameters
    ----------
    x1, x2 : np.ndarray, shape (n_channels, n_samples)
    alpha   : float — interpolation weight for x1 (default: 0.4)

    Returns
    -------
    mixed : np.ndarray, shape (n_channels, n_samples), float32
    """

    fft1 = np.fft.rfft(x1, axis=-1)
    fft2 = np.fft.rfft(x2, axis=-1)

    mixed_fft = alpha * fft1 + (1 - alpha) * fft2

    mixed = np.fft.irfft(mixed_fft, axis=-1)

    return mixed.astype(np.float32)


# =========================================================
# AUGMENT EEG (EEGNet online augmentation)
# =========================================================

def augment_eeg(x):
    """
    Online augmentation applied per batch during EEGNet training.

    Three independent operations, each applied with 50% probability:
      - Gaussian noise  : adds N(0, 0.01) noise to all elements.
      - Time masking    : zeros a random 10% window along the time axis.
      - Channel dropout : zeros a random 10% subset of channels.

    Parameters
    ----------
    x : torch.Tensor, shape (B, n_channels, n_samples)

    Returns
    -------
    x : torch.Tensor, same shape, augmented in-place
    """

    # Gaussian noise
    x = x + 0.01 * torch.randn_like(x)

    # Time masking
    if torch.rand(1) < 0.5:

        t         = x.size(-1)
        mask_size = int(t * 0.1)
        start     = torch.randint(0, t - mask_size, (1,))

        x[:, :, start:start + mask_size] = 0

    # Channel dropout
    if torch.rand(1) < 0.5:

        c       = x.size(1)
        drop_ch = torch.randperm(c)[:int(c * 0.1)]

        x[:, drop_ch, :] = 0

    return x


# =========================================================
# EEGDATASET
# =========================================================

class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG trials with optional frequency mixup.

    When augment=True, each sample is independently mixed with a
    randomly chosen second trial with probability 0.5. The mixing
    ratio lam is sampled from Beta(alpha, alpha) at each call,
    following the standard mixup formulation (Zhang et al., 2018).

    For compatibility with both augmented and non-augmented batches,
    __getitem__ always returns a 4-tuple:
        (x, y1, y2, lam)
    where y2 == y1 and lam == 1.0 when no augmentation is applied.

    Parameters
    ----------
    X       : np.ndarray, shape (n_trials, n_channels, n_samples)
    y       : np.ndarray, shape (n_trials,), integer labels
    augment : bool  — whether to apply frequency mixup (default: False)
    alpha   : float — Beta distribution parameter for lam (default: 0.4)
    """

    def __init__(self, X, y, augment=False, alpha=0.4):

        self.X       = X
        self.y       = y
        self.augment = augment
        self.alpha   = alpha

        self.total_samples = 0
        self.mixup_count   = 0
        self.no_aug_count  = 0


    def __len__(self):

        return len(self.X)


    def __getitem__(self, idx):

        x1 = self.X[idx].copy()
        y1 = self.y[idx]

        self.total_samples += 1

        if self.augment and np.random.rand() < 0.5:

            j = np.random.randint(0, len(self.X) - 1)

            if j >= idx:
                j += 1

            x2  = self.X[j]
            y2  = self.y[j]

            lam = float(np.random.beta(self.alpha, self.alpha))

            x_mix = frequency_mixup(x1, x2, alpha=lam)

            self.mixup_count += 1

            return (
                torch.tensor(x_mix, dtype=torch.float32),
                torch.tensor(y1,    dtype=torch.long),
                torch.tensor(y2,    dtype=torch.long),
                torch.tensor(lam,   dtype=torch.float32))

        self.no_aug_count += 1

        return (
            torch.tensor(x1,  dtype=torch.float32),
            torch.tensor(y1,  dtype=torch.long),
            torch.tensor(y1,  dtype=torch.long),
            torch.tensor(1.0, dtype=torch.float32))


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    cfg,
    device):
    """
    Training loop with early stopping and best-model restoration.

    Optimizer, scheduler, loss and hyperparameters are taken from cfg
    so that EEGNet and lightweightEEGNet can use different training configurations
    without modifying this function.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader — yields (x, y1, y2, lam) batches
    val_loader   : DataLoader — yields (x, y1, y2, lam) batches
    cfg          : dict with keys:
                     epochs, patience, lr, weight_decay,
                     optimizer ('adam'|'adamw'),
                     scheduler ('cosine'|'plateau'),
                     label_smoothing
    device       : torch.device

    Returns
    -------
    preds        : np.ndarray — predicted labels on the validation set
    labels       : np.ndarray — true labels on the validation set
    train_losses : list[float]
    val_losses   : list[float]
    """

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.0))

    if cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"])
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"])
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5)

    best_loss    = float("inf")
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}
    counter      = 0
    train_losses = []
    val_losses   = []


    for epoch in range(cfg["epochs"]):

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.train()
        running_train = 0.0

        for xb, y1, y2, lam in train_loader:

            xb  = xb.to(device)
            y1  = y1.to(device)
            y2  = y2.to(device)
            lam = lam.to(device)

            xb = augment_eeg(xb)

            optimizer.zero_grad()

            out = model(xb)

            loss = (
                lam * criterion(out, y1)
                + (1 - lam) * criterion(out, y2)
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)

            optimizer.step()

            running_train += loss.item()


        train_losses.append(running_train / len(train_loader))


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        model.eval()
        running_val = 0.0

        with torch.no_grad():

            for xb, y1, _, _ in val_loader:

                xb = xb.to(device)
                y1 = y1.to(device)

                out  = model(xb)
                loss = criterion(out, y1)

                running_val += loss.item()


        val_loss = running_val / len(val_loader)
        val_losses.append(val_loss)

        if cfg["scheduler"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)


        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1

        if counter >= cfg["patience"]:
            break


    model.load_state_dict(best_state)


    # -------------------------------------------------
    # FINAL EVALUATION ON VALIDATION SET
    # -------------------------------------------------

    preds  = []
    labels = []

    model.eval()

    with torch.no_grad():

        for xb, y1, _, _ in val_loader:

            xb = xb.to(device)

            out  = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()

            preds.extend(pred)
            labels.extend(y1.numpy())

    return (
        np.array(preds),
        np.array(labels),
        train_losses,
        val_losses)


# =========================================================
# CROSS VALIDATION
# =========================================================

def run_cv(
    X,
    y,
    model_class,
    model_kwargs,
    train_cfg,
    device,
    augment=False,
    n_splits=10,
    seed=42):
    """
    Stratified k-fold cross-validation for one subject.

    Normalization is computed from training data only inside each fold
    to prevent data leakage from validation samples.

    Parameters
    ----------
    X            : np.ndarray, shape (n_trials, n_channels, n_samples)
    y            : np.ndarray, shape (n_trials,), integer labels
    model_class  : class — EEGNet or lightweightEEGNet (from models.py)
    model_kwargs : dict  — passed to model_class(...) at each fold
    train_cfg    : dict  — passed to train_model(...)
    device       : torch.device
    augment      : bool  — frequency mixup on training set (default: False)
    n_splits     : int   — number of CV folds (default: 10)
    seed         : int   — random state for fold assignment (default: 42)

    Returns
    -------
    mean_acc         : float
    conf_total       : np.ndarray, shape (n_classes, n_classes)
    all_train_losses : list[list[float]]
    all_val_losses   : list[list[float]]
    """

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed)

    n_classes        = len(np.unique(y))
    accuracies       = []
    conf_total       = np.zeros((n_classes, n_classes))
    all_train_losses = []
    all_val_losses   = []

    batch_size = train_cfg.get("batch_size", 32)


    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        print(f"\n  Fold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]


        # -------------------------------------------------
        # NORMALIZATION — train statistics only
        # FIX: mean and std are computed from X_train only.
        # The same statistics are applied to X_val to prevent
        # any leakage of validation information.
        # -------------------------------------------------

        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std  = X_train.std( axis=(0, 2), keepdims=True) + 1e-6

        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std


        # -------------------------------------------------
        # DATASETS AND DATALOADERS
        # Aumentation is in EEGDataset (augmentation is in __getitem__).
        # Both produce (x, y1, y2, lam) batches for compatibility
        # with the shared train_model loss computation.
        # -------------------------------------------------

        train_ds = EEGDataset(X_train, y_train, augment=augment)
        val_ds   = EEGDataset(X_val,   y_val,   augment=False)


        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True)

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False)


        # -------------------------------------------------
        # MODEL INSTANTIATION
        # n_samples is passed from the actual data shape so
        # the dummy forward pass in CNN.__init__ uses the real
        # temporal length.
        # -------------------------------------------------

        model = model_class(
            n_samples=X.shape[-1],
            **model_kwargs
        ).to(device)


        preds, labels, tr_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            train_cfg,
            device)

        acc = accuracy_score(labels, preds)
        cm  = confusion_matrix(labels, preds)

        accuracies.append(acc)
        conf_total       += cm
        all_train_losses.append(tr_losses)
        all_val_losses.append(val_losses)

        print(f"  Fold accuracy: {acc:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


    return (
        float(np.mean(accuracies)),
        conf_total,
        all_train_losses,
        all_val_losses)


# =========================================================
# RUN ALL SUBJECTS
# =========================================================

def run_all_subjects(
    folder,
    period_name,
    model_class,
    model_kwargs,
    train_cfg,
    device,
    augment=False):
    """
    Run cross-validation for all subject CSVs in a folder.

    Parameters
    ----------
    folder       : str  — path to folder containing subject CSVs
    period_name  : str  — label for console output ('BSL'/'SENS'/'DELAY')
    model_class  : class
    model_kwargs : dict
    train_cfg    : dict
    device       : torch.device
    augment      : bool

    Returns
    -------
    results : dict mapping subject ID (str) → {
                  'accuracy', 'conf_matrix',
                  'train_losses', 'val_losses' }
    """

    print(f"\n{'='*20} {period_name} {'='*20}")

    results = {}
    count   = 1

    for file in sorted(os.listdir(folder)):

        if not file.endswith(".csv"):
            continue

        subj = file.split("_")[1].split(".")[0]

        print(f"\nSubject {subj}  [{count}/20]")
        count += 1

        X, y_raw = load_subject_csv(os.path.join(folder, file))

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        acc, cm, tr, val = run_cv(
            X, y,
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_cfg=train_cfg,
            device=device,
            augment=augment)

        print(f"Subject {subj} — mean accuracy: {acc:.4f}")
        print(cm)

        results[subj] = {
            "accuracy":     acc,
            "conf_matrix":  cm.tolist(),
            "train_losses": tr,
            "val_losses":   val}

        del X, y
        gc.collect()

    return results


# =========================================================
# SUMMARIZE RESULTS
# =========================================================

def summarize_results(results, period_name):
    """Print mean accuracy across subjects for one experimental period."""

    accs = [r["accuracy"] for r in results.values()]

    print(f"{period_name} — mean accuracy: {np.mean(accs):.4f} "
          f"(std: {np.std(accs):.4f})")


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results, results_dir, period_name):
    """Save results dict to a JSON file in results_dir."""

    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(
        results_dir,
        f"results_{period_name.lower()}.json")

    with open(out_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved → {out_path}")