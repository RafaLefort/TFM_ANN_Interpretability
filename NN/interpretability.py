"""
interpretability.py
===================
Interpretability tools for EEG working memory classification models.

All functions are designed to work with models saved via train.py
(SAVE_MODEL = True), which stores per-fold state dicts inside the
results JSON produced by save_results().

Functions
---------
load_subject_models         : Reconstruct trained models from saved state dicts.
compute_saliency_trial      : Vanilla gradient saliency for a single trial.
compute_saliency_subject    : Aggregate saliency over all folds for one subject.
compute_saliency_period     : Aggregate saliency over all subjects for one period.
plot_saliency_maps          : Plot and save channel × time saliency heatmaps
                                (one panel per class, one figure per period).
run_saliency_analysis       : End-to-end pipeline: load results, compute maps,
                                save arrays and figures for every requested period.

Usage
-----
Configure the EXPERIMENT block in train.py identically, then call:

    from interpretability import run_saliency_analysis
    run_saliency_analysis(
        results_dir  = RESULTS_DIR,
        period_folders = PERIOD_FOLDERS,
        periods      = PERIODS,
        model_class  = MODEL_CLASS,
        model_kwargs = MODEL_KWARGS,
        device       = DEVICE,
        class_names  = ['Verbal', 'Spatial', 'Visual'])
"""

import os
import gc
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from utils import load_subject_csv


# =========================================================
# LOAD SUBJECT MODELS
# =========================================================

def load_subject_models(
    state_dicts,
    model_class,
    model_kwargs,
    n_samples,
    device):
    """
    Reconstruct trained models from a list of saved state dicts.

    Each entry in state_dicts corresponds to one CV fold.  The function
    instantiates a fresh model, loads the weights, and sets it to eval
    mode with gradients enabled for saliency computation.

    Parameters
    ----------
    state_dicts  : list[dict] — per-fold state dicts (from results JSON)
    model_class  : class      — EEGNet or lightweightEEGNet
    model_kwargs : dict       — passed to model_class(...)
    n_samples    : int        — temporal length of the EEG trials
    device       : torch.device

    Returns
    -------
    models : list[nn.Module]  — one loaded model per fold, on device
    """

    models = []

    for sd in state_dicts:

        model = model_class(
            n_samples=n_samples,
            **model_kwargs
        ).to(device)

        # State dicts saved via save_results() store tensors as plain
        # Python lists (JSON-serialized). Re-convert them here.
        tensor_sd = {
            k: torch.tensor(np.array(v, dtype=np.float32)).to(device)
            for k, v in sd.items()}

        model.load_state_dict(tensor_sd)
        model.eval()

        models.append(model)

    return models


# =========================================================
# COMPUTE SALIENCY — SINGLE TRIAL
# =========================================================

def compute_saliency_trial(model, x, target_class, device):
    """
    Vanilla gradient saliency for a single EEG trial.

    Computes ∂logit[target_class] / ∂x, then takes the absolute value
    to obtain a non-negative importance map of shape (n_channels, n_samples).

    Parameters
    ----------
    model        : nn.Module — trained model in eval mode
    x            : np.ndarray, shape (n_channels, n_samples), float32
    target_class : int — class index for which to differentiate
    device       : torch.device

    Returns
    -------
    saliency : np.ndarray, shape (n_channels, n_samples), float32
    """

    xb = torch.tensor(x[np.newaxis], dtype=torch.float32,
                      requires_grad=True, device=device)

    out  = model(xb)
    score = out[0, target_class]
    score.backward()

    saliency = xb.grad.detach().cpu().numpy()[0]

    return np.abs(saliency).astype(np.float32)


# =========================================================
# COMPUTE SALIENCY — ONE SUBJECT, ALL FOLDS
# =========================================================

def compute_saliency_subject(
    models,
    X,
    y,
    n_classes,
    n_splits=10,
    seed=42):
    """
    Aggregate saliency over all CV folds for one subject.

    The fold splits are re-created with the same StratifiedKFold
    parameters as run_cv() so that each fold's model is evaluated
    on its own held-out validation set.  Normalization statistics
    are recomputed from training data only, matching run_cv() exactly.

    Only correctly-classified trials contribute to the aggregation
    to ensure the maps reflect the model's actual discriminative signal.

    Parameters
    ----------
    models    : list[nn.Module] — one loaded model per fold
    X         : np.ndarray, shape (n_trials, n_channels, n_samples)
    y         : np.ndarray, shape (n_trials,), integer labels
    n_classes : int
    n_splits  : int — must match the value used in run_cv() (default: 10)
    seed      : int — must match the value used in run_cv() (default: 42)

    Returns
    -------
    saliency_per_class : np.ndarray, shape (n_classes, n_channels, n_samples)
                         Mean absolute gradient over correctly classified trials,
                         averaged across folds.  NaN where no correct trials exist.
    counts_per_class   : np.ndarray, shape (n_classes,)
                         Total number of correctly classified trials per class.
    """

    device = next(models[0].parameters()).device

    n_channels = X.shape[1]
    n_samples  = X.shape[2]

    # Accumulators: sum of saliency maps and trial counts, per class
    sal_sum = np.zeros((n_classes, n_channels, n_samples), dtype=np.float64)
    counts  = np.zeros(n_classes, dtype=np.int64)

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        model = models[fold_idx]

        X_train = X[train_idx]
        X_val   = X[val_idx]
        y_val   = y[val_idx]


        # -------------------------------------------------
        # NORMALIZATION — identical to run_cv()
        # -------------------------------------------------

        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std  = X_train.std( axis=(0, 2), keepdims=True) + 1e-6

        X_val_norm = (X_val - mean) / std


        # -------------------------------------------------
        # SALIENCY ACCUMULATION — correctly classified only
        # -------------------------------------------------

        for i, (x, true_label) in enumerate(zip(X_val_norm, y_val)):

            with torch.no_grad():
                xb  = torch.tensor(x[np.newaxis], dtype=torch.float32,
                                   device=device)
                out  = model(xb)
                pred = int(torch.argmax(out, dim=1).cpu())

            if pred != true_label:
                continue

            sal = compute_saliency_trial(model, x, true_label, device)

            sal_sum[true_label] += sal
            counts[true_label]  += 1


    # -------------------------------------------------
    # AVERAGE — protect against empty classes
    # -------------------------------------------------

    saliency_per_class = np.where(
        counts[:, np.newaxis, np.newaxis] > 0,
        sal_sum / np.maximum(counts[:, np.newaxis, np.newaxis], 1),
        np.nan)

    return saliency_per_class.astype(np.float32), counts


# =========================================================
# COMPUTE SALIENCY — ALL SUBJECTS FOR ONE PERIOD
# =========================================================

def compute_saliency_period(
    results,
    folder,
    model_class,
    model_kwargs,
    device,
    n_classes=3,
    n_splits=10,
    seed=42):
    """
    Aggregate per-class saliency maps across all subjects for one period.

    For each subject the per-fold models are loaded, saliency is computed
    on the held-out validation splits, and the results are averaged
    weighted by the number of correctly classified trials.

    Parameters
    ----------
    results      : dict — subject results from load_period_results()
                          keys: subject IDs, values must contain 'models'
    folder       : str  — path to the subject CSV folder
    model_class  : class
    model_kwargs : dict
    device       : torch.device
    n_classes    : int  (default: 3)
    n_splits     : int  (default: 10)
    seed         : int  (default: 42)

    Returns
    -------
    period_saliency : np.ndarray, shape (n_classes, n_channels, n_samples)
                      Weighted mean saliency across all subjects and folds.
    """

    # Weighted-sum accumulators across subjects
    sal_sum_global = None
    counts_global  = np.zeros(n_classes, dtype=np.float64)

    for file in sorted(os.listdir(folder)):

        if not file.endswith(".csv"):
            continue

        subj = file.split("_")[1].split(".")[0]

        if subj not in results:
            print(f"  [skip] subject {subj} not found in results dict")
            continue

        subj_data = results[subj]

        if subj_data.get("models") is None:
            print(f"  [skip] subject {subj} has no saved models "
                  f"(re-run train.py with SAVE_MODEL = True)")
            continue

        print(f"  Computing saliency — subject {subj}")

        X, y_raw = load_subject_csv(os.path.join(folder, file))

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        models = load_subject_models(
            state_dicts  = subj_data["models"],
            model_class  = model_class,
            model_kwargs = model_kwargs,
            n_samples    = X.shape[-1],
            device       = device)

        sal, counts = compute_saliency_subject(
            models    = models,
            X         = X,
            y         = y,
            n_classes = n_classes,
            n_splits  = n_splits,
            seed      = seed)

        # -------------------------------------------------
        # WEIGHTED ACCUMULATION
        # sal is (n_classes, C, T); weight each class map by
        # how many correctly classified trials it came from.
        # -------------------------------------------------

        if sal_sum_global is None:
            sal_sum_global = np.zeros_like(sal, dtype=np.float64)

        for c in range(n_classes):
            if not np.isnan(sal[c]).all():
                sal_sum_global[c] += sal[c] * counts[c]
                counts_global[c]  += counts[c]

        del models, X, y
        gc.collect()
        torch.cuda.empty_cache()

    if sal_sum_global is None:
        raise RuntimeError("No subjects were processed. "
                           "Check folder and results dict.")

    period_saliency = np.where(
        counts_global[:, np.newaxis, np.newaxis] > 0,
        sal_sum_global / np.maximum(counts_global[:, np.newaxis, np.newaxis], 1),
        np.nan)

    return period_saliency.astype(np.float32)


# =========================================================
# PLOT SALIENCY MAPS
# =========================================================

def plot_saliency_maps(
    saliency,
    period_name,
    results_dir,
    class_names=None,
    channel_names=None):
    """
    Plot and save channel × time saliency heatmaps for one period.

    One subplot per class is arranged horizontally.  The colour scale
    is normalised per class so within-class spatial and temporal patterns
    are visually comparable across plots.

    Parameters
    ----------
    saliency      : np.ndarray, shape (n_classes, n_channels, n_samples)
    period_name   : str  — used in the figure title and filename
    results_dir   : str  — output directory (created if absent)
    class_names   : list[str] | None — default: ['Class 0', 'Class 1', ...]
    channel_names : list[str] | None — y-axis tick labels; omitted if None
                    (64 channel names would be unreadable at default fig size)

    Saves
    -----
    <results_dir>/saliency_<period_name.lower()>.png
    """

    n_classes  = saliency.shape[0]
    n_channels = saliency.shape[1]
    n_samples  = saliency.shape[2]

    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    fig = plt.figure(figsize=(6 * n_classes, 5))
    fig.suptitle(
        f"Saliency Maps — {period_name}",
        fontsize=14, fontweight="bold", y=1.02)

    gs = gridspec.GridSpec(
        1, n_classes + 1,
        width_ratios=[1] * n_classes + [0.05],
        wspace=0.35)

    # Shared colour range: 0 → global max across all classes
    vmax = np.nanmax(saliency)
    vmin = 0.0

    axes = []

    for c in range(n_classes):

        ax = fig.add_subplot(gs[0, c])
        axes.append(ax)

        sal_c = saliency[c]

        im = ax.imshow(
            sal_c,
            aspect="auto",
            origin="upper",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest")

        ax.set_title(class_names[c], fontsize=11)
        ax.set_xlabel("Time sample", fontsize=9)

        if c == 0:
            ax.set_ylabel("Channel", fontsize=9)

            if channel_names is not None:
                ax.set_yticks(range(n_channels))
                ax.set_yticklabels(channel_names, fontsize=6)
            else:
                ax.set_yticks([0, n_channels // 2, n_channels - 1])
                ax.set_yticklabels(
                    [0, n_channels // 2, n_channels - 1],
                    fontsize=8)
        else:
            ax.set_yticks([])

        # X-axis: a few evenly spaced ticks
        tick_positions = np.linspace(0, n_samples - 1, 5, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=8)

    # Shared colourbar
    cax = fig.add_subplot(gs[0, n_classes])
    plt.colorbar(im, cax=cax, label="|gradient|")

    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(
        results_dir,
        f"saliency_{period_name.lower()}.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saliency figure saved → {out_path}")


# =========================================================
# SAVE SALIENCY ARRAYS
# =========================================================

def save_saliency(saliency, results_dir, period_name):
    """
    Save a saliency array to disk as a compressed NumPy archive.

    Parameters
    ----------
    saliency     : np.ndarray, shape (n_classes, n_channels, n_samples)
    results_dir  : str
    period_name  : str

    Saves
    -----
    <results_dir>/saliency_<period_name.lower()>.npz
    """

    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(
        results_dir,
        f"saliency_{period_name.lower()}.npz")

    np.savez_compressed(out_path, saliency=saliency)

    print(f"Saliency array  saved → {out_path}")


# =========================================================
# LOAD PERIOD RESULTS
# =========================================================

def load_period_results(results_dir, period_name):
    """
    Load the JSON results file produced by save_results() for one period.

    Parameters
    ----------
    results_dir : str
    period_name : str — e.g. 'BSL', 'SENS', 'DELAY'

    Returns
    -------
    results : dict — subject ID → result dict (including 'models' key)
    """

    path = os.path.join(
        results_dir,
        f"results_{period_name.lower()}.json")

    with open(path, "r") as f:
        results = json.load(f)

    return results


# =========================================================
# RUN SALIENCY ANALYSIS  — end-to-end pipeline
# =========================================================

def run_saliency_analysis(
    results_dir,
    period_folders,
    periods,
    model_class,
    model_kwargs,
    device,
    class_names=None,
    channel_names=None,
    n_classes=3,
    n_splits=10,
    seed=42):
    """
    End-to-end saliency pipeline: load results, compute maps, save outputs.

    For each period in `periods`:
      1. Load the results JSON from results_dir.
      2. Aggregate saliency over all subjects and folds.
      3. Save the saliency array as a compressed .npz file.
      4. Save a channel × time heatmap figure as a .png file.

    Both outputs land in results_dir alongside the existing JSON files.

    Parameters
    ----------
    results_dir    : str  — directory produced by train.py (RESULTS_DIR)
    period_folders : dict — maps period name → CSV folder path
                            (same PERIOD_FOLDERS dict used in train.py)
    periods        : list[str] — subset of ['BSL', 'SENS', 'DELAY']
    model_class    : class — EEGNet or lightweightEEGNet
    model_kwargs   : dict  — passed to model_class(...)
    device         : torch.device
    class_names    : list[str] | None — stimulus labels for plot titles
    channel_names  : list[str] | None — EEG channel labels for y-axis
    n_classes      : int  (default: 3)
    n_splits       : int  (default: 10)
    seed           : int  (default: 42)

    Returns
    -------
    all_saliency : dict — maps period name → np.ndarray
                          shape (n_classes, n_channels, n_samples)
    """

    all_saliency = {}

    for period in periods:

        print(f"\n{'='*20} SALIENCY — {period} {'='*20}")

        results = load_period_results(results_dir, period)

        saliency = compute_saliency_period(
            results      = results,
            folder       = period_folders[period],
            model_class  = model_class,
            model_kwargs = model_kwargs,
            device       = device,
            n_classes    = n_classes,
            n_splits     = n_splits,
            seed         = seed)

        save_saliency(saliency, results_dir, period)

        plot_saliency_maps(
            saliency      = saliency,
            period_name   = period,
            results_dir   = results_dir,
            class_names   = class_names,
            channel_names = channel_names)

        all_saliency[period] = saliency

    print("\nSaliency analysis complete.")

    return all_saliency