"""
interpretability.py
=======================
Stand-alone interpretability pipeline for EEG working memory classification.

Run this script after train.py has completed with SAVE_MODEL = True.
It loads the saved per-fold model weights from the results JSON files
and runs all requested interpretability analyses without re-training.

Usage
-----
Configure the EXPERIMENT block below (must mirror train.py exactly),
then run:

    python interpretability.py

Analyses
--------
SALIENCY_MAP  : Vanilla gradient saliency, aggregated across all subjects
                and folds for each period.  Produces a channel × time
                heatmap (n_classes panels) per period.
                Implemented in interpretability.py → run_saliency_analysis().

GRADCAM       : Gradient-weighted Class Activation Mapping focused on
                channel relevance.  Hooks the spatial depthwise convolution
                in both architectures — the layer that explicitly learns to
                weight EEG channels — and computes a (n_channels,) importance
                score per class per period.
                Implemented below → run_gradcam_analysis().
"""

import os
import gc
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from models import EEGNet, lightweightEEGNet
from utils import load_subject_csv
from interpretability_utils import (
    load_subject_models,
    run_saliency_analysis)


# =========================================================
# EXPERIMENT — mirror train.py exactly
# =========================================================

MODEL   = "lightweightEEGNet"       # 'EEGNet' | 'lightweightEEGNet'
AUGMENT = False                     # must match train.py
PERIODS = ['BSL', 'SENS', 'DELAY']  # any subset of ['BSL', 'SENS', 'DELAY']

CLASS_NAMES   = ['Verbal', 'Spatial', 'Visual']
CHANNEL_NAMES = None  # list[str] of length 64, or None to use indices


# =========================================================
# ANALYSES TO RUN
# =========================================================

SALIENCY_MAP = True   # vanilla gradient saliency (channel × time heatmap)
GRADCAM      = True   # GradCAM channel relevance bar charts


# =========================================================
# PATHS — mirror train.py exactly
# =========================================================

BASE_PATH = "NN/results/"

PERIOD_FOLDERS = {
    "BSL":   os.path.join("", "BSL_subjects"),
    "SENS":  os.path.join("", "SENS_subjects"),
    "DELAY": os.path.join("", "DELAY_subjects"),
}

_aug_tag    = "_data_augmentation" if AUGMENT else ""
_model_tag  = "lightweightEEGNet" if MODEL == "lightweightEEGNet" else "EEGNet"
RESULTS_DIR = os.path.join(BASE_PATH, f"{_model_tag}{_aug_tag}")


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice  : {DEVICE}")
print(f"Model   : {MODEL.upper()}")
print(f"Periods : {PERIODS}")
print(f"Results : {RESULTS_DIR}")
print(f"Saliency: {SALIENCY_MAP}")
print(f"GradCAM : {GRADCAM}\n")


# =========================================================
# MODEL CONFIG — mirror train.py exactly
# =========================================================

if MODEL == "EEGNet":

    MODEL_CLASS  = EEGNet
    MODEL_KWARGS = {
        "n_channels":   64,
        "n_classes":    3,
        "dropout_rate": 0.5,
    }

elif MODEL == "lightweightEEGNet":

    MODEL_CLASS  = lightweightEEGNet
    MODEL_KWARGS = {
        "n_channels": 64,
        "n_classes":  3,
    }

else:
    raise ValueError(f"Unknown model '{MODEL}'. "
                     f"Choose 'EEGNet' or 'lightweightEEGNet'.")


# =========================================================
# GRADCAM — TARGET LAYER RESOLUTION
# =========================================================
#
# Channel relevance requires hooking the spatial depthwise convolution,
# which is the layer that explicitly learns to weight EEG channels
# (kernel shape: (n_channels, 1)).  Its *input* feature map has shape
# (B, F_in, n_channels, T), so gradients w.r.t. that tensor carry
# per-channel importance information before the channel axis is collapsed.
#
# Layer locations per architecture:
#   EEGNet            → model.block1[2]   (nn.Conv2d, groups=F1)
#   lightweightEEGNet → model.spatial_conv (nn.Conv2d, groups=6)
#
# GradCAM formula applied here:
#   For each trial and target class c:
#     1. Forward pass → logit[c]
#     2. Backward pass → grad of logit[c] w.r.t. spatial conv INPUT
#        shape: (1, F_in, n_channels, T)
#     3. Global-average-pool over filter (dim 1) and time (dim 3)
#        → weight vector of shape (n_channels,)
#     4. ReLU — keep only positively contributing channels
#   Aggregate over correctly-classified trials, then across folds
#   and subjects with trial-count weighting.
#
# =========================================================

def _get_spatial_conv(model):
    """
    Return the spatial depthwise Conv2d for the given model instance.

    This is the layer whose input gradients encode channel relevance.

    Parameters
    ----------
    model : nn.Module — EEGNet or lightweightEEGNet instance

    Returns
    -------
    layer : nn.Conv2d
    """

    if isinstance(model, EEGNet):
        # block1 is nn.Sequential; index 2 is the depthwise spatial conv
        return model.block1[2]

    elif isinstance(model, lightweightEEGNet):
        return model.spatial_conv

    else:
        raise TypeError(
            f"Unsupported model type '{type(model).__name__}'. "
            f"Add its spatial conv location to _get_spatial_conv().")


# =========================================================
# GRADCAM — SINGLE TRIAL
# =========================================================

def compute_gradcam_trial(model, x, target_class, device):
    """
    GradCAM channel relevance for a single EEG trial.

    Hooks the input gradient of the spatial depthwise convolution,
    global-average-pools over filter and time dimensions, and applies
    ReLU to retain only positively contributing channels.

    Parameters
    ----------
    model        : nn.Module — trained model in eval mode
    x            : np.ndarray, shape (n_channels, n_samples), float32
    target_class : int
    device       : torch.device

    Returns
    -------
    cam : np.ndarray, shape (n_channels,), float32 — channel importance ≥ 0
    """

    spatial_conv = _get_spatial_conv(model)

    # Storage for the hook
    grad_input = {}

    def _hook(module, grad_in, grad_out):
        # grad_in[0]: gradient w.r.t. the layer's input tensor
        # shape: (B, F_in, n_channels, T)
        grad_input["value"] = grad_in[0]

    handle = spatial_conv.register_full_backward_hook(_hook)

    xb = torch.tensor(
        x[np.newaxis], dtype=torch.float32,
        requires_grad=True, device=device)

    out   = model(xb)
    score = out[0, target_class]
    score.backward()

    handle.remove()

    # grad shape: (1, F_in, n_channels, T)
    grad = grad_input["value"].detach().cpu().numpy()[0]

    # Pool over filter (axis 0) and time (axis 2) → (n_channels,)
    weights = grad.mean(axis=(0, 2))

    # ReLU: negative weights indicate suppression, not relevance
    cam = np.maximum(weights, 0).astype(np.float32)

    return cam


# =========================================================
# GRADCAM — ONE SUBJECT, ALL FOLDS
# =========================================================

def compute_gradcam_subject(
    models,
    X,
    y,
    n_classes,
    n_splits=10,
    seed=42):
    """
    Aggregate GradCAM channel relevance over all CV folds for one subject.

    Fold splits and normalization are reproduced identically to run_cv()
    to ensure each fold model is evaluated on its own held-out data.
    Only correctly-classified trials contribute to the accumulation.

    Parameters
    ----------
    models    : list[nn.Module]
    X         : np.ndarray, shape (n_trials, n_channels, n_samples)
    y         : np.ndarray, shape (n_trials,)
    n_classes : int
    n_splits  : int (default: 10)
    seed      : int (default: 42)

    Returns
    -------
    cam_per_class : np.ndarray, shape (n_classes, n_channels), float32
                    Mean channel importance per class; NaN where no correct
                    trials exist.
    counts        : np.ndarray, shape (n_classes,), int64
                    Number of correctly classified trials per class.
    """

    device     = next(models[0].parameters()).device
    n_channels = X.shape[1]

    cam_sum = np.zeros((n_classes, n_channels), dtype=np.float64)
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
        # GRADCAM ACCUMULATION — correctly classified only
        # -------------------------------------------------

        for x, true_label in zip(X_val_norm, y_val):

            with torch.no_grad():
                xb   = torch.tensor(x[np.newaxis], dtype=torch.float32,
                                    device=device)
                out  = model(xb)
                pred = int(torch.argmax(out, dim=1).cpu())

            if pred != true_label:
                continue

            cam = compute_gradcam_trial(model, x, true_label, device)

            cam_sum[true_label] += cam
            counts[true_label]  += 1

    cam_per_class = np.where(
        counts[:, np.newaxis] > 0,
        cam_sum / np.maximum(counts[:, np.newaxis], 1),
        np.nan)

    return cam_per_class.astype(np.float32), counts


# =========================================================
# GRADCAM — ALL SUBJECTS FOR ONE PERIOD
# =========================================================

def compute_gradcam_period(
    results,
    folder,
    model_class,
    model_kwargs,
    device,
    n_classes=3,
    n_splits=10,
    seed=42):
    """
    Aggregate GradCAM channel relevance across all subjects for one period.

    Parameters
    ----------
    results      : dict — subject results loaded from JSON
    folder       : str  — path to subject CSV folder
    model_class  : class
    model_kwargs : dict
    device       : torch.device
    n_classes    : int  (default: 3)
    n_splits     : int  (default: 10)
    seed         : int  (default: 42)

    Returns
    -------
    period_cam : np.ndarray, shape (n_classes, n_channels), float32
                 Trial-count-weighted mean channel importance across all subjects.
    """

    cam_sum_global = None
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

        print(f"  Computing GradCAM — subject {subj}")

        X, y_raw = load_subject_csv(os.path.join(folder, file))

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        models = load_subject_models(
            state_dicts  = subj_data["models"],
            model_class  = model_class,
            model_kwargs = model_kwargs,
            n_samples    = X.shape[-1],
            device       = device)

        cam, counts = compute_gradcam_subject(
            models    = models,
            X         = X,
            y         = y,
            n_classes = n_classes,
            n_splits  = n_splits,
            seed      = seed)

        if cam_sum_global is None:
            cam_sum_global = np.zeros_like(cam, dtype=np.float64)

        for c in range(n_classes):
            if not np.isnan(cam[c]).all():
                cam_sum_global[c] += cam[c] * counts[c]
                counts_global[c]  += counts[c]

        del models, X, y
        gc.collect()
        torch.cuda.empty_cache()

    if cam_sum_global is None:
        raise RuntimeError(
            "No subjects were processed. Check folder and results dict.")

    period_cam = np.where(
        counts_global[:, np.newaxis] > 0,
        cam_sum_global / np.maximum(counts_global[:, np.newaxis], 1),
        np.nan)

    return period_cam.astype(np.float32)


# =========================================================
# PLOT GRADCAM — CHANNEL RELEVANCE BAR CHARTS
# =========================================================

def plot_gradcam_channel_relevance(
    cam,
    period_name,
    results_dir,
    class_names=None,
    channel_names=None,
    top_k=20):
    """
    Plot and save GradCAM channel relevance bar charts for one period.

    Layout: one row per class, showing the top_k most relevant channels
    as a horizontal bar chart.  A second figure shows all channels as a
    heatmap (classes × channels) for cross-class comparison.

    Parameters
    ----------
    cam           : np.ndarray, shape (n_classes, n_channels)
    period_name   : str
    results_dir   : str
    class_names   : list[str] | None
    channel_names : list[str] | None — if None, channel indices are used
    top_k         : int — number of top channels shown in bar charts (default: 20)

    Saves
    -----
    <results_dir>/gradcam_bars_<period>.png   — top-k bar charts per class
    <results_dir>/gradcam_heatmap_<period>.png — full heatmap (classes × channels)
    """

    n_classes  = cam.shape[0]
    n_channels = cam.shape[1]
    period_lo  = period_name.lower()

    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    ch_labels = (
        channel_names
        if channel_names is not None
        else [str(i) for i in range(n_channels)])

    os.makedirs(results_dir, exist_ok=True)


    # -------------------------------------------------
    # FIGURE 1 — Top-k horizontal bar charts, one per class
    # -------------------------------------------------

    fig, axes = plt.subplots(
        n_classes, 1,
        figsize=(10, 3.5 * n_classes),
        constrained_layout=True)

    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        f"GradCAM Channel Relevance — {period_name}\n"
        f"(top {top_k} channels, spatial conv input gradients)",
        fontsize=13, fontweight="bold")

    # Normalise across all classes so bars are directly comparable
    global_max = np.nanmax(cam)

    for c, ax in enumerate(axes):

        scores = cam[c].copy()
        scores[np.isnan(scores)] = 0.0

        # Sort descending, take top_k
        order   = np.argsort(scores)[::-1][:top_k]
        top_ch  = [ch_labels[i] for i in order]
        top_sc  = scores[order] / (global_max + 1e-9)   # normalised 0–1

        colours = plt.cm.YlOrRd(top_sc)

        bars = ax.barh(
            range(top_k),
            top_sc,
            color=colours,
            edgecolor="none",
            height=0.7)

        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_ch, fontsize=8)
        ax.invert_yaxis()                      # highest relevance at top
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalised relevance", fontsize=9)
        ax.set_title(class_names[c], fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate raw scores on bars
        for bar, sc in zip(bars, top_sc):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sc:.3f}",
                va="center", ha="left", fontsize=7)

    bars_path = os.path.join(results_dir, f"gradcam_bars_{period_lo}.png")
    fig.savefig(bars_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"GradCAM bar chart saved → {bars_path}")


    # -------------------------------------------------
    # FIGURE 2 — Full heatmap: classes × channels
    # Useful for spotting which channels are class-specific
    # vs. universally relevant.
    # -------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(max(12, n_channels * 0.18), 3.5),
        constrained_layout=True)

    fig2.suptitle(
        f"GradCAM Channel Relevance Heatmap — {period_name}",
        fontsize=13, fontweight="bold")

    cam_norm = cam / (global_max + 1e-9)

    im = ax2.imshow(
        cam_norm,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        interpolation="nearest")

    ax2.set_yticks(range(n_classes))
    ax2.set_yticklabels(class_names, fontsize=10)

    tick_step = max(1, n_channels // 32)
    tick_pos  = list(range(0, n_channels, tick_step))

    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(
        [ch_labels[i] for i in tick_pos],
        fontsize=7,
        rotation=45,
        ha="right")

    ax2.set_xlabel("Channel", fontsize=10)

    cbar = plt.colorbar(im, ax=ax2, fraction=0.015, pad=0.01)
    cbar.set_label("Normalised relevance", fontsize=9)

    heatmap_path = os.path.join(
        results_dir, f"gradcam_heatmap_{period_lo}.png")

    fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"GradCAM heatmap     saved → {heatmap_path}")


# =========================================================
# SAVE GRADCAM ARRAYS
# =========================================================

def save_gradcam(cam, results_dir, period_name):
    """
    Save a GradCAM array to disk as a compressed NumPy archive.

    Parameters
    ----------
    cam          : np.ndarray, shape (n_classes, n_channels)
    results_dir  : str
    period_name  : str

    Saves
    -----
    <results_dir>/gradcam_<period_name.lower()>.npz
    """

    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(
        results_dir, f"gradcam_{period_name.lower()}.npz")

    np.savez_compressed(out_path, cam=cam)

    print(f"GradCAM array  saved → {out_path}")


# =========================================================
# RUN GRADCAM ANALYSIS — end-to-end pipeline
# =========================================================

def run_gradcam_analysis(
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
    seed=42,
    top_k=20):
    """
    End-to-end GradCAM pipeline: load results, compute channel relevance,
    save arrays and figures for every requested period.

    Parameters
    ----------
    results_dir    : str
    period_folders : dict  — maps period name → CSV folder path
    periods        : list[str]
    model_class    : class
    model_kwargs   : dict
    device         : torch.device
    class_names    : list[str] | None
    channel_names  : list[str] | None
    n_classes      : int  (default: 3)
    n_splits       : int  (default: 10)
    seed           : int  (default: 42)
    top_k          : int  — channels shown in bar charts (default: 20)

    Returns
    -------
    all_cam : dict — maps period name → np.ndarray (n_classes, n_channels)
    """

    all_cam = {}

    for period in periods:

        print(f"\n{'='*20} GRADCAM — {period} {'='*20}")

        results_path = os.path.join(
            results_dir, f"results_{period.lower()}.json")

        with open(results_path, "r") as f:
            results = json.load(f)

        cam = compute_gradcam_period(
            results      = results,
            folder       = period_folders[period],
            model_class  = model_class,
            model_kwargs = model_kwargs,
            device       = device,
            n_classes    = n_classes,
            n_splits     = n_splits,
            seed         = seed)

        save_gradcam(cam, results_dir, period)

        plot_gradcam_channel_relevance(
            cam           = cam,
            period_name   = period,
            results_dir   = results_dir,
            class_names   = class_names,
            channel_names = channel_names,
            top_k         = top_k)

        all_cam[period] = cam

    print("\nGradCAM analysis complete.")

    return all_cam


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -------------------------------------------------
    # SALIENCY MAP
    # -------------------------------------------------

    if SALIENCY_MAP:

        run_saliency_analysis(
            results_dir    = RESULTS_DIR,
            period_folders = PERIOD_FOLDERS,
            periods        = PERIODS,
            model_class    = MODEL_CLASS,
            model_kwargs   = MODEL_KWARGS,
            device         = DEVICE,
            class_names    = CLASS_NAMES,
            channel_names  = CHANNEL_NAMES)

    # -------------------------------------------------
    # GRADCAM
    # -------------------------------------------------

    if GRADCAM:

        run_gradcam_analysis(
            results_dir    = RESULTS_DIR,
            period_folders = PERIOD_FOLDERS,
            periods        = PERIODS,
            model_class    = MODEL_CLASS,
            model_kwargs   = MODEL_KWARGS,
            device         = DEVICE,
            class_names    = CLASS_NAMES,
            channel_names  = CHANNEL_NAMES,
            top_k          = 20)