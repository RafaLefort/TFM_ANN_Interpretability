"""
interpretability_utils.py
===========================
Interpretability tools for EEG working memory classification models.

All functions are designed to work with models saved via train.py
(SAVE_MODEL = True), which stores per-fold state dicts inside the
results JSON produced by save_results().

This module holds every reusable, callable interpretability function.
interpretability.py itself is now just a thin configuration + runner
script that imports from here — it contains no analysis logic of its own.

Functions
---------
load_subject_models         : Reconstruct trained models from saved state dicts.
compute_saliency_trial      : Vanilla gradient saliency for a single trial.
compute_saliency_subject    : Aggregate saliency over all folds for one subject.
compute_saliency_period     : Aggregate saliency over all subjects for one period.
plot_saliency_maps          : Plot and save channel × time saliency heatmaps
                                (one panel per class, one figure per period).
plot_saliency_maps_by_stimulus : Saliency heatmap across the full stimulus
                                timeline (-300 ms → 2000 ms), one figure per class.
save_saliency                : Save saliency array as compressed .npz.
load_period_results          : Load a period results JSON produced by save_results().
run_saliency_analysis        : End-to-end pipeline: load results, compute maps,
                                save arrays and figures for every requested period.

GradCAM (channel relevance)
----------------------------
compute_gradcam_trial       : GradCAM channel relevance for a single trial.
compute_gradcam_subject     : Aggregate GradCAM over all folds for one subject.
compute_gradcam_period      : Aggregate GradCAM over all subjects for one period.
plot_gradcam_channel_relevance : Scalp topography (or heatmap fallback) per period.
save_gradcam                 : Save GradCAM array as compressed .npz.
run_gradcam_analysis         : End-to-end pipeline for all requested periods.

Topomap (GradCAM on scalp surface)
------------------------------------
plot_topomap_gradcam        : Plot GradCAM channel relevance on the 10-20 montage.
run_topomap_analysis         : End-to-end pipeline for all requested periods.

GLM comparison
--------------
load_glm_weights            : Load LASSO coefficient CSVs exported from R.
compare_cnn_glm              : Plot CNN GradCAM vs LASSO |weights| scalp
                                topographies side-by-side, one row per stimulus
                                class.
run_glm_comparison           : End-to-end pipeline for all requested periods.

Usage
-----
Configure the EXPERIMENT block in train.py identically, then call:

    from interpretability_utils import run_saliency_analysis
    run_saliency_analysis(
        results_dir  = RESULTS_DIR,
        period_folders = PERIOD_FOLDERS,
        periods      = PERIODS,
        model_class  = MODEL_CLASS,
        model_kwargs = MODEL_KWARGS,
        device       = DEVICE,
        class_names  = ['Verbal', 'Spatial', 'Visual'])

R export (add to GLM_model_training_and_results.R)
---------------------------------------------------
Run the snippet below after fitting to export LASSO weights to CSV.
The Python loader expects one file per period named:
    lasso_weights_bsl.csv / lasso_weights_sens.csv / lasso_weights_delay.csv

    export_lasso_weights <- function(model_list, df_proc, period_name,
                                     out_dir = "GLM/lasso_weights") {
      dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
      subjects <- unique(df_proc$subjectID)
      rows <- list()
      for (i in seq_along(subjects)) {
        model  <- model_list[[i]]
        coeffs <- coef(model, s = "lambda.min")   # named list: one matrix per class
        for (cls in names(coeffs)) {
          beta <- as.numeric(coeffs[[cls]])[-1]   # drop intercept (index 1)
          rows[[length(rows) + 1]] <- c(
            subjectID = subjects[i],
            class     = cls,
            setNames(beta, paste0("chn", seq_along(beta))))
        }
      }
      df_out <- do.call(rbind, lapply(rows, function(r) as.data.frame(t(r),
                        stringsAsFactors = FALSE)))
      write.csv(df_out,
                file = file.path(out_dir,
                                 paste0("lasso_weights_",
                                        tolower(period_name), ".csv")),
                row.names = FALSE)
    }

    export_lasso_weights(models_BSL,   BSL_proc,   "BSL")
    export_lasso_weights(models_SENS,  SENS_proc,  "SENS")
    export_lasso_weights(models_DELAY, DELAY_proc, "DELAY")
"""

import os
import gc
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import mne

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from utils import load_subject_csv
from models import EEGNet, lightweightEEGNet


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
            cmap="jet",
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


def plot_saliency_maps_by_stimulus(
    all_saliency,
    results_dir,
    class_names=None,
    channel_names=None,
    period_n_samples=None,
    sfreq=None,
    ms_per_sample=2.0):
    """
    Plot saliency maps arranged along the full stimulus timeline
    (-300 ms → 2000 ms), with one figure per class.

    The three periods are concatenated along the time axis and displayed
    as a single heatmap.  A vertical dashed line separates each period.
    The X-axis shows time in milliseconds relative to stimulus onset (0 ms),
    computed as  t_ms = sample_index * ms_per_sample + period_start_ms,
    where ms_per_sample = 2.0 (500 Hz sampling rate, 1 sample = 2 ms).

    Period time boundaries (ms):
        BSL   : -300  →    0
        SENS  :    0  → 1000
        DELAY : 1000  → 2000

    Parameters
    ----------
    all_saliency  : dict — maps period name ('BSL'|'SENS'|'DELAY') →
                          np.ndarray of shape (n_classes, n_channels, n_samples)
    results_dir   : str  — output directory (created if absent)
    class_names   : list[str] | None
    channel_names : list[str] | None — y-axis channel labels (64 names)
    period_n_samples : unused, kept for API compatibility
    sfreq         : unused, kept for API compatibility
    ms_per_sample : float — milliseconds per EEG sample (default: 2.0 → 500 Hz)

    Saves (one file per class)
    --------------------------
    <results_dir>/saliency_stimulus_<class_name.lower()>.png
    """

    # -------------------------------------------------------
    # Period order and start times in ms
    # -------------------------------------------------------
    PERIOD_ORDER      = ['BSL', 'SENS', 'DELAY']
    PERIOD_START_MS   = {'BSL': -300.0, 'SENS': 0.0, 'DELAY': 1000.0}
    PERIOD_BOUNDARY_MS = {'BSL': 0.0, 'SENS': 1000.0}   # inner boundary ms values

    periods_present = [p for p in PERIOD_ORDER if p in all_saliency]

    if not periods_present:
        print("[plot_saliency_maps_by_stimulus] No saliency data provided.")
        return

    first      = all_saliency[periods_present[0]]
    n_classes  = first.shape[0]
    n_channels = first.shape[1]

    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    # -------------------------------------------------------
    # Concatenate along time axis; build exact ms axis using
    # ms_per_sample so every sample lands on the correct ms value.
    # -------------------------------------------------------
    concat_parts             = []
    period_sample_boundaries = [0]
    ms_segments              = []

    for p in periods_present:
        sal   = all_saliency[p]            # (n_classes, n_channels, T)
        T_p   = sal.shape[2]
        t0_ms = PERIOD_START_MS[p]

        # exact ms for each sample in this period
        ms_seg = t0_ms + np.arange(T_p) * ms_per_sample
        ms_segments.append(ms_seg)

        concat_parts.append(sal)
        period_sample_boundaries.append(period_sample_boundaries[-1] + T_p)

    sal_concat = np.concatenate(concat_parts, axis=2)   # (n_classes, n_ch, T_total)
    ms_axis    = np.concatenate(ms_segments)              # (T_total,)
    T_total    = sal_concat.shape[2]

    # -------------------------------------------------------
    # Colour scale: 0 → global max across all classes/periods
    # -------------------------------------------------------
    vmax = float(np.nanmax(sal_concat))
    vmin = 0.0

    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------
    # Figure height: enough px per channel that 64 channel labels
    # at a readable font size don't overlap or shrink away once
    # the figure is placed in the thesis at print width.
    # Width scales with total duration.
    # -------------------------------------------------------
    fig_height = max(18, n_channels * 0.5)   # ~32 in for 64 ch
    fig_width  = 20

    for c in range(n_classes):

        fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                               constrained_layout=True)

        fig.suptitle(
            f"Saliency Map — {class_names[c]}",
            fontsize=28, fontweight="bold")

        sal_c = sal_concat[c]   # (n_channels, T_total)

        # Use pcolormesh instead of imshow so the ms_axis values map
        # directly onto the x-axis without any extent rescaling.
        # We need column edges → append one extra value.
        ms_edges = np.append(ms_axis, ms_axis[-1] + ms_per_sample)
        ch_edges = np.arange(n_channels + 1) - 0.5

        mesh = ax.pcolormesh(
            ms_edges, ch_edges, sal_c,
            cmap   = "jet",
            vmin   = vmin,
            vmax   = vmax,
            shading= "flat")

        # Y-axis — all channel names, readable
        if channel_names is not None:
            ax.set_yticks(np.arange(n_channels))
            ax.set_yticklabels(channel_names, fontsize=26)
        else:
            step   = max(1, n_channels // 8)
            ticks  = list(range(0, n_channels, step))
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(i) for i in ticks], fontsize=13)

        ax.set_ylabel("Channel", fontsize=26)
        ax.invert_yaxis()   # channel 0 at top, matching EEG convention

        # X-axis — ticks every 200 ms across the full range
        ms_lo = ms_axis[0]
        ms_hi = ms_axis[-1] + ms_per_sample
        x_ticks_ms = np.arange(
            int(np.ceil(ms_lo  / 200)) * 200,
            int(np.floor(ms_hi / 200)) * 200 + 1,
            200)
        ax.set_xticks(x_ticks_ms)
        ax.set_xticklabels([str(int(t)) for t in x_ticks_ms], fontsize=14)
        ax.set_xlabel("Time (ms)", fontsize=26)
        ax.set_xlim(ms_lo, ms_hi)

        # Vertical dashed lines at period boundaries (inner boundaries only)
        for p_name, bnd_ms in PERIOD_BOUNDARY_MS.items():
            if p_name in periods_present:
                ax.axvline(bnd_ms, color="white", linewidth=1.8,
                           linestyle="--", alpha=0.9)

        # Period label annotations just above the top of the heatmap
        for i, p in enumerate(periods_present):
            t_start = period_sample_boundaries[i]
            t_end   = period_sample_boundaries[i + 1]
            mid_ms  = (ms_axis[t_start] + ms_axis[t_end - 1]) / 2.0
            ax.text(mid_ms, 1.01, p,
                    ha="center", va="bottom",
                    fontsize=26, fontweight="bold",
                    color="black",
                    transform=ax.get_xaxis_transform())

        cbar = fig.colorbar(mesh, ax=ax, fraction=0.018, pad=0.012)
        cbar.set_label("|gradient|", fontsize=26)
        cbar.ax.tick_params(labelsize=13)

        class_tag = class_names[c].lower().replace(" ", "_")
        out_path  = os.path.join(
            results_dir, f"saliency_stimulus_{class_tag}.png")

        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"Stimulus saliency figure saved → {out_path}")


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

    # -------------------------------------------------
    # CROSS-PERIOD STIMULUS TIMELINE FIGURES
    # One figure per class, x-axis = -300 ms → 2000 ms
    # -------------------------------------------------
    if len(all_saliency) > 0:
        print("\nGenerating stimulus-timeline saliency figures …")
        plot_saliency_maps_by_stimulus(
            all_saliency  = all_saliency,
            results_dir   = results_dir,
            class_names   = class_names,
            channel_names = channel_names)

    print("\nSaliency analysis complete.")

    return all_saliency

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
    Plot GradCAM channel relevance as scalp topographies for one period.

    One subplot per class is shown in a single figure using the jet
    colormap on the standard 10-20 MNE montage.  The bar-chart output
    has been removed in favour of the topomap, which is more informative
    for spatially distributed EEG signals.

    Requires MNE-Python (pip install mne).  If MNE is not available the
    function falls back to a classes × channels heatmap.

    Parameters
    ----------
    cam           : np.ndarray, shape (n_classes, n_channels)
    period_name   : str
    results_dir   : str
    class_names   : list[str] | None
    channel_names : list[str] | None — 10-20 names matching the channel axis
    top_k         : int — unused (kept for API compatibility)

    Saves
    -----
    <results_dir>/gradcam_topomap_<period>.png
    """

    n_classes  = cam.shape[0]
    n_channels = cam.shape[1]
    period_lo  = period_name.lower()

    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    ch_labels = (channel_names if channel_names is not None
                 else [str(i) for i in range(n_channels)])

    os.makedirs(results_dir, exist_ok=True)

    global_max = np.nanmax(cam)
    cam_norm   = cam / (global_max + 1e-9)

    # ------------------------------------------------------------------
    # TOPOMAP path (requires MNE + valid 10-20 channel names)
    # ------------------------------------------------------------------

    if channel_names is not None:

        info    = mne.create_info(
            ch_names = channel_names,
            sfreq    = 1.0,
            ch_types = "eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="warn")

        fig, axes = plt.subplots(
            1, n_classes,
            figsize=(5 * n_classes, 5),
            constrained_layout=True)

        if n_classes == 1:
            axes = [axes]

        fig.suptitle(
            f"GradCAM Scalp Topography — {period_name}",
            fontsize=13, fontweight="bold")

        im_last = None

        for c, ax in enumerate(axes):

            scores = cam_norm[c].copy()
            scores[np.isnan(scores)] = 0.0

            im_last, _ = mne.viz.plot_topomap(
                scores,
                info,
                axes     = ax,
                cmap     = "jet",
                vlim     = (0, 1),
                contours = 6,
                show     = False,
                sensors  = True,
                names    = channel_names)

            ax.set_title(class_names[c], fontsize=11,
                         fontweight="bold", pad=10)

        if im_last is not None:
            cbar = fig.colorbar(im_last, ax=axes,
                                shrink=0.6, pad=0.02)
            cbar.set_label("Normalised GradCAM relevance", fontsize=9)

        out_path = os.path.join(
            results_dir, f"gradcam_topomap_{period_lo}.png")

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"GradCAM topomap saved → {out_path}")

    else:

        # ------------------------------------------------------------------
        # FALLBACK — heatmap (classes × channels) when no channel names given
        # ------------------------------------------------------------------

        fig2, ax2 = plt.subplots(
            figsize=(max(12, n_channels * 0.18), 3.5),
            constrained_layout=True)

        fig2.suptitle(
            f"GradCAM Channel Relevance Heatmap — {period_name}",
            fontsize=13, fontweight="bold")

        im = ax2.imshow(
            cam_norm,
            aspect="auto",
            cmap="jet",
            vmin=0, vmax=1,
            interpolation="nearest")

        ax2.set_yticks(range(n_classes))
        ax2.set_yticklabels(class_names, fontsize=10)

        tick_step = max(1, n_channels // 32)
        tick_pos  = list(range(0, n_channels, tick_step))

        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(
            [ch_labels[i] for i in tick_pos],
            fontsize=7, rotation=45, ha="right")

        ax2.set_xlabel("Channel", fontsize=10)

        cbar = plt.colorbar(im, ax=ax2, fraction=0.015, pad=0.01)
        cbar.set_label("Normalised relevance", fontsize=9)

        heatmap_path = os.path.join(
            results_dir, f"gradcam_heatmap_{period_lo}.png")

        fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        print(f"GradCAM heatmap saved → {heatmap_path}")


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
# TOPOMAP — GRADCAM ON SCALP SURFACE (SENS PERIOD ONLY)
# =========================================================
#
# Projects the per-class GradCAM channel relevance vector onto a 2-D
# scalp surface using MNE-Python's standard 10-20 montage and
# plot_topomap().  One subplot per class, one figure for SENS.
#
# Requires: pip install mne
#
# CHANNEL_NAMES must contain valid 10-20 labels recognisable by MNE
# (e.g. 'Fp1', 'Cz') and must be ordered identically to the channel
# axis of the GradCAM array (sorted 'channel' column in subject CSVs).
# =========================================================

def plot_topomap_gradcam(
    cam,
    period_name,
    results_dir,
    channel_names,
    class_names=None):
    """
    Plot GradCAM channel relevance as scalp topographies (10-20 montage).

    Requires MNE-Python.  Each class gets one subplot; the colour scale
    is shared across classes so they are directly comparable.

    Parameters
    ----------
    cam           : np.ndarray, shape (n_classes, n_channels)
    period_name   : str
    results_dir   : str
    channel_names : list[str] — 10-20 names matching the channel axis order
    class_names   : list[str] | None

    Saves
    -----
    <results_dir>/topomap_gradcam_<period_name.lower()>.png
    """

    n_classes = cam.shape[0]

    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    # -------------------------------------------------
    # BUILD MNE INFO WITH STANDARD 10-20 MONTAGE
    # -------------------------------------------------

    info = mne.create_info(
        ch_names = channel_names,
        sfreq    = 1.0,
        ch_types = "eeg")

    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="warn")

    # -------------------------------------------------
    # FIGURE
    # -------------------------------------------------

    fig, axes = plt.subplots(
        1, n_classes,
        figsize=(5 * n_classes, 5),
        constrained_layout=True)

    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        f"GradCAM Scalp Topography — {period_name}",
        fontsize=13, fontweight="bold")

    global_max = np.nanmax(cam)
    cam_norm   = cam / (global_max + 1e-9)

    for c, ax in enumerate(axes):

        scores = cam_norm[c].copy()
        scores[np.isnan(scores)] = 0.0

        im, _ = mne.viz.plot_topomap(
            scores,
            info,
            axes      = ax,
            cmap      = "jet",
            vlim      = (0, 1),
            contours  = 6,
            show      = False,
            sensors   = True,
            names     = channel_names)

        ax.set_title(class_names[c], fontsize=11, fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Normalised GradCAM relevance", fontsize=9)

    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(
        results_dir,
        f"topomap_gradcam_{period_name.lower()}.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Topomap figure saved → {out_path}")


def run_topomap_analysis(
    results_dir,
    period_folders,
    model_class,
    model_kwargs,
    device,
    channel_names,
    periods,
    class_names=None,
    n_classes=3,
    n_splits=10,
    seed=42):
    """
    End-to-end topomap pipeline for all requested periods.

    For each period: loads the saved GradCAM .npz if it exists,
    otherwise re-computes it from the results JSON.  Then renders
    and saves the scalp topography using the jet colormap.

    Parameters
    ----------
    results_dir    : str
    period_folders : dict — maps period name → CSV folder path
    model_class    : class
    model_kwargs   : dict
    device         : torch.device
    channel_names  : list[str] — valid 10-20 names in MNE montage
    periods        : list[str] — e.g. ['BSL', 'SENS', 'DELAY']
    class_names    : list[str] | None
    n_classes      : int  (default: 3)
    n_splits       : int  (default: 10)
    seed           : int  (default: 42)
    """

    for period_name in periods:

        print(f"\n{'='*20} TOPOMAP — {period_name} {'='*20}")

        cam_path = os.path.join(
            results_dir, f"gradcam_{period_name.lower()}.npz")

        if os.path.exists(cam_path):
            print(f"  Loading pre-computed GradCAM from {cam_path}")
            cam = np.load(cam_path)["cam"]

        else:
            print("  No pre-computed GradCAM found — computing now …")

            results_path = os.path.join(
                results_dir, f"results_{period_name.lower()}.json")

            with open(results_path, "r") as f:
                results = json.load(f)

            cam = compute_gradcam_period(
                results      = results,
                folder       = period_folders[period_name],
                model_class  = model_class,
                model_kwargs = model_kwargs,
                device       = device,
                n_classes    = n_classes,
                n_splits     = n_splits,
                seed         = seed)

            save_gradcam(cam, results_dir, period_name)

        plot_topomap_gradcam(
            cam           = cam,
            period_name   = period_name,
            results_dir   = results_dir,
            channel_names = channel_names,
            class_names   = class_names)

    print("Topomap analysis complete.")



# =========================================================
# GLM COMPARISON — LOAD LASSO WEIGHTS
# =========================================================
#
# The R script exports one CSV per period with columns:
#   subjectID, class, chn1, chn2, …, chn64
#
# class values are the R factor levels produced by cv.glmnet
# multinomial — typically 'spatial', 'verbal', 'visual'
# (alphabetical order).  They are mapped to CLASS_NAMES order
# via the `class_map` argument.
#
# The comparison uses |coefficient| because the LASSO sign
# encodes direction relative to the reference category, not
# importance magnitude.  Both CNN GradCAM and |LASSO| are
# normalised to [0, 1] before plotting so they are on the
# same visual scale.
# =========================================================

def load_glm_weights(
    glm_dir,
    period_name,
    class_map=None,
    n_channels=64):
    """
    Load per-subject LASSO weights exported from R and average across subjects.

    Expects a file named lasso_weights_<period_name.lower()>.csv in glm_dir,
    produced by the export_lasso_weights() R function documented in the module
    docstring.

    Parameters
    ----------
    glm_dir      : str  — directory containing the CSV files
    period_name  : str  — 'BSL', 'SENS', or 'DELAY'
    class_map    : dict | None — maps R class label → Python class index,
                   e.g. {'spatial': 1, 'verbal': 0, 'visual': 2}.
                   If None, classes are sorted alphabetically and mapped 0, 1, 2.
    n_channels   : int  (default: 64)

    Returns
    -------
    glm_weights : np.ndarray, shape (n_classes, n_channels)
                  Mean absolute LASSO coefficient across subjects, per class.
    class_order : list[str] — class labels in row order (matches class_map)
    """

    csv_path = os.path.join(
        glm_dir, f"lasso_weights_{period_name.lower()}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"LASSO weight file not found: {csv_path}\n"
            f"Run export_lasso_weights() in R first (see module docstring).")

    df = pd.read_csv(csv_path)

    # Ensure numeric types for coefficient columns
    chn_cols = [f"chn{i}" for i in range(1, n_channels + 1)]
    df[chn_cols] = df[chn_cols].apply(pd.to_numeric, errors="coerce")

    # Resolve class ordering
    r_classes = sorted(df["class"].unique())     # alphabetical R default

    if class_map is None:
        class_map = {cls: idx for idx, cls in enumerate(r_classes)}

    n_classes   = len(class_map)
    class_order = sorted(class_map, key=class_map.get)   # by Python index

    glm_weights = np.zeros((n_classes, n_channels), dtype=np.float64)

    for r_cls, py_idx in class_map.items():

        sub_df = df[df["class"] == r_cls]

        if sub_df.empty:
            print(f"  [warning] class '{r_cls}' not found in {csv_path}")
            continue

        # Mean |coefficient| across subjects
        glm_weights[py_idx] = sub_df[chn_cols].abs().mean(axis=0).values

    return glm_weights.astype(np.float32), class_order


# =========================================================
# GLM COMPARISON — PLOT AND CORRELATE
# =========================================================

def compare_cnn_glm(
    cam,
    glm_weights,
    period_name,
    results_dir,
    class_names=None,
    channel_names=None):
    """
    Plot CNN GradCAM vs LASSO |weights| scalp topographies side-by-side.

    Layout (one figure per period) — rotated 90° clockwise relative to the
    original row-per-class layout, so it matches the orientation of the
    GradCAM topomaps (one row, n_classes columns):
      Row 0 (top)    — LASSO |β| scalp topography, one panel per class
      Row 1 (bottom) — CNN GradCAM scalp topography, one panel per class

    LASSO and CNN GradCAM are normalised completely independently of one
    another (each on its own [0, 1] scale, each with its own colourbar),
    and within each method every class is normalised independently too
    (per-row max), so the spatial pattern of each individual topomap is
    not washed out by a larger value in a different class.

    Requires MNE-Python (pip install mne) for topographic plots.
    If MNE is not available the topomap panels are replaced by bar charts.

    Parameters
    ----------
    cam           : np.ndarray, shape (n_classes, n_channels)
                    GradCAM channel relevance (from run_gradcam_analysis).
    glm_weights   : np.ndarray, shape (n_classes, n_channels)
                    Mean absolute LASSO coefficients (from load_glm_weights).
    period_name   : str
    results_dir   : str
    class_names   : list[str] | None
    channel_names : list[str] | None — 10-20 channel names for topomap axes

    Saves
    -----
    <results_dir>/cnn_vs_glm_<period_name.lower()>.png

    Returns
    -------
    None
    """

    try:
        import mne as _mne
        _HAS_MNE = True
    except ImportError:
        _HAS_MNE = False

    n_classes  = cam.shape[0]
    n_channels = cam.shape[1]
    period_lo  = period_name.lower()

    if class_names is None:
        class_names = [f"Class {c}" for c in range(n_classes)]

    ch_labels = (channel_names if channel_names is not None
                 else [str(i) for i in range(n_channels)])

    # ---------------------------------------------------------------
    # Independent normalisation: LASSO and CNN GradCAM are each
    # normalised on their own scale, and within each method every
    # class (row) is normalised by its own max — not by a global
    # max shared across classes or across methods.
    # ---------------------------------------------------------------

    cam_raw = cam.copy().astype(np.float64)
    glm_raw = glm_weights.copy().astype(np.float64)
    cam_raw[np.isnan(cam_raw)] = 0.0
    glm_raw[np.isnan(glm_raw)] = 0.0

    cam_row_max = np.max(cam_raw, axis=1, keepdims=True)
    glm_row_max = np.max(glm_raw, axis=1, keepdims=True)

    cam_norm = cam_raw / (cam_row_max + 1e-9)
    glm_norm = glm_raw / (glm_row_max + 1e-9)

    # Build MNE info once (shared across all panels)
    mne_info = None
    if _HAS_MNE and channel_names is not None:
        mne_info = _mne.create_info(
            ch_names = channel_names,
            sfreq    = 1.0,
            ch_types = "eeg")
        montage = _mne.channels.make_standard_montage("standard_1020")
        mne_info.set_montage(montage, on_missing="warn")

    # Figure layout: 2 rows (LASSO / CNN) × n_classes columns,
    # i.e. the row-per-class / column-per-method layout rotated
    # 90° clockwise so it reads like the GradCAM topomap figures.
    fig = plt.figure(figsize=(5 * n_classes, 9), constrained_layout=True)
    fig.suptitle(
        f"CNN GradCAM vs LASSO |weights| — scalp topography — {period_name}",
        fontsize=13, fontweight="bold")

    outer_gs = gridspec.GridSpec(
        2, n_classes, figure=fig,
        height_ratios=[1, 1], hspace=0.25)

    im_glm = None   # keep handles for separate, independent colourbars
    im_cnn = None

    for c in range(n_classes):

        # -------------------------------------------------
        # TOP ROW — LASSO topomap (or bar fallback)
        # -------------------------------------------------

        ax_glm = fig.add_subplot(outer_gs[0, c])

        if _HAS_MNE and mne_info is not None:
            im_glm, _ = _mne.viz.plot_topomap(
                glm_norm[c],
                mne_info,
                axes     = ax_glm,
                cmap     = "jet",
                vlim     = (0, 1),
                contours = 6,
                show     = False,
                sensors  = True)
            ax_glm.set_title(
                f"{class_names[c]}",
                fontsize=11, fontweight="bold", pad=8)
        else:
            # Fallback: horizontal bar chart
            x_pos = np.arange(n_channels)
            ax_glm.barh(x_pos, glm_norm[c],
                        color=plt.cm.jet(glm_norm[c]),
                        edgecolor="none", height=0.7)
            ax_glm.invert_yaxis()
            ax_glm.set_yticks(x_pos)
            ax_glm.set_yticklabels(ch_labels, fontsize=5)
            ax_glm.set_xlim(0, 1.05)
            ax_glm.set_xlabel("Normalised relevance", fontsize=8)
            ax_glm.set_title(
                f"{class_names[c]}",
                fontsize=11, fontweight="bold")
            ax_glm.spines[["top", "right"]].set_visible(False)

        if c == 0:
            ax_glm.set_ylabel("LASSO |β|", fontsize=11, fontweight="bold")
            ax_glm.yaxis.labelpad = 12

        # -------------------------------------------------
        # BOTTOM ROW — CNN GradCAM topomap (or bar fallback)
        # -------------------------------------------------

        ax_cnn = fig.add_subplot(outer_gs[1, c])

        if _HAS_MNE and mne_info is not None:
            im_cnn, _ = _mne.viz.plot_topomap(
                cam_norm[c],
                mne_info,
                axes     = ax_cnn,
                cmap     = "jet",
                vlim     = (0, 1),
                contours = 6,
                show     = False,
                sensors  = True)
        else:
            x_pos = np.arange(n_channels)
            ax_cnn.barh(x_pos, cam_norm[c],
                        color=plt.cm.jet(cam_norm[c]),
                        edgecolor="none", height=0.7)
            ax_cnn.invert_yaxis()
            ax_cnn.set_yticks(x_pos)
            ax_cnn.set_yticklabels(ch_labels, fontsize=5)
            ax_cnn.set_xlim(0, 1.05)
            ax_cnn.set_xlabel("Normalised relevance", fontsize=8)
            ax_cnn.spines[["top", "right"]].set_visible(False)

        if c == 0:
            ax_cnn.set_ylabel("CNN GradCAM", fontsize=11, fontweight="bold")
            ax_cnn.yaxis.labelpad = 12

    # Single shared colourbar for the whole figure (only with MNE)
    if _HAS_MNE and mne_info is not None and im_cnn is not None:
        cbar = fig.colorbar(
            im_cnn, ax=fig.axes,
            shrink=0.5, pad=0.02, fraction=0.02)
        cbar.set_label("Normalised relevance (per class)", fontsize=9)

    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, f"cnn_vs_glm_{period_lo}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"CNN vs GLM topography figure saved → {out_path}")


# =========================================================
# GLM COMPARISON — END-TO-END PIPELINE
# =========================================================

def run_glm_comparison(
    results_dir,
    glm_dir,
    periods,
    class_names=None,
    channel_names=None,
    class_map=None,
    n_channels=64):
    """
    End-to-end CNN vs LASSO comparison pipeline for all requested periods.

    Loads GradCAM arrays from disk (produced by run_gradcam_analysis) and
    LASSO weight CSVs exported from R, then calls compare_cnn_glm() for
    each period.

    Parameters
    ----------
    results_dir   : str  — directory with gradcam_<period>.npz files
    glm_dir       : str  — directory with lasso_weights_<period>.csv files
    periods       : list[str]
    class_names   : list[str] | None
    channel_names : list[str] | None
    class_map     : dict | None — R class label → Python class index.
                    Default: {'spatial': 1, 'verbal': 0, 'visual': 2}
                    (matches CLASS_NAMES = ['Verbal', 'Spatial', 'Visual'])
    n_channels    : int (default: 64)

    Returns
    -------
    None
    """

    # Default map matches CLASS_NAMES = ['Verbal', 'Spatial', 'Visual']
    # R sorts class labels alphabetically: spatial=0, verbal=1, visual=2
    # but Python order is Verbal=0, Spatial=1, Visual=2
    if class_map is None:
        class_map = {"verbal": 0, "spatial": 1, "visual": 2}

    for period in periods:

        print(f"\n{'='*20} CNN vs GLM — {period} {'='*20}")

        # Load GradCAM (reuse if already computed)
        cam_path = os.path.join(results_dir, f"gradcam_{period.lower()}.npz")

        if not os.path.exists(cam_path):
            raise FileNotFoundError(
                f"GradCAM file not found: {cam_path}\n"
                f"Run run_gradcam_analysis() first.")

        cam = np.load(cam_path)["cam"]

        glm_weights, class_order = load_glm_weights(
            glm_dir      = glm_dir,
            period_name  = period,
            class_map    = class_map,
            n_channels   = n_channels)

        compare_cnn_glm(
            cam           = cam,
            glm_weights   = glm_weights,
            period_name   = period,
            results_dir   = results_dir,
            class_names   = class_names,
            channel_names = channel_names)

    print("\nCNN vs GLM comparison complete.")


# =========================================================
# LEARNED FILTER VISUALIZATION
# =========================================================
#
# Visualizes the raw learned convolutional kernel weights themselves
# (not gradients / relevance), in the style of EEGNet (Lawhern et al.,
# 2018), Figures 6 and 7:
#
#   - Temporal filters: 1-D kernels learned by the first (temporal)
#     Conv2d, shape (F1, kernel_length).
#   - Spatial filters: depthwise Conv2d kernels, shape (F1, D, n_channels),
#     where each of the F1 temporal filters has its own independent set
#     of D spatial filters (grouped convolution: groups = F1).
#
# Because raw filter weights are read directly off a single trained
# model instance (no gradients, no data), there is no notion of
# averaging across CV folds or subjects — different training runs can
# learn filters in a different order / sign (permutation + sign
# ambiguity), so each figure is tied to one specific fold's model,
# exactly as in the original EEGNet paper ("for one particular
# cross-subject fold").
# =========================================================

def _get_temporal_and_spatial_conv(model):
    """
    Return (temporal_conv, spatial_conv, F1, D) for the given model.

    F1 — number of temporal filters (out_channels of temporal_conv).
    D  — number of depthwise spatial filters learned per temporal
         filter. Because the spatial conv is grouped with
         groups = F1, PyTorch lays its out_channels out as F1
         contiguous blocks of size D, i.e. output channel index
         (g * D + d) belongs to group g (temporal filter g),
         depth-within-group d.

    Parameters
    ----------
    model : nn.Module — EEGNet or lightweightEEGNet instance

    Returns
    -------
    temporal_conv : nn.Conv2d
    spatial_conv  : nn.Conv2d
    F1            : int
    D             : int
    """

    if isinstance(model, EEGNet):
        temporal_conv = model.block1[0]
        spatial_conv  = model.block1[2]

    elif isinstance(model, lightweightEEGNet):
        temporal_conv = model.temporal_conv
        spatial_conv  = model.spatial_conv

    else:
        raise TypeError(
            f"Unsupported model type '{type(model).__name__}'. "
            f"Add its temporal/spatial conv locations to "
            f"_get_temporal_and_spatial_conv().")

    F1 = temporal_conv.out_channels
    D  = spatial_conv.out_channels // spatial_conv.groups

    return temporal_conv, spatial_conv, F1, D


def get_filter_weights(model):
    """
    Extract the learned temporal and spatial filter weights from a
    trained model as plain NumPy arrays.

    Parameters
    ----------
    model : nn.Module — EEGNet or lightweightEEGNet instance (eval mode)

    Returns
    -------
    temporal_weights : np.ndarray, shape (F1, kernel_length)
                        One 1-D temporal kernel per temporal filter.
    spatial_weights  : np.ndarray, shape (F1, D, n_channels)
                        spatial_weights[f, d] is the d-th depthwise
                        spatial filter associated with temporal
                        filter f.
    """

    temporal_conv, spatial_conv, F1, D = _get_temporal_and_spatial_conv(model)

    # temporal_conv.weight: (F1, 1, 1, kernel_length)
    temporal_weights = (temporal_conv.weight
                         .detach().cpu().numpy()
                         .squeeze(axis=(1, 2)))   # → (F1, kernel_length)

    # spatial_conv.weight: (F1*D, 1, n_channels, 1), grouped so output
    # channel (g*D + d) ↔ (temporal filter g, depth d) — see docstring
    # of _get_temporal_and_spatial_conv().
    n_channels = spatial_conv.weight.shape[2]

    spatial_weights = (spatial_conv.weight
                        .detach().cpu().numpy()
                        .reshape(F1, D, n_channels))

    return temporal_weights.astype(np.float32), spatial_weights.astype(np.float32)


def plot_temporal_spatial_filters(
    model,
    channel_names,
    results_dir,
    period_name,
    tag=None,
    sfreq=500.0):
    """
    Plot learned temporal kernels with their associated depthwise
    spatial filters, in the style of EEGNet (Lawhern et al., 2018)
    Figure 7: one column per temporal filter — top panel shows the
    1-D temporal kernel waveform, the D panels below it show that
    temporal filter's own spatial filters as scalp topoplots.

    Requires MNE-Python (pip install mne) for the topoplot panels.

    Parameters
    ----------
    model         : nn.Module — trained EEGNet / lightweightEEGNet, eval mode
    channel_names : list[str] — 10-20 channel names (must match the
                    montage used to train the spatial conv's channel axis)
    results_dir   : str — output directory (created if absent)
    period_name   : str — used in the title and filename, e.g. 'SENS'
    tag           : str | None — extra identifier for the filename,
                    e.g. a subject ID, so multiple folds/subjects don't
                    overwrite each other. If None, omitted from filename.
    sfreq         : float — sampling rate in Hz, used to convert the
                    temporal kernel's x-axis from samples to ms
                    (default: 500.0)

    Saves
    -----
    <results_dir>/filters_<period_name.lower()>[_<tag>].png
    """

    try:
        import mne as _mne
        _HAS_MNE = True
    except ImportError:
        _HAS_MNE = False

    temporal_weights, spatial_weights = get_filter_weights(model)
    F1, kernel_length = temporal_weights.shape
    _, D, n_channels  = spatial_weights.shape

    ms_per_sample = 1000.0 / sfreq
    t_axis_ms     = np.arange(kernel_length) * ms_per_sample

    # Build MNE info once for all spatial topoplots
    mne_info = None
    if _HAS_MNE and channel_names is not None:
        mne_info = _mne.create_info(
            ch_names = channel_names,
            sfreq    = 1.0,
            ch_types = "eeg")
        montage = _mne.channels.make_standard_montage("standard_1020")
        mne_info.set_montage(montage, on_missing="warn")

    n_rows = 1 + D   # temporal kernel row + D spatial filter rows

    fig = plt.figure(figsize=(3.2 * F1, 2.6 * n_rows), constrained_layout=True)

    period_label = period_name if tag is None else f"{period_name} — {tag}"
    fig.suptitle(
        f"Learned Temporal and Spatial Filters — {period_label}",
        fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(n_rows, F1, figure=fig, hspace=0.35, wspace=0.25)

    # Shared colour scale across all spatial topoplots so filter
    # magnitudes are directly comparable
    spatial_absmax = float(np.max(np.abs(spatial_weights)) + 1e-12)

    im_last = None

    for f in range(F1):

        # -------------------------------------------------
        # TOP ROW — temporal kernel waveform
        # -------------------------------------------------

        ax_t = fig.add_subplot(gs[0, f])
        ax_t.plot(t_axis_ms, temporal_weights[f], color="tab:blue", linewidth=1.4)
        ax_t.axhline(0, color="grey", linewidth=0.6, alpha=0.6)
        ax_t.set_title(f"Temp. Filter {f + 1}", fontsize=11, fontweight="bold")
        ax_t.set_xlim(t_axis_ms[0], t_axis_ms[-1])

        if f == 0:
            ax_t.set_ylabel("Weight", fontsize=9)

        ax_t.tick_params(labelsize=7)

        # -------------------------------------------------
        # ROWS BELOW — D spatial filters for this temporal filter
        # -------------------------------------------------

        for d in range(D):

            ax_s = fig.add_subplot(gs[1 + d, f])

            if _HAS_MNE and mne_info is not None:
                im_last, _ = _mne.viz.plot_topomap(
                    spatial_weights[f, d],
                    mne_info,
                    axes     = ax_s,
                    cmap     = "RdBu_r",
                    vlim     = (-spatial_absmax, spatial_absmax),
                    contours = 4,
                    show     = False,
                    sensors  = True)
            else:
                # Fallback: bar chart over channels
                x_pos = np.arange(n_channels)
                ax_s.bar(x_pos, spatial_weights[f, d],
                         color="tab:red", width=0.8)
                ax_s.set_xticks([])

            if f == 0:
                ax_s.set_ylabel(f"Spat. Filter {d + 1}",
                                fontsize=9, fontweight="bold")
                ax_s.yaxis.labelpad = 6

    if _HAS_MNE and mne_info is not None and im_last is not None:
        cbar = fig.colorbar(im_last, ax=fig.axes, shrink=0.5,
                            pad=0.015, fraction=0.012)
        cbar.set_label("Spatial filter weight", fontsize=10)

    os.makedirs(results_dir, exist_ok=True)

    period_lo = period_name.lower()
    tag_part  = "" if tag is None else f"_{tag}"
    out_path  = os.path.join(
        results_dir, f"filters_{period_lo}{tag_part}.png")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Temporal/spatial filter figure saved → {out_path}")


def run_filter_analysis(
    results_dir,
    period_folders,
    periods,
    model_class,
    model_kwargs,
    device,
    channel_names,
    subject_id=None,
    fold_idx=0,
    sfreq=500.0):
    """
    End-to-end pipeline: load one trained fold's model per period and
    plot its learned temporal + spatial filters.

    Because filter weights aren't averageable across independently
    trained folds (permutation/sign ambiguity — see module docstring),
    this picks ONE representative subject and fold per period rather
    than aggregating, matching how the original EEGNet paper visualizes
    filters ("for one particular cross-subject fold").

    Parameters
    ----------
    results_dir    : str
    period_folders : dict — maps period name → CSV folder path. Used to
                     read the chosen subject's CSV so the model is
                     rebuilt with the SAME n_samples it was trained
                     with — BSL/SENS/DELAY trial lengths differ, and
                     lightweightEEGNet's classifier dimension depends
                     on n_samples, so getting this wrong causes a
                     state_dict shape mismatch on load.
    periods        : list[str] — e.g. ['BSL', 'SENS', 'DELAY']
    model_class    : class — EEGNet or lightweightEEGNet
    model_kwargs   : dict  — passed to model_class(...)
    device         : torch.device
    channel_names  : list[str] — 10-20 channel names
    subject_id     : str | None — which subject's saved model to use.
                     If None, the first subject found in the results
                     JSON (with both saved models and a CSV file in
                     period_folders[period]) is used.
    fold_idx       : int — which CV fold's state dict to load (default: 0)
    sfreq          : float — sampling rate in Hz for the temporal axis

    Returns
    -------
    None — one figure per period is saved directly to results_dir.
    """

    for period in periods:

        print(f"\n{'='*20} FILTERS — {period} {'='*20}")

        results = load_period_results(results_dir, period)
        folder  = period_folders[period]

        # Map subject ID → CSV filename present in this period's folder
        subj_to_file = {}
        for file in sorted(os.listdir(folder)):
            if file.endswith(".csv"):
                subj = file.split("_")[1].split(".")[0]
                subj_to_file[subj] = file

        chosen_subj = subject_id
        if chosen_subj is None:
            for subj, data in results.items():
                if data.get("models") is not None and subj in subj_to_file:
                    chosen_subj = subj
                    break

        if (chosen_subj is None
                or results.get(chosen_subj, {}).get("models") is None
                or chosen_subj not in subj_to_file):
            print(f"  [skip] No subject with both saved models and a "
                  f"CSV file found for {period}")
            continue

        state_dicts = results[chosen_subj]["models"]

        if fold_idx >= len(state_dicts):
            print(f"  [skip] fold_idx={fold_idx} out of range "
                  f"({len(state_dicts)} folds available)")
            continue

        # Read the subject's actual trial length for this period —
        # required so the rebuilt model's classifier dimension matches
        # the checkpoint (lightweightEEGNet infers it from n_samples).
        X, _ = load_subject_csv(os.path.join(folder, subj_to_file[chosen_subj]))
        n_samples = X.shape[-1]
        del X

        models = load_subject_models(
            state_dicts  = [state_dicts[fold_idx]],
            model_class  = model_class,
            model_kwargs = model_kwargs,
            n_samples    = n_samples,
            device       = device)

        model = models[0]

        plot_temporal_spatial_filters(
            model         = model,
            channel_names = channel_names,
            results_dir   = results_dir,
            period_name   = period,
            tag           = f"subj{chosen_subj}_fold{fold_idx}",
            sfreq         = sfreq)

        del models
        gc.collect()
        torch.cuda.empty_cache()

    print("\nFilter analysis complete.")