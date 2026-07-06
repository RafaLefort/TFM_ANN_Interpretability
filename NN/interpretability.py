"""
interpretability.py
=======================
Stand-alone interpretability pipeline for EEG working memory classification.

Run this script after train.py has completed with SAVE_MODEL = True.
It loads the saved per-fold model weights from the results JSON files
and runs all requested interpretability analyses without re-training.

This script holds NO analysis logic of its own — every callable function
(saliency, GradCAM, topomap, GLM comparison) lives in
interpretability_utils.py and is imported below. This file is only
configuration (the EXPERIMENT / ANALYSES TO RUN / PATHS blocks) plus the
__main__ runner that calls those functions.

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
                Implemented in interpretability_utils.py → run_saliency_analysis().

GRADCAM       : Gradient-weighted Class Activation Mapping focused on
                channel relevance.  Hooks the spatial depthwise convolution
                in both architectures — the layer that explicitly learns to
                weight EEG channels — and computes a (n_channels,) importance
                score per class per period.
                Implemented in interpretability_utils.py → run_gradcam_analysis().

TOPOMAP       : GradCAM channel relevance projected onto a 2-D scalp surface
                using the standard 10-20 montage (MNE-Python).
                Requires: pip install mne
                Implemented in interpretability_utils.py → run_topomap_analysis().

GLM_COMPARE   : Side-by-side scalp topography comparison of CNN GradCAM
                channel relevance vs LASSO |coefficients| from the R GLM —
                one row per stimulus class, one figure per period.
                Requires LASSO weight CSVs exported from R (see
                interpretability_utils.run_glm_comparison docstring).
                Implemented in interpretability_utils.py → run_glm_comparison().
"""

import os
import random

import numpy as np
import torch

from models import EEGNet, lightweightEEGNet
from interpretability_utils import (
    run_filter_analysis,
    run_saliency_analysis,
    run_gradcam_analysis,
    run_topomap_analysis,
    run_glm_comparison,
    run_filter_analysis)


# =========================================================
# EXPERIMENT — mirror train.py exactly
# =========================================================

MODEL   = "lightweightEEGNet"       # 'EEGNet' | 'lightweightEEGNet'
AUGMENT = False                     # must match train.py
PERIODS = ['BSL', 'SENS', 'DELAY']  # any subset of ['BSL', 'SENS', 'DELAY']

CLASS_NAMES = ['Verbal', 'Spatial', 'Visual']

# Standard 64-channel 10-20 system — order must match the sorted 'channel'
# column in your CSV files.  Adjust if your cap layout differs.
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',  'FC5',
    'FC1', 'FC2', 'FC6', 'T7',  'C3',  'Cz',  'C4',  'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10','P7',  'P3',
    'Pz',  'P4',  'P8',  'PO9', 'O1',  'Oz',  'O2',  'PO10',
    'AF7', 'AF3', 'AF4', 'AF8', 'F5',  'F1',  'F2',  'F6',
    'FC3', 'FCz', 'FC4', 'C5',  'C1',  'C2',  'C6',  'CP3',
    'CPz', 'CP4', 'P5',  'P1',  'P2',  'P6',  'PO7', 'PO3',
    'POz', 'PO4', 'PO8', 'FT9', 'FT10','TP7', 'TP8', 'Iz',
]  # Channel 64 is Iz (inion) in the standard BioSemi 64-ch 10/20 cap;
   # the label 'Oz2' is non-standard and has been corrected here.


# =========================================================
# ANALYSES TO RUN
# =========================================================

SALIENCY_MAP = False  # vanilla gradient saliency (channel × time heatmap)
GRADCAM      = False  # GradCAM channel relevance topomaps
TOPOMAP      = False  # GradCAM projected onto scalp surface (all periods)
GLM_COMPARE  = True  # CNN GradCAM vs LASSO |weights| from R GLM
FILTERS      = False  # learned temporal/spatial filter weights (one fold)


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

# Directory containing lasso_weights_bsl/sens/delay.csv exported from R
GLM_DIR = os.path.join("GLM", "lasso_weights")


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

print(f"\nDevice          : {DEVICE}")
print(f"Model           : {MODEL.upper()}")
print(f"Periods         : {PERIODS}")
print(f"Results         : {RESULTS_DIR}")
print(f"Saliency        : {SALIENCY_MAP}")
print(f"GradCAM         : {GRADCAM}")
print(f"Topomap         : {TOPOMAP}")
print(f"GLM compare     : {GLM_COMPARE}")
print(f"Filters         : {FILTERS}\n")

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

    # -------------------------------------------------
    # TOPOMAP
    # -------------------------------------------------

    if TOPOMAP:

        if CHANNEL_NAMES is None:
            print("[topomap] CHANNEL_NAMES must be set to valid 10-20 names. "
                  "Skipping topomap.")
        else:
            run_topomap_analysis(
                results_dir    = RESULTS_DIR,
                period_folders = PERIOD_FOLDERS,
                model_class    = MODEL_CLASS,
                model_kwargs   = MODEL_KWARGS,
                device         = DEVICE,
                channel_names  = CHANNEL_NAMES,
                periods        = PERIODS,
                class_names    = CLASS_NAMES)

    # -------------------------------------------------
    # GLM COMPARISON — CNN GradCAM vs LASSO |weights|
    # Requires GRADCAM to have been run first (or gradcam_*.npz
    # files already present in RESULTS_DIR), and LASSO weight
    # CSVs exported from R in GLM_DIR.
    # -------------------------------------------------

    if GLM_COMPARE:

        run_glm_comparison(
            results_dir   = RESULTS_DIR,
            glm_dir       = GLM_DIR,
            periods       = PERIODS,
            class_names   = CLASS_NAMES,
            channel_names = CHANNEL_NAMES)
        
    # -------------------------------------------------
    # FILTER ANALYSIS — learned temporal/spatial filter weights
    # -------------------------------------------------
        
    if FILTERS:
        run_filter_analysis(
            results_dir    = RESULTS_DIR,
            period_folders = PERIOD_FOLDERS,
            periods        = PERIODS,
            model_class    = MODEL_CLASS,
            model_kwargs   = MODEL_KWARGS,
            device         = DEVICE,
            channel_names  = CHANNEL_NAMES,
            subject_id     = None,   # or pin a specific subject
            fold_idx       = 0)
