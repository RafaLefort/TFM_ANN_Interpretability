"""
train.py
========
Entry point for EEG working memory classification experiments.

Usage
-----
Configure the EXPERIMENT block below, then run:

    python train.py

Configuration
-------------
MODEL            : 'EEGNet' or 'lightweightEEGNet'
AUGMENT          : True  → frequency mixup on training set (lightweightEEGNet only)
                   False → no augmentation
PERIODS          : list of periods to run, any subset of
                   ['BSL', 'SENS', 'DELAY']
SAVE_MODEL       : save per-fold model weights to the results JSON;
                   required for any post-hoc interpretability analysis
SALIENCY_MAP_TIME: run saliency map analysis after training completes;
                   requires SAVE_MODEL = True

All other hyperparameters are set in the TRAINING CONFIG block.
"""

import os
import random

import numpy as np
import torch

from models import EEGNet, lightweightEEGNet
from utils import (
    run_all_subjects,
    summarize_results,
    save_results)
from interpretability import run_saliency_analysis


# =========================================================
# EXPERIMENT — edit this block to configure your run
# =========================================================

MODEL               = "lightweightEEGNet"      # 'EEGNet' | 'lightweightEEGNet'
AUGMENT             = False                    # True | False
PERIODS             = ['BSL', 'SENS', 'DELAY'] # any subset of ['BSL', 'SENS', 'DELAY']
SALIENCY_MAP_TIME   = True                     # run saliency maps after training


# =========================================================
# PATHS
# =========================================================

BASE_PATH = "/home/rlefort/"

PERIOD_FOLDERS = {
    "BSL":   os.path.join(BASE_PATH, "BSL_subjects"),
    "SENS":  os.path.join(BASE_PATH, "SENS_subjects"),
    "DELAY": os.path.join(BASE_PATH, "DELAY_subjects"),
}

# Results directory is named automatically from the experiment config
_aug_tag    = "_data_augmentation" if AUGMENT else ""
_model_tag  = "lightweightEEGNet" if MODEL == "lightweightEEGNet" else "EEGNet"
RESULTS_DIR = os.path.join(
    BASE_PATH,
    f"{_model_tag}{_aug_tag }")


# =========================================================
# REPRODUCIBILITY
# =========================================================

SAVE_MODEL = SALIENCY_MAP_TIME  # Must save models to do saliency analysis later 

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice  : {DEVICE}")
print(f"Model   : {MODEL.upper()}")
print(f"Augment : {AUGMENT}")
print(f"Periods : {PERIODS}")
print(f"Saliency: {SALIENCY_MAP_TIME}")
print(f"Results : {RESULTS_DIR}\n")


# =========================================================
# MODEL AND TRAINING CONFIG
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
    raise ValueError(f"Unknown model '{MODEL}'. Choose 'EEGNet' or 'lightweightEEGNet'.")

TRAIN_CFG = {
    "epochs":           100,
    "patience":         20,
    "lr":               1e-3,
    "weight_decay":     1e-5,
    "optimizer":        "adamw",
    "scheduler":        "cosine",
    "label_smoothing":  0.0,
    "batch_size":       32,
}

# =========================================================
# RUN EXPERIMENTS
# =========================================================

all_results = {}

for period in PERIODS:

    folder = PERIOD_FOLDERS[period]

    results = run_all_subjects(
        folder        = folder,
        period_name   = period,
        model_class   = MODEL_CLASS,
        model_kwargs  = MODEL_KWARGS,
        train_cfg     = TRAIN_CFG,
        device        = DEVICE,
        augment       = AUGMENT,
        save_models   = SAVE_MODEL)

    all_results[period] = results

    save_results(results, RESULTS_DIR, period)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)

for period in PERIODS:
    summarize_results(all_results[period], period)


# =========================================================
# SALIENCY MAP ANALYSIS
# =========================================================

if SALIENCY_MAP_TIME:

    if not SAVE_MODEL:
        print("\n[WARNING] SALIENCY_MAP_TIME = True but SAVE_MODEL = False. "
              "Re-run with SAVE_MODEL = True to enable saliency analysis.")

    else:
        run_saliency_analysis(
            results_dir    = RESULTS_DIR,
            period_folders = PERIOD_FOLDERS,
            periods        = PERIODS,
            model_class    = MODEL_CLASS,
            model_kwargs   = MODEL_KWARGS,
            device         = DEVICE,
            class_names    = ['Verbal', 'Spatial', 'Visual'])