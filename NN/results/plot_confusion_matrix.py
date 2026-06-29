"""
plot_confusion_matrix.py
=========================
Generates two diagnostic figures from saved lightweightEEGNet results:

1. Confusion matrices (Baseline, Sensory, Delay) — row-normalized
   classification accuracy (%) per trained/tested class, jet colormap.
2. Classification accuracy per individual — paired Sensory→Delay
   slopes, one line/color per subject, ranked by Sensory accuracy.

Input
-----
Reads per-period JSON result files saved by utils.save_results(), i.e.:

    NN/results/lightweightEEGNet/results_BSL.json
    NN/results/lightweightEEGNet/results_SENS.json
    NN/results/lightweightEEGNet/results_DELAY.json

Each JSON is expected to contain, per subject, at least:
    - "accuracy"        : float, mean CV accuracy
    - "confusion_matrix" : list[list[int]] (3x3), summed over folds,
                            rows = true class, cols = predicted class
      (also accepts the key "conf_total", used as an alias)

Class order is assumed to be [visual, spatial, verbal] matching label
order in the original GLM step (1, 3, 5) -> (visual, spatial, verbal).

Output
------
Saves two figures to the same results directory:
    confusion_matrices.png
    accuracy_per_individual.png
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "NN/results/lightweightEEGNet"

PERIODS      = ["BSL", "SENS", "DELAY"]
PERIOD_TITLE = {"BSL": "Baseline", "SENS": "Sensory", "DELAY": "Delay"}

CLASS_NAMES = ["Visual", "Spatial", "Verbal"]

OUT_CONFUSION = os.path.join(RESULTS_DIR, "confusion_matrices.png")
OUT_INDIVIDUAL = os.path.join(RESULTS_DIR, "accuracy_per_individual.png")


# =========================================================
# LOADING HELPERS
# =========================================================

def load_period_results(results_dir, period):
    """
    Load the results JSON for a given period.

    Tries a couple of common filename patterns to stay robust to
    naming differences across save_results versions.
    """

    candidates = [
        os.path.join(results_dir, f"results_{period}.json"),
        os.path.join(results_dir, f"{period}.json"),
        os.path.join(results_dir, f"results_{period.lower()}.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Could not find results file for period '{period}' in "
        f"'{results_dir}'. Tried: {candidates}"
    )


def get_subject_entries(period_data):
    """
    Normalize the JSON structure into a dict {subject_id: subject_dict}.

    Handles both:
        {"subjects": {"1383": {...}, "4179": {...}}}
    and:
        {"1383": {...}, "4179": {...}}
    """

    if "subjects" in period_data and isinstance(period_data["subjects"], dict):
        return period_data["subjects"]

    # Fallback: assume top-level dict is already subject -> data
    return {
        k: v for k, v in period_data.items()
        if isinstance(v, dict) and ("accuracy" in v or "confusion_matrix" in v or "conf_total" in v)
    }


def get_confusion_matrix(subject_dict):
    """Return the 3x3 confusion matrix as a numpy array, trying known keys."""

    for key in ("conf_matrix", "conf_total", "confusion"):
        if key in subject_dict:
            return np.array(subject_dict[key], dtype=float)

    raise KeyError(
        "No confusion matrix found in subject entry. "
        "Expected one of: 'confusion_matrix', 'conf_total', 'confusion'."
    )


def get_accuracy(subject_dict):
    """Return the subject's mean CV accuracy as a percentage (0-100)."""

    for key in ("accuracy", "mean_accuracy", "cv_accuracy"):
        if key in subject_dict:
            val = subject_dict[key]
            # Accuracies might be stored as fraction (0-1) or percentage (0-100)
            return val * 100 if val <= 1.0 else val

    raise KeyError(
        "No accuracy value found in subject entry. "
        "Expected one of: 'accuracy', 'mean_accuracy', 'cv_accuracy'."
    )


# =========================================================
# FIGURE 1 — CONFUSION MATRICES (BSL / SENS / DELAY)
# =========================================================

def build_period_confusion_pct(period_data):
    """
    Aggregate confusion matrices across all subjects for one period,
    then row-normalize to obtain classification accuracy (%) per
    trained (row) / tested (column) class pair.
    """

    subjects = get_subject_entries(period_data)

    total_cm = np.zeros((3, 3), dtype=float)

    for subj_id, subj_data in subjects.items():
        cm = get_confusion_matrix(subj_data)
        total_cm += cm

    row_sums = total_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero

    cm_pct = 100 * total_cm / row_sums

    return cm_pct


def plot_confusion_matrices(all_period_data):
    """Create the 3-panel confusion matrix figure (jet colormap)."""

    # Reserve space on the right (right=0.88) up front for the colorbar,
    # so the third panel is never squeezed/clipped by it afterwards.
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2),
        gridspec_kw={"wspace": 0.15})
    fig.subplots_adjust(left=0.06, right=0.88, top=0.82, bottom=0.18)

    fig.suptitle("Confusion matrices", fontsize=15, fontweight="bold")

    vmin, vmax = 10, 80  # matches the colorbar range in the reference image

    im = None

    for ax, period in zip(axes, PERIODS):

        cm_pct = build_period_confusion_pct(all_period_data[period])

        im = ax.imshow(cm_pct, cmap="jet", vmin=vmin, vmax=vmax)

        ax.set_title(PERIOD_TITLE[period], fontsize=13)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)

        ax.set_xlabel("Tested class")
        if period == PERIODS[0]:
            ax.set_ylabel("Trained class")

        # Annotate each cell with its value
        for i in range(3):
            for j in range(3):
                value = cm_pct[i, j]
                # Choose text color for contrast against the jet colormap
                text_color = "white" if value < 45 or value > 70 else "black"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Dedicated colorbar axes on the right, sized/positioned independently
    # of the 3 panels so nothing gets clipped.
    cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Classification accuracy (%)", fontsize=11)

    plt.savefig(OUT_CONFUSION, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_CONFUSION}")


# =========================================================
# FIGURE 2 — ACCURACY PER INDIVIDUAL (SENSORY -> DELAY)
# =========================================================

def build_subject_accuracy_table(sens_data, delay_data):
    """
    Build a list of (subject_id, sens_accuracy, delay_accuracy),
    matched by subject id and sorted by sensory accuracy ascending.
    """

    sens_subjects = get_subject_entries(sens_data)
    delay_subjects = get_subject_entries(delay_data)

    common_ids = sorted(
        set(sens_subjects.keys()) & set(delay_subjects.keys()),
        key=lambda sid: get_accuracy(sens_subjects[sid]))

    table = []
    for sid in common_ids:
        sens_acc = get_accuracy(sens_subjects[sid])
        delay_acc = get_accuracy(delay_subjects[sid])
        table.append((sid, sens_acc, delay_acc))

    return table


def plot_accuracy_per_individual(all_period_data):
    """Create the paired Sensory -> Delay accuracy-per-subject figure."""

    table = build_subject_accuracy_table(
        all_period_data["SENS"], all_period_data["DELAY"])

    n_subjects = len(table)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    cmap = plt.get_cmap("jet")
    colors = [cmap(i / max(n_subjects - 1, 1)) for i in range(n_subjects)]

    for rank, (sid, sens_acc, delay_acc) in enumerate(table):

        color = colors[rank]

        ax.plot(
            [0, 1], [sens_acc, delay_acc],
            color=color, linewidth=1.2, alpha=0.85,
            marker="o", markersize=7,
            markerfacecolor=color, markeredgecolor=color,
            label=f"{rank + 1}")

    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sensory", "Delay"], fontsize=12)

    ax.set_ylabel("Classification accuracy (%)", fontsize=12)
    ax.set_title("Classification accuracy per individual", fontsize=14, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        title="Accuracy\nat Sensory",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, title_fontsize=9, frameon=False,
        ncol=1, handlelength=1.5, labelspacing=0.4)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(OUT_INDIVIDUAL, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_INDIVIDUAL}")


# =========================================================
# MAIN
# =========================================================

def main():

    all_period_data = {
        period: load_period_results(RESULTS_DIR, period)
        for period in PERIODS
    }

    plot_confusion_matrices(all_period_data)
    plot_accuracy_per_individual(all_period_data)


if __name__ == "__main__":
    main()