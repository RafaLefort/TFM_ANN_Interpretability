import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# LOAD RESULTS
# =========================

with open("NN/results/results_bsl.pkl", "rb") as f:
    results_bsl = pickle.load(f)

with open("NN/results/results_sens.pkl", "rb") as f:
    results_sens = pickle.load(f)

with open("NN/results/results_delay.pkl", "rb") as f:
    results_delay = pickle.load(f)

# =========================
# AVERAGE LOSSES
# =========================

def average_losses(train_losses, val_losses):
    max_len = max(len(l) for l in train_losses)

    train_avg = np.zeros(max_len)
    val_avg = np.zeros(max_len)
    counts = np.zeros(max_len)

    for t, v in zip(train_losses, val_losses):
        for i in range(len(t)):
            train_avg[i] += t[i]
            val_avg[i] += v[i]
            counts[i] += 1

    train_avg /= counts
    val_avg /= counts

    return train_avg, val_avg

# =========================
# PLOT GRID FUNCTION
# =========================

def plot_period(results, name):
    save_dir = "NN/results/plots"
    os.makedirs(save_dir, exist_ok=True)

    subjects = sorted(results.keys())

    fig, axes = plt.subplots(5, 4, figsize=(8.27, 11.69), sharex=True)

    fig.suptitle(f"{name} - Convergence across subjects", fontsize=16)

    for idx, subj in enumerate(subjects):
        row = idx // 4
        col = idx % 4

        ax = axes[row, col]

        train_losses = results[subj]["train_losses"]
        val_losses = results[subj]["val_losses"]

        train_avg, val_avg = average_losses(train_losses, val_losses)

        ax.plot(train_avg, label="Train")
        ax.plot(val_avg, label="Val")

        ax.set_title(f"Subject {subj}", fontsize=8)
        ax.grid()

        # ✅ Y-axis only on left of each row
        if col == 0:
            ax.set_ylabel("Loss")
        else:
            ax.set_yticklabels([])

        # ✅ X-axis: only last row shows ticks + label
        if row == 4:
            ax.set_xlabel("Epoch")
            ax.tick_params(axis='x', labelsize=7)
        else:
            ax.set_xticklabels([])

        # Optional: smaller ticks everywhere
        ax.tick_params(axis='y', labelsize=7)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"{save_dir}/convergence_{name}_grid.png", dpi=300)
    plt.close()

# =========================
# RUN
# =========================

plot_period(results_bsl, "BSL")
plot_period(results_sens, "SENS")
plot_period(results_delay, "DELAY")

print("Grid plots saved!")