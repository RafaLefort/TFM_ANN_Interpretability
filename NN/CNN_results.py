import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import pandas as pd

# =========================
# Load data
# =========================

with open("NN/results/results_bsl.pkl", "rb") as f:
    results_bsl = pickle.load(f)

with open("NN/results/results_sens.pkl", "rb") as f:
    results_sens = pickle.load(f)

with open("NN/results/results_delay.pkl", "rb") as f:
    results_delay = pickle.load(f)

sns.set_theme(style="whitegrid", font_scale=2)

# =========================
# Prepare dataframe
# =========================

acc_bsl = [res["accuracy"] * 100 for res in results_bsl.values()]
acc_sens = [res["accuracy"] * 100 for res in results_sens.values()]
acc_delay = [res["accuracy"] * 100 for res in results_delay.values()]

df = pd.DataFrame({
    "Accuracy": acc_bsl + acc_sens + acc_delay,
    "Condition": (["Baseline"] * len(acc_bsl) +
                  ["Sensory"] * len(acc_sens) +
                  ["Delay"] * len(acc_delay))
})

palette = {
    "Baseline": "#9e9e9e",
    "Sensory": "#2ca02c",
    "Delay": "#9c27b0"
}

conditions = ["Baseline", "Sensory", "Delay"]

# =========================
# Plot
# =========================

plt.figure(figsize=(16, 9))

sns.violinplot(data=df,
               x="Condition", y="Accuracy",
               hue="Condition",
               palette=palette,
               inner=None,
               linewidth=1.5,
               saturation=0.3,
               legend=False,
               width=0.55)

sns.stripplot(data=df,
              x="Condition", y="Accuracy",
              color="black",
              size=10,
              jitter=0.1,
              alpha=0.6,
              zorder=2)

# =========================
# Statistics
# =========================

grouped = df.groupby("Condition")["Accuracy"]
means = grouped.mean()
medians = grouped.median()
stds = grouped.std()

# =========================
# Add mean, median, SD
# =========================

for i, cond in enumerate(conditions):
    mean_val = means[cond]
    median_val = medians[cond]
    std_val = stds[cond]
    color = palette[cond]

    # --- SD (vertical error bar) ---
    plt.errorbar(i, mean_val,
                 yerr=std_val,
                 fmt='none',
                 ecolor='black',
                 elinewidth=2,
                 capsize=5,
                 zorder=3)

    # --- Mean line ---
    plt.hlines(mean_val, i - 0.15, i + 0.15,
               colors="black", linewidth=2, zorder=4)

    # --- Mean label ---
    plt.text(i + 0.30, mean_val + 2,
             f"{mean_val:.1f}%",
             color=color,
             fontsize=17,
             fontweight='bold',
             va='center',
             zorder=5)

    # --- Median marker (white diamond) ---
    plt.scatter(i, median_val,
                marker='D',
                s=120,
                facecolor='white',
                edgecolor='black',
                linewidth=1.5,
                zorder=6)

# =========================
# Chance level
# =========================

plt.axhline(33.33, linestyle="--", color="black", linewidth=2)

# =========================
# Legend
# =========================

dot_baseline = Line2D([0], [0], marker='o', color=palette["Baseline"], linestyle='None', markersize=6)
dot_sensory = Line2D([0], [0], marker='o', color=palette["Sensory"], linestyle='None', markersize=6)
dot_delay = Line2D([0], [0], marker='o', color=palette["Delay"], linestyle='None', markersize=6)

median_marker = Line2D([0], [0], marker='D', color='black',
                       markerfacecolor='white', linestyle='None', markersize=8)

sd_line = Line2D([0], [0],
                 marker='|',
                 color='black',
                 linestyle='None',
                 markersize=15,
                 markeredgewidth=2)

legend_elements = [
    Line2D([0], [0], color='black', lw=2),
    median_marker,
    sd_line,
    (dot_baseline, dot_sensory, dot_delay),
    Line2D([0], [0], color='black', linestyle='--', lw=1)
]

plt.legend(handles=legend_elements,
           labels=['Mean accuracy', 'Median accuracy', 'Standart deviation', 'Individual data points', 'Chance level'],
           handler_map={tuple: HandlerTuple(ndivide=None)},
           loc='upper left',
           frameon=True)

# =========================
# Labels & layout
# =========================

plt.ylabel("Classification accuracy (%)")
plt.title("Classification of time-averaged patterns")
plt.ylim(17, 93)

sns.despine()
plt.tight_layout()
plt.show()