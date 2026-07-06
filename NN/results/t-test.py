"""
t-test.py

Paired-samples t-tests comparing classification accuracy across the 5 models
in the TFM (EEGNet, EEGNet_data_augmentation, lightweightEEGNet,
lightweightEEGNet_data_augmentation, GLM) for the SENS period.

For each of the 20 subjects, both models being compared were evaluated on
the same subject, so a paired design is used (scipy.stats.ttest_rel).

Expected file layout (relative to project root):
    NN/results/EEGNet/results_sens.json
    NN/results/EEGNet_data_augmentation/results_sens.json
    NN/results/lightweightEEGNet/results_sens.json
    NN/results/lightweightEEGNet_data_augmentation/results_sens.json
    GLM/results/results_sens.csv   (produced by GLM_model__training_and_results.R,
                                     section 9 "EXPORT PER-SUBJECT ACCURACIES")

Run from anywhere; paths are resolved relative to this script's location:
    NN/results/t-test.py  ->  project root is two levels up.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

PERIOD = "sens"  # this script is fixed to the SENS period

SCRIPT_DIR = Path(__file__).resolve().parent          # NN/results
PROJECT_ROOT = SCRIPT_DIR.parents[1]                   # project root

MODEL_PATHS = {
    "EEGNet": PROJECT_ROOT / "NN" / "results" / "EEGNet" / f"results_{PERIOD}.json",
    "EEGNet_aug": PROJECT_ROOT / "NN" / "results" / "EEGNet_data_augmentation" / f"results_{PERIOD}.json",
    "lightweightEEGNet": PROJECT_ROOT / "NN" / "results" / "lightweightEEGNet" / f"results_{PERIOD}.json",
    "lightweightEEGNet_aug": PROJECT_ROOT / "NN" / "results" / "lightweightEEGNet_data_augmentation" / f"results_{PERIOD}.json",
    "GLM": PROJECT_ROOT / "GLM" / "results" / f"results_{PERIOD}.csv",
}

ALPHA = 0.05


# ----------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------

def load_nn_accuracies(json_path: Path) -> dict:
    """Load {subjectID: accuracy} from a NN results_*.json file (accuracy in [0, 1])."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {str(subj): float(vals["accuracy"]) for subj, vals in data.items()}


def load_glm_accuracies(csv_path: Path) -> dict:
    """Load {subjectID: accuracy} from the GLM results_*.csv file (accuracy in [0, 1])."""
    df = pd.read_csv(csv_path)
    return {str(int(row.subjectID)): float(row.accuracy) for row in df.itertuples()}


def load_all_models(model_paths: dict) -> dict:
    accuracies = {}
    for model_name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing results file for '{model_name}': {path}\n"
                f"(Run the training/export scripts for this model first.)"
            )
        if path.suffix == ".json":
            accuracies[model_name] = load_nn_accuracies(path)
        elif path.suffix == ".csv":
            accuracies[model_name] = load_glm_accuracies(path)
        else:
            raise ValueError(f"Unsupported file type for '{model_name}': {path}")
    return accuracies


# ----------------------------------------------------------------------
# PAIRED T-TEST
# ----------------------------------------------------------------------

def paired_ttest(acc_a: dict, acc_b: dict, name_a: str, name_b: str) -> dict:
    """Paired t-test on matched subjects between two accuracy dicts (values in [0, 1])."""
    common_subjects = sorted(set(acc_a) & set(acc_b), key=lambda s: int(s))
    n = len(common_subjects)

    if n < 2:
        raise ValueError(
            f"Only {n} subject(s) in common between '{name_a}' and '{name_b}'. "
            f"Cannot run a paired t-test."
        )

    a = np.array([acc_a[s] for s in common_subjects]) * 100  # -> percentage
    b = np.array([acc_b[s] for s in common_subjects]) * 100

    mean_a, sd_a = a.mean(), a.std(ddof=1)
    mean_b, sd_b = b.mean(), b.std(ddof=1)

    t_stat, p_val = stats.ttest_rel(a, b)

    diff = a - b
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se_diff = sd_diff / np.sqrt(n)

    ci_low, ci_high = stats.t.interval(1 - ALPHA, df=n - 1, loc=mean_diff, scale=se_diff)

    cohens_dz = mean_diff / sd_diff

    return {
        "model_a": name_a,
        "model_b": name_b,
        "n": n,
        "mean_a": mean_a,
        "sd_a": sd_a,
        "mean_b": mean_b,
        "sd_b": sd_b,
        "t_stat": t_stat,
        "df": n - 1,
        "p_value": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_dz": cohens_dz,
        "significant": p_val < ALPHA,
    }


def format_result_sentence(res: dict) -> str:
    """Format the result following the TFM sentence template."""
    sig_str = "statistically significant" if res["significant"] else "not statistically significant"
    conclusion = (
        "exhibit significantly different classification performance across subjects"
        if res["significant"]
        else "do not differ significantly in classification performance across subjects"
    )
    return (
        f"A paired-samples t-test was conducted using the mean classification accuracy "
        f"obtained for each of the {res['n']} subjects to compare {res['model_a']} and "
        f"{res['model_b']}. The null hypothesis (H0) stated that the mean difference in "
        f"classification accuracy between the two classifiers was zero (\u03bcd = 0), while the "
        f"alternative hypothesis (H1) stated that the mean classification accuracy differed "
        f"between the two classifiers (\u03bcd \u2260 0). {res['model_a']} achieved a mean accuracy of "
        f"{res['mean_a']:.1f} \u00b1 {res['sd_a']:.1f}%, whereas {res['model_b']} achieved "
        f"{res['mean_b']:.1f} \u00b1 {res['sd_b']:.1f}%. The paired t-test indicated that the "
        f"difference was {sig_str} (t({res['df']}) = {res['t_stat']:.2f}, "
        f"p = {res['p_value']:.3f}, 95% CI [{res['ci_low']:.1f}, {res['ci_high']:.1f}], "
        f"Cohen's dz = {res['cohens_dz']:.2f}). These findings indicate that the two "
        f"classifiers {conclusion}."
    )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print(f"Loading per-subject accuracies for period: {PERIOD.upper()}\n")
    accuracies = load_all_models(MODEL_PATHS)

    for model, acc in accuracies.items():
        print(f"  {model}: {len(acc)} subjects loaded")

    model_names = list(accuracies.keys())
    results = []

    print("\n" + "=" * 100)
    print(f"PAIRWISE PAIRED T-TESTS \u2014 {PERIOD.upper()} PERIOD")
    print("=" * 100)

    for name_a, name_b in itertools.combinations(model_names, 2):
        res = paired_ttest(accuracies[name_a], accuracies[name_b], name_a, name_b)
        results.append(res)

        print(f"\n--- {name_a} vs {name_b} (n = {res['n']}) ---")
        print(format_result_sentence(res))

    # Save full results table
    df_results = pd.DataFrame(results)
    out_csv = SCRIPT_DIR / f"ttest_results_{PERIOD}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n\nFull results table saved to: {out_csv}")


if __name__ == "__main__":
    main()