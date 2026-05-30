# ANN Interpretability for Working Memory Content Classification

> Master's Thesis (TFM) — Rafael Lefort | 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](environment.yml)
[![R](https://img.shields.io/badge/R-4.4-276DC3.svg)](renv.lock)

---

## Overview

This repository contains the full analysis pipeline developed for a Master's Thesis on **working memory** and **neural network interpretability**. The project applies both classical statistical models and deep learning to EEG recordings, then uses interpretability techniques to understand what the models learn.

The pipeline covers three stages:

1. **Exploratory analysis** of EEG data across cognitive conditions (Baseline, Sensory, Delay)
2. **Generalized Linear Models (GLM)** to establish a statistical baseline
3. **Convolutional Neural Networks (CNN)** with interpretability analysis (SHAP, saliency maps)

---

## Repository structure

```
TFM_ANN_Interpretability/
│
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml              ← Python reproducible environment
├── renv.lock                    ← R reproducible environment
│
├── data/
│   └── README.md                ← Dataset download instructions
│
├── Exploratory Analysis/
│   └── Exploratory analysis.R   ← First-pass EEG data exploration
│
├── GLM/
│   └── BSL, DELAY & SENS subject model.R  ← Subject-level GLM
│
├── Data Transformation/
│   └── RData_to_paq.R           ← Converts R data to .parquet for Python
│
└── NN/
    ├── CNN.py                   ← CNN architecture and training
    ├── CNN_results.py           ← Results visualisation
    └── Converge_CNN.py          ← Convergence diagnostics
```

---

## Data source

The dataset comes from the following publication:

> Turoman, N., et al. (2023). *Decoding the content of working memory in school-aged children.* Cortex, Volume 171, February 2024, Pages 136–152. https://doi.org/10.1016/j.cortex.2023.10.019

EEG recordings were collected during a working memory task across three experimental conditions: **Baseline (BSL)**, **Sensory (SENS)**, and **Delay**. See [`data/README.md`](data/README.md) for download instructions.

---

## Setup

### Requirements

- [conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or pip
- R ≥ 4.4 with [renv](https://rstudio.github.io/renv/)

### Python environment

```bash
git clone https://github.com/RafaLefort/TFM_ANN_Interpretability.git
cd TFM_ANN_Interpretability

conda env create -f environment.yml
conda activate tfm-ann-interpretability
```

Key Python packages: `torch 2.12`, `scikit-learn 1.8`, `numpy 2.4`, `pandas 3.0`, `shap 0.47`, `captum 0.8`, `matplotlib 3.10`, `seaborn 0.13`.

### R environment

```r
# Inside R, from the project root:
install.packages("renv")
renv::restore()
```

Key R packages: `tidyverse`, `ggplot2`, `dplyr`, `glmnet`, `caret`, `arrow`.

---

## Reproducing the results

Run the scripts in the following order:

### Step 1 — Exploratory analysis (R)

```bash
Rscript "Exploratory_Analysis/exploratory_analysis.R"
```

Inspects the EEG dataset, visualises channel activity distributions, and explores condition-level patterns.

### Step 2 — GLM baseline (R)

```bash
Rscript "GLM/GLM_model,_training_and_results.R"
```

Fits subject-level Generalized Linear Models for each experimental condition (BSL, DELAY, SENS).

### Step 3 — Data conversion (R → Python)

```bash
Rscript "Data_Transformation/RData_to_csv.R"
```

Converts the R dataset to `.csv` format so it can be read by the Python pipeline.

### Step 4 — CNN training (Python)

```bash
python NN/train.py
```

Trains the Convolutional Neural Network on the EEG data for working memory content classification. You can choose between two networks (EEGNet or lightweight_EEGNET) and including data augmentation (True or False).

### Step 5 — Results and diagnostics (Python)

```bash
python NN/results/plot_accuracy_results_by_period.py
python NN/results/plot_convergence_by_subject_and_period.py
```

Generates result plots and convergence diagnostics for the trained model.

---

## Methods

### Statistical baseline
Subject-level GLMs quantify the relationship between neural activity (EEG channels × time) and experimental condition. Results serve as an interpretable reference for comparison with the CNN.

### Deep learning
A CNN trained on the EEG data classifies working memory content across conditions. The architecture is described in [`NN/CNN.py`](NN/CNN.py).

### Interpretability
Following the IML/XAI framework (Molnar 2019, Biecek & Burzykowski 2021):

- **Global methods** — SHAP feature importance across the full dataset
- **Local methods** — SHAP values and saliency maps for individual predictions

---

## Reproducibility

All experiments use a fixed random seed for full reproducibility:

```python
# Python
torch.manual_seed(42)
numpy.random.seed(42)
```

```r
# R
set.seed(42)
```

Environment snapshots are provided in `environment.yml` (Python) and `renv.lock` (R) with exact package versions.

---

## References

- Turoman, N., et al. (2023). Decoding the content of working memory in school-aged children. *Cortex*, 171, 136–152. https://doi.org/10.1016/j.cortex.2023.10.019
- Molnar, C. (2019). *Interpretable Machine Learning*. Lulu.com. https://christophm.github.io/interpretable-ml-book/
- Biecek, P. & Burzykowski, T. (2021). *Explanatory Model Analysis*. Chapman and Hall/CRC. https://ema.drwhy.ai/
- Delicado, P. (2022). *Introduction to Interpretability in Machine Learning*. eBISS Summer School, Cesena.
- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16, 199–231.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.