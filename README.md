# 📘 TFM — Working Memory & ANN Interpretability

This repository contains the code and analysis developed for my Master’s Thesis (TFM) on **Working Memory**, **neural data analysis**, and **interpretability of Artificial Neural Networks (ANNs)**.  
The project is organized into three main components:

- **Exploratory Analysis**  
- **Generalized Linear Models (GLM)**  
- **Neural Networks (NN)**  

Each component corresponds to a different stage of the analysis pipeline.

---

## 📂 Repository Structure

├── Exploratory Analysis/  
│   └── Exploratory analysis.R  
│  
├── GLM/  
│   └── BSL, DELAY & SENS subject model.R  
│  
├── NN/  
│   └── RData_to_paq.R  
│   └── CNN.py  
│   └── CNN_results.py  
│  
├── .gitignore  
└── README.md  


---

## 🧬 Data Source

The dataset used in this project comes from the study:

**Turoman, N., et al. (2023).**  
*Decoding the content of working memory in school-aged children.*  
Cortex. Volume 171, February 2024, Pages 136-152
https://doi.org/10.1016/j.cortex.2023.10.019

The data were collected using EEG (EEG) recordings during a working‑memory task involving Baseline, Sensory, and Delay periods.  
All analyses in this repository are based on the dataset provided in that publication.

---

## 🔍 1. Exploratory Analysis

This folder contains scripts used to:

- Inspect the dataset  
- Visualize distributions and correlations  
- Explore channel activity across conditions  
- Identify patterns that guide model selection  

**File:**  
- `Exploratory analysis.R` — first-pass exploration of the dataset.

---

## 📊 2. GLM (Generalized Linear Models)

This section includes models used to quantify the relationship between neural activity and experimental conditions (Baseline, Delay, Sensory).

**File:**  
- `BSL, DELAY & SENS subject model.R` — GLM implementation for subject-level analysis.

---

## 🧠 3. Neural Networks (NN)

This folder will contain:

- CNN architectures  
- Training scripts  
- Interpretability analyses (e.g., saliency maps, feature importance)  

**Files:**  
- `RData_to_paq.R` — Conversion of the RData dataset to paq format for importing it to python.
- `CNN.py` — CNN implementation for subject-level analysis.
- `CNN_results.py` — Plotting of the CNN results.

---

## 🎯 Project Goals

- Understand neural dynamics underlying working memory  
- Compare classical statistical models (GLM) with ANN-based approaches  
- Apply interpretability techniques to neural networks  
- Evaluate model performance across cognitive conditions  

---

## 🛠️ Requirements

This project uses **R** and **Python**.  
Recommended **R** packages include:

- `tidyverse`
- `ggplot2`
- `dplyr`
- `glmnet`
- `caret`
- `arrow`

Install missing packages with:

```r
install.packages(c("tidyverse", "ggplot2", "dplyr", "glmnet", "caret", "arrow")


Recommended **Python** packages include:

- `pickle`
- `torch`
- `pandas`
- `numpy`
- `sklearn`
- `seaborn`
- `matplotlib`

Install missing packages with:

```r
pip install "<package_name>"
