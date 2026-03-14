# 📘 TFM — Working Memory \& ANN Interpretability



This repository contains the code and analysis developed for my Master’s Thesis (TFM) on \*\*Working Memory\*\*, \*\*neural data analysis\*\*, and \*\*interpretability of Artificial Neural Networks (ANNs)\*\*.  

The project is organized into three main components:



\- \*\*Exploratory Analysis\*\*  

\- \*\*Generalized Linear Models (GLM)\*\*  

\- \*\*Neural Networks (NN)\*\*  



Each component corresponds to a different stage of the analysis pipeline.



\---



\## 📂 Repository Structure



├── Exploratory Analysis/

│   └── Exploratory analysis.R

│

├── GLM/

│   └── BSL, DELAY \& SENS subject model.R

│

├── NN/

│   └── (empty for now — NN models will be added here)

│

├── .gitignore

└── README.md





\---



\## 🧬 Data Source



The dataset used in this project comes from the study:



\*\*Pani, P., et al. (2023).\*\*  

\*Working memory representations in human cortex during delay periods.\*  

Cortex.  

https://doi.org/10.1016/j.cortex.2023.10.019



The data were collected using intracranial EEG (iEEG) recordings during a working‑memory task involving Baseline, Sensory, and Delay periods.  

All analyses in this repository are based on the dataset provided in that publication.



\---



\## 🔍 1. Exploratory Analysis



This folder contains scripts used to:



\- Inspect the dataset  

\- Visualize distributions and correlations  

\- Explore channel activity across conditions  

\- Identify patterns that guide model selection  



\*\*File:\*\*  

\- `Exploratory analysis.R` — first-pass exploration of the dataset.



\---



\## 📊 2. GLM (Generalized Linear Models)



This section includes models used to quantify the relationship between neural activity and experimental conditions (Baseline, Delay, Sensory).



\*\*File:\*\*  

\- `BSL, DELAY \& SENS subject model.R` — GLM implementation for subject-level analysis.



\---



\## 🧠 3. Neural Networks (NN)



This folder will contain:



\- ANN architectures  

\- Training scripts  

\- Interpretability analyses (e.g., saliency maps, feature importance)  



Currently empty — models will be added as the project progresses.



\---



\## 🎯 Project Goals



\- Understand neural dynamics underlying working memory  

\- Compare classical statistical models (GLM) with ANN-based approaches  

\- Apply interpretability techniques to neural networks  

\- Evaluate model performance across cognitive conditions  



\---



\## 🛠️ Requirements



This project uses \*\*R\*\*.  

Recommended packages include:



\- `tidyverse`

\- `ggplot2`

\- `dplyr`

\- `glmnet`

\- `caret`

\- `keras` (for NN models)

\- `data.table`



Install missing packages with:



```r

install.packages(c("tidyverse", "ggplot2", "dplyr", "glmnet", "caret", "data.table"))





