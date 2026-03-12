# 20 models for every period, one for subject. Fist iteration
# Finished on the 06/03/2026 

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\Base Dataset")

# Loading the needed libraries
library(dplyr)
library(tidyr)
library(glmnet)
library(caret)
library(ggplot2)
library(tibble)
library(doParallel)

# ----- 0. PARALLELISATION -----

n_cores <- parallel::detectCores()
registerDoParallel(cores = n_cores)

# ----- 1. LOADING DATA -----

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

# Function that assigns written categories for easy interpretation
assign_categories <- function(df){
  df %>%
    group_by(subjectID, trialID) %>%
    mutate(category = case_when(
      y == 1 ~ "visual",
      y == 3 ~ "spatial",
      y == 5 ~ "verbal",
      TRUE ~ NA_character_
    )) %>%
    ungroup()
}

BSL   <- assign_categories(BSL)
SENS  <- assign_categories(SENS)
DELAY <- assign_categories(DELAY)

# Deleting the repeated column
BSL   <- BSL   %>% select(-y)
SENS  <- SENS  %>% select(-y)
DELAY <- DELAY %>% select(-y)

# ----- 2. PROCESSING DATA -----

# Function that join the data from every trial into one row, that later will be
# used for creating the X matrix for the model
process_dataset <- function(df) {
  
  results_df <- tibble(
    subjectID = numeric(),
    trialID   = numeric(),
    category  = character(),
    EEG       = list())
  
  for (subj in unique(df$subjectID)) {
    
    df_subj <- df %>% filter(subjectID == subj)
    subj_trials <- unique(df_subj$trialID)
    
    for (tr in subj_trials) {
      
      df_trial <- df_subj %>% filter(trialID == tr) %>% arrange(channel)
      
      ch_values <- numeric(64)
      
      for (chn in 1:64) {
        ts_vec <- as.numeric(df_trial[chn, 4][[1]])
        
        ch_values[chn] <- mean(ts_vec)
      }
      
      cat <- df_trial$category[1]
      
      trial_df <- tibble(
        subjectID = subj,
        trialID   = tr,
        category  = cat,
        EEG       = list(ch_values)
      )
      
      results_df <- bind_rows(results_df, trial_df)
    }
  }
  
  return(results_df)
}

BSL_proc   <- process_dataset(BSL)
SENS_proc  <- process_dataset(SENS)
DELAY_proc <- process_dataset(DELAY)

# Removing the unused data for optimization
rm(BSL, SENS, DELAY)
gc()

# 4. ----- CREATION OF X AND y -----

# Functions that creates the X matrix with 64 columns (one for channel) and 6067
# rows, one for every trial
make_X_matrix <- function(df_proc) {
  X <- t(sapply(df_proc$EEG, function(x) unlist(x)))
  return(X)
}

# Function that creates a vector with the category for every trial (6067 elem.)
make_y_vector <- function(df_proc) df_proc$category

X_BSL   <- make_X_matrix(BSL_proc)
y_BSL   <- make_y_vector(BSL_proc)

X_SENS  <- make_X_matrix(SENS_proc)
y_SENS  <- make_y_vector(SENS_proc)

X_DELAY <- make_X_matrix(DELAY_proc)
y_DELAY <- make_y_vector(DELAY_proc)

# Changing the column names to chn1, chn2, ... for easy visualitzation
colnames(X_BSL)   <- paste0("chn", 1:64)
colnames(X_SENS)  <- paste0("chn", 1:64)
colnames(X_DELAY) <- paste0("chn", 1:64)

# ----- 5. MODEL FORMULA -----

# Printing the model formula on the screen
make_model_formula <- function(n_channels = 64, y_name = "category") {
  predictors <- paste0("ch", 1:n_channels)
  rhs <- paste(predictors, collapse = " + ")
  as.formula(paste(y_name, "~", rhs))
}

model_formula <- make_model_formula(64)
print(model_formula)

# ----- 6. TRAINING MODELS BY SUBJECT AND PERIOD (TOTAL 60 MODELS) -----

# Function for training the model of every subject
train_models_by_subject <- function(df_proc, X, y) {
  
  subjects <- unique(df_proc$subjectID)
  model_list <- vector("list", length(subjects))
  names(model_list) <- paste0("subject_", subjects)
  
  for (i in seq_along(subjects)) {
    
    subj <- subjects[i]
    idx  <- which(df_proc$subjectID == subj)
    
    X_subj <- X[idx, , drop = FALSE]
    y_subj <- y[idx]
    
    model <- cv.glmnet(
      x = X_subj,
      y = y_subj,
      family = "multinomial",
      type.measure = "class",
      alpha = 1,
      parallel = TRUE
    )
    
    model_list[[i]] <- model
  }
  
  return(model_list)
}

models_BSL   <- train_models_by_subject(BSL_proc,   X_BSL,   y_BSL)
models_SENS  <- train_models_by_subject(SENS_proc,  X_SENS,  y_SENS)
models_DELAY <- train_models_by_subject(DELAY_proc, X_DELAY, y_DELAY)

# ----- 7. ACCURACY BY SUBJECT AND PERIOD (CV ACCURACY) -----

# Function for computing the accuracy of every subjects model
compute_accuracy_from_cv <- function(df_proc, model_list, dataset_name) {
  
  subjects <- unique(df_proc$subjectID)
  acc_df <- tibble(
    subjectID = numeric(),
    accuracy  = numeric(),
    dataset   = character()
  )
  
  for (i in seq_along(subjects)) {
    
    subj  <- subjects[i]
    model <- model_list[[i]]
    
    lambda_min <- model$lambda.min
    err <- model$cvm[model$lambda == lambda_min]
    acc <- 1 - err
    
    acc_df <- bind_rows(
      acc_df,
      tibble(
        subjectID = subj,
        accuracy  = as.numeric(acc)*100,
        dataset   = dataset_name
      )
    )
  }
  
  acc_df
}

acc_BSL   <- compute_accuracy_from_cv(BSL_proc,   models_BSL,   "BSL")
acc_SENS  <- compute_accuracy_from_cv(SENS_proc,  models_SENS,  "SENS")
acc_DELAY <- compute_accuracy_from_cv(DELAY_proc, models_DELAY, "DELAY")

# Mean accuracy by period
print(paste('Accuracy BSL:', round(mean(acc_BSL$accuracy), 1), '%'))
print(paste('Accuracy SENS:', round(mean(acc_SENS$accuracy), 1), '%'))
print(paste('Accuracy DELAY:', round(mean(acc_DELAY$accuracy), 1), '%'))

df_violin <- bind_rows(acc_BSL, acc_SENS, acc_DELAY)

# ----- 8. VIOLIN PLOT -----

df_violin$dataset <- factor(df_violin$dataset, 
                            levels = c("BSL", "SENS", "DELAY"))

ggplot(df_violin, aes(x = dataset, y = accuracy)) +
  
  geom_violin(aes(fill = dataset), trim = FALSE, alpha = 0.6, color = "black",
              show.legend = FALSE) +
  
  geom_point(aes(fill = dataset),
             shape = 21,
             size = 3,
             color = "black",
             stroke = 0.7,
             position = position_jitter(width = 0.05, height = 0),
             show.legend = FALSE) +
  
  stat_summary(fun = mean, geom = "crossbar", width = 0.4,
               fatten = 2, color = "black", show.legend = FALSE) +
  
  stat_summary(fun = median, geom = "point", size = 4, color = "white",
               show.legend = FALSE) +
  

  stat_summary(fun = mean, geom = "text",
             aes(label = sprintf("%.1f %%", ..y..)),
             vjust = -0.8, hjust = -1.5, size = 6, color = "black") +


  geom_hline(yintercept = 1/3*100, linetype = "dashed", color = "black",
           show.legend = FALSE) +
  
  scale_fill_manual(
    values = c("BSL" = "gray60", "SENS" = "forestgreen", "DELAY" = "purple")) +
  
  theme_minimal(base_size = 15) +
  labs(title = "Accuracy per subjecte i període",
       x = "Període experimental",
       y = "Accuracy (%)") +
  
  theme(axis.title = element_text(size = 13),
        axis.text = element_text(size = 12),
        legend.position = "none")