# Exploratory analysis
# Finished on the 16/03/2026 

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\GitHub")

# Loading the needed libraries
library(tidyverse)

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
      TRUE ~ NA_character_)) %>%
    ungroup()
}

BSL   <- assign_categories(BSL)
SENS  <- assign_categories(SENS)
DELAY <- assign_categories(DELAY)

# Deleting the repeated column
BSL   <- BSL   %>% select(-y)
SENS  <- SENS  %>% select(-y)
DELAY <- DELAY %>% select(-y)

# ----- 2. EXPLORATORY ANALYSIS -----

# Making sure there is no missing values
if (anyNA(BSL) && anyNA(SENS) && anyNA(DELAY)){
  print("There is NAs in the datasets")
}else{
  print("There are no NAs in the datasets")
}

# Number of trials by subject
stats <- BSL %>% 
  group_by(subjectID) %>% 
  summarise(max_trial = max(trialID)) %>% 
  summarise(
    mean_val = mean(max_trial),
    sd_val   = sd(max_trial)
  )

mean_trials <- stats$mean_val
sd_trials   <- stats$sd_val

BSL %>% 
  group_by(subjectID) %>% 
  summarise(max_trial = max(trialID)) %>% 
  ggplot(aes(x = factor(subjectID), y = max_trial)) +
  
  geom_col(fill = "#4f81bd") +
  
  # --- Línies reals al gràfic ---
  geom_hline(aes(yintercept = mean_trials, 
                 color = "Mean  303.4", 
                 linetype = "Mean  303.4"), 
             linewidth = 1) +
  
  geom_hline(aes(yintercept = mean_trials + sd_trials, 
                 color = "Standart Deviation  47.9", 
                 linetype = "Standart Deviation  47.9"), 
             linewidth = 1) +
  
  geom_hline(aes(yintercept = mean_trials - sd_trials, 
                 color = "Standart Deviation  47.9", 
                 linetype = "Standart Deviation  47.9"), 
             linewidth = 1) +
  
  # --- Escales per a la llegenda ---
  scale_color_manual(
    name = "Summary",
    values = c("Mean  303.4" = "black", "Standart Deviation  47.9" = "red")
  ) +
  
  scale_linetype_manual(
    name = "Summary",
    values = c("Mean  303.4" = "dashed", "Standart Deviation  47.9" = "dashed")
  ) +
  
  labs(title = "Number of trials by subject",
       x = "Subject",
       y = "Number of trials") +
  
  coord_cartesian(ylim = c(0, 450)) +
  
  theme_minimal(base_size = 15) +
  
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "lightgray", color = "black"),
    legend.title = element_text(size = 15, face = "bold"),
    legend.text = element_text(size = 13)
  )



# Percentage of each category
BSL %>% 
  count(category) %>% 
  mutate(percent = n / sum(n) * 100) %>% 
  ggplot(aes(x = category, y = percent, fill = category)) +
  
  geom_col(width = 0.7) +
  
  scale_fill_manual(values = c(
    "visual"  = "#4f81bd",
    "spatial" = "#4f81bd",
    "verbal"  = "#4f81bd")) +
  
  labs(title = "Category distribution (percentatge)",
       x = "Category",
       y = "Percentatge (%)") +
  
  theme_minimal(base_size = 15) +
  theme(legend.position = "none")