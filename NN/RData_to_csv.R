# Converting .RData to .csv
# Finished on the 17/03/2026 

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\GitHub")

library(data.table)
library(dplyr)

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

# Function
expand_eeg_columns <- function(df, list_col = "EEG") {
  
  eeg_matrix <- df[[list_col]]
  
  colnames(eeg_matrix) <- paste0("EEG.V", seq_len(ncol(eeg_matrix)))

  df_out <- cbind(df, eeg_matrix)
  
  df_out <- select(df_out, -EEG)
  
  return(df_out)
}

BSL_exp <- expand_eeg_columns(BSL, "EEG")
SENS_exp <- expand_eeg_columns(SENS, "EEG")
DELAY_exp <- expand_eeg_columns(DELAY, "EEG")

fwrite(BSL_exp, "C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\GitHub\\NN\\BSL_allChan.csv")
fwrite(SENS_exp, "C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\GitHub\\NN\\SENS_allChan.csv")
fwrite(DELAY_exp, "C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\GitHub\\NN\\DELAY_allChan.csv")