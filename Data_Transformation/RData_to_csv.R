# =========================
# Libraries
# =========================

library(data.table)
library(dplyr)

# =========================
# Working directory
# =========================

setwd("C:/Users/lefor/Desktop/OneDriveBackupFiles/Documentos/Q4 MUEI/TFM/GitHub")

# =========================
# Load datasets
# =========================

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

# =========================
# Expand EEG matrix into columns
# =========================

expand_eeg_columns <- function(df, eeg_col="EEG") {

  cat("Expanding EEG columns...\n")

  eeg_matrix <- df[[eeg_col]]

  colnames(eeg_matrix) <- paste0(
    "EEG.V",
    seq_len(ncol(eeg_matrix))
  )

  df_out <- cbind(
    df,
    eeg_matrix
  )

  df_out <- dplyr::select(
    df_out,
    -all_of(eeg_col)
  )

  return(df_out)
}

# =========================
# Save one CSV per subject
# =========================

save_by_subject <- function(df, out_dir){

  dir.create(
    out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  subjects <- unique(df$subjectID)

  cat(
    length(subjects),
    "subjects found\n"
  )

  for (s in subjects){

    cat("Saving subject:", s, "\n")

    subj_data <- df %>%
      filter(subjectID == s)

    fwrite(
      subj_data,
      file = file.path(
        out_dir,
        paste0("subject_", s, ".csv")),
        quote=FALSE,
        na="NA"
    )

    rm(subj_data)
    gc()
  }

  cat(
    "Finished writing:",
    out_dir,
    "\n"
  )
}


# =========================
# Process BSL
# =========================

cat("\nProcessing BSL...\n")

BSL_exp <- expand_eeg_columns(
  BSL,
  "EEG"
)

save_by_subject(
  BSL_exp,
  "BSL_subjects"
)

rm(BSL_exp)
gc()


# =========================
# Process SENS
# =========================

cat("\nProcessing SENS...\n")

SENS_exp <- expand_eeg_columns(
  SENS,
  "EEG"
)

save_by_subject(
  SENS_exp,
  "SENS_subjects"
)

rm(SENS_exp)
gc()


# =========================
# Process DELAY
# =========================

cat("\nProcessing DELAY...\n")

DELAY_exp <- expand_eeg_columns(
  DELAY,
  "EEG"
)

save_by_subject(
  DELAY_exp,
  "DELAY_subjects"
)

rm(DELAY_exp)
gc()


cat("\nAll subject CSV files created successfully.\n")