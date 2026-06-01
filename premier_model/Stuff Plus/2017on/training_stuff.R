# =============================================================================
# Comprehensive Stuff+ Model Script - 2017-2022
# Trains models on previous 2 years, tests on target year
# Outputs combined CSV and all models to specified directory
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGE LOADING AND SETUP
# -----------------------------------------------------------------------------

# Install and load required packages
packages <- c("dplyr", "readr", "tidyr", "xgboost", "purrr", "lubridate", 
              "rBayesianOptimization", "caret")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages)
}

# Load packages
library(dplyr)
library(readr)
library(tidyr)
library(xgboost)
library(purrr)
library(lubridate)

# Install baseballr if needed
if(!"baseballr" %in% installed.packages()[,"Package"]) {
  if(!"remotes" %in% installed.packages()[,"Package"]) install.packages("remotes")
  remotes::install_github("BillPetti/baseballr")
}
library(baseballr)

# Try to load Bayesian optimization packages
bayesian_available <- FALSE
tryCatch({
  library(rBayesianOptimization)
  bayesian_available <- TRUE
  cat("Bayesian optimization available\n")
}, error = function(e) {
  cat("Bayesian optimization not available - using default parameters\n")
})

# Try to load caret
caret_available <- FALSE
tryCatch({
  library(caret)
  caret_available <- TRUE
  cat("caret package loaded\n")
}, error = function(e) {
  cat("caret package not available - using manual preprocessing\n")
})

# Set seed for reproducibility
set.seed(42)

# Set output directory
output_dir <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2017on/"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Comprehensive Stuff+ Model Script 2017-2022 ===\n")
cat("Output directory:", output_dir, "\n\n")

# -----------------------------------------------------------------------------
# 2. DATA COLLECTION FROM STATCAST (2015-2022)
# -----------------------------------------------------------------------------

cat("Fetching Statcast data from 2015-2022...\n")
cat("This will take a very long time - be patient!\n\n")

# Define date ranges for each year
years <- 2015:2022
all_data <- list()

for(year in years) {
  cat("Fetching", year, "data...\n")
  
  # Define season dates (approximate)
  season_start <- as.Date(paste0(year, "-03-20"))
  season_end <- as.Date(paste0(year, "-11-05"))
  date_vec <- seq(season_start, season_end, by = "1 day")
  
  year_data <- map_dfr(
    date_vec,
    function(d) {
      tryCatch({
        res <- statcast_search(
          start_date = d,
          end_date = d,
          player_type = "pitcher"
        )
        
        if (nrow(res) == 0) {
          return(NULL)
        }
        
        res
        
      }, error = function(e) {
        return(NULL)
      })
    }
  )
  
  all_data[[as.character(year)]] <- year_data
  cat("  ", year, "complete:", nrow(year_data), "rows\n")
  
  # Save yearly backup
  write_csv(year_data, paste0(output_dir, "statcast_", year, "_backup.csv"))
}

# Combine all years
df_all <- bind_rows(all_data)
cat("\nTotal data collected:", nrow(df_all), "rows\n")

# Save complete dataset backup
write_csv(df_all, paste0(output_dir, "statcast_2015_2022_complete.csv"))

# -----------------------------------------------------------------------------
# 3. LOAD RUN VALUES DATA
# -----------------------------------------------------------------------------

# Load run values (assuming this file exists)
run_values_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Data/run_values.csv"
if(!file.exists(run_values_path)) {
  stop("Run values file not found at: ", run_values_path)
}

df_run_values <- read_csv(run_values_path)
cat("Run values data loaded\n")

# -----------------------------------------------------------------------------
# 4. RUN VALUE ASSIGNMENT FUNCTION
# -----------------------------------------------------------------------------

assign_run_values <- function(df) {
  # Define pitch outcome groupings
  des_dict <- c(
    'Ball' = 'ball',
    'In play, run(s)' = 'hit_into_play',
    'In play, out(s)' = 'hit_into_play',
    'In play, no out' = 'hit_into_play',
    'Called Strike' = 'called_strike',
    'Foul' = 'foul',
    'Swinging Strike' = 'swinging_strike',
    'Blocked Ball' = 'ball',
    'Swinging Strike (Blocked)' = 'swinging_strike',
    'Foul Tip' = 'swinging_strike',
    'Foul Bunt' = 'foul',
    'Hit By Pitch' = 'hit_by_pitch',
    'Pitchout' = 'ball',
    'Missed Bunt' = 'swinging_strike',
    'Bunt Foul Tip' = 'swinging_strike',
    'Foul Pitchout' = 'foul',
    'Ball In Dirt' = 'ball'
  )
  
  # Remove existing delta_run_exp columns
  df <- df %>%
    select(-any_of(c("delta_run_exp", "delta_run_exp.x", "delta_run_exp.y", 
                     "delta_run_exp.x.x", "delta_run_exp.y.y")))
  
  # First join on events
  df <- df %>%
    left_join(df_run_values, by = c("events" = "event", "balls", "strikes"))
  
  # Map descriptions
  df <- df %>%
    mutate(description_mapped = ifelse(description %in% names(des_dict), 
                                       des_dict[description], description))
  
  # Second join on description
  df <- df %>%
    left_join(df_run_values, by = c("description_mapped" = "event", "balls", "strikes"), 
              suffix = c("_event", "_desc"))
  
  # Create target
  df <- df %>%
    mutate(target = coalesce(delta_run_exp_event, delta_run_exp_desc))
  
  return(df)
}

# -----------------------------------------------------------------------------
# 5. FEATURE ENGINEERING FUNCTION
# -----------------------------------------------------------------------------

feature_engineering <- function(df) {
  # Extract year
  df <- df %>%
    mutate(year = as.integer(substr(game_date, 1, 4)))
  
  # Convert pfx values and mirror for lefties
  df <- df %>%
    mutate(
      pfx_x_inches = pfx_x * 12,
      pfx_z_inches = pfx_z * 12
    ) %>%
    mutate(pfx_x_inches = ifelse(p_throws == 'L', -pfx_x_inches, pfx_x_inches))
  
  # Mirror horizontal release point for lefties
  df <- df %>%
    mutate(vx0 = ifelse(p_throws == 'L', release_pos_x, -release_pos_x))
  
  # Calculate arm slot
  df <- df %>%
    mutate(
      release_magnitude = sqrt(release_pos_x^2 + release_extension^2 + release_pos_z^2),
      cos_arm_angle = release_pos_z / release_magnitude,
      arm_angle_rad = acos(cos_arm_angle),
      arm_slot_Vector = arm_angle_rad * (180 / pi)
    )
  
  # Remove rows with NA target
  df <- df %>%
    filter(!is.na(target))
  
  # Calculate fastball averages
  pitch_types <- c('SI', 'FF', 'FC')
  
  df_filtered <- df %>%
    filter(pitch_type %in% pitch_types)
  
  df_agg <- df_filtered %>%
    group_by(pitcher, year, pitch_type) %>%
    summarise(
      avg_fastball_speed = mean(release_speed, na.rm = TRUE),
      avg_fastball_pfx_z = mean(pfx_z_inches, na.rm = TRUE),
      avg_fastball_pfx_x = mean(pfx_x_inches, na.rm = TRUE),
      count = n(),
      .groups = 'drop'
    ) %>%
    arrange(pitcher, year, desc(count), desc(avg_fastball_speed)) %>%
    group_by(pitcher, year) %>%
    slice_head(n = 1) %>%
    ungroup() %>%
    select(pitcher, year, avg_fastball_speed, avg_fastball_pfx_z, avg_fastball_pfx_x)
  
  # Join fastball averages
  df <- df %>%
    left_join(df_agg, by = c('pitcher', 'year'))
  
  # Fill missing values
  df <- df %>%
    group_by(pitcher) %>%
    mutate(
      avg_fastball_speed = ifelse(is.na(avg_fastball_speed), 
                                  ifelse(all(is.na(release_speed)), NA, max(release_speed, na.rm = TRUE)), 
                                  avg_fastball_speed),
      avg_fastball_pfx_z = ifelse(is.na(avg_fastball_pfx_z), 
                                  ifelse(all(is.na(pfx_z_inches)), NA, max(pfx_z_inches, na.rm = TRUE)), 
                                  avg_fastball_pfx_z),
      avg_fastball_pfx_x = ifelse(is.na(avg_fastball_pfx_x), 
                                  ifelse(all(is.na(pfx_x_inches)), NA, max(pfx_x_inches, na.rm = TRUE)), 
                                  avg_fastball_pfx_x)
    ) %>%
    ungroup()
  
  # Calculate differentials
  df <- df %>%
    mutate(
      speed_diff = release_speed - avg_fastball_speed,
      pfx_z_diff = pfx_z_inches - avg_fastball_pfx_z,
      pfx_x_diff = abs(pfx_x_inches - avg_fastball_pfx_x)
    )
  
  # Remove problematic records
  df <- df %>%
    filter(!is.infinite(avg_fastball_speed) & 
             !is.infinite(avg_fastball_pfx_z) & 
             !is.infinite(avg_fastball_pfx_x) &
             !is.na(avg_fastball_speed) &
             !is.na(avg_fastball_pfx_z) &
             !is.na(avg_fastball_pfx_x))
  
  return(df)
}

# -----------------------------------------------------------------------------
# 6. BAYESIAN OPTIMIZATION FUNCTION
# -----------------------------------------------------------------------------

run_bayesian_optimization <- function(xgb_train, features) {
  if(!bayesian_available) {
    # Return default parameters if Bayesian optimization not available
    return(list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = 6,
      min_child_weight = 50,
      colsample_bytree = 0.8,
      subsample = 0.8,
      alpha = 0.1,
      lambda = 0.1,
      eta = 0.05,
      seed = 42
    ))
  }
  
  # Bayesian optimization objective function
  xgb_bayesian_opt <- function(max_depth, min_child_weight,
                               colsample_bytree, subsample,
                               alpha, lambda, eta) {
    
    max_depth <- round(max_depth)
    min_child_weight <- round(min_child_weight)
    
    params <- list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      colsample_bytree = colsample_bytree,
      subsample = subsample,
      alpha = alpha,
      lambda = lambda,
      eta = eta,
      seed = 42
    )
    
    tryCatch({
      cv <- xgb.cv(
        params = params,
        data = xgb_train,
        nfold = 5,
        nrounds = 1000,
        early_stopping_rounds = 50,
        verbose = 0
      )
      
      rmse <- min(cv$evaluation_log$test_rmse_mean)
      return(list(Score = -rmse, Pred = 0))
      
    }, error = function(e) {
      return(list(Score = -999999, Pred = 0))
    })
  }
  
  # Parameter bounds
  bounds <- list(
    max_depth = c(5L, 15L),
    min_child_weight = c(10L, 100L),
    colsample_bytree = c(0.6, 1.0),
    subsample = c(0.6, 1.0),
    alpha = c(0.0, 1.0),
    lambda = c(0.0, 1.0),
    eta = c(0.01, 0.1)
  )
  
  # Run optimization
  opt_result <- BayesianOptimization(
    FUN = xgb_bayesian_opt,
    bounds = bounds,
    init_points = 5,
    n_iter = 10,
    acq = "ucb",
    kappa = 2.576,
    verbose = FALSE
  )
  
  # Return best parameters
  best_params <- opt_result$Best_Par
  
  final_params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = round(best_params[["max_depth"]]),
    min_child_weight = round(best_params[["min_child_weight"]]),
    colsample_bytree = best_params[["colsample_bytree"]],
    subsample = best_params[["subsample"]],
    alpha = best_params[["alpha"]],
    lambda = best_params[["lambda"]],
    eta = best_params[["eta"]],
    seed = 42
  )
  
  return(final_params)
}

# -----------------------------------------------------------------------------
# 7. PREPROCESSING FUNCTION
# -----------------------------------------------------------------------------

preprocess_data <- function(X_train, X_test = NULL) {
  if(caret_available) {
    preprocess_params <- preProcess(X_train, method = c("center", "scale"))
    X_train_scaled <- predict(preprocess_params, X_train)
    
    if(!is.null(X_test)) {
      X_test_scaled <- predict(preprocess_params, X_test)
      return(list(
        X_train_scaled = X_train_scaled,
        X_test_scaled = X_test_scaled,
        preprocess_params = preprocess_params
      ))
    } else {
      return(list(
        X_train_scaled = X_train_scaled,
        preprocess_params = preprocess_params
      ))
    }
  } else {
    # Manual scaling
    means <- sapply(X_train, mean, na.rm = TRUE)
    sds <- sapply(X_train, sd, na.rm = TRUE)
    
    X_train_scaled <- scale(X_train, center = means, scale = sds)
    X_train_scaled <- as.data.frame(X_train_scaled)
    
    preprocess_params <- list(means = means, sds = sds)
    
    if(!is.null(X_test)) {
      X_test_scaled <- scale(X_test, center = means, scale = sds)
      X_test_scaled <- as.data.frame(X_test_scaled)
      
      return(list(
        X_train_scaled = X_train_scaled,
        X_test_scaled = X_test_scaled,
        preprocess_params = preprocess_params
      ))
    } else {
      return(list(
        X_train_scaled = X_train_scaled,
        preprocess_params = preprocess_params
      ))
    }
  }
}

# -----------------------------------------------------------------------------
# 8. MAIN PROCESSING: APPLY RUN VALUES AND FEATURE ENGINEERING
# -----------------------------------------------------------------------------

cat("Applying run value assignment and feature engineering...\n")

# Process the data
df_processed <- assign_run_values(df_all)
df_processed <- feature_engineering(df_processed)

cat("Data processing complete\n")
cat("Final dataset:", nrow(df_processed), "rows\n\n")

# Define features
features <- c('release_speed', 'release_spin_rate', 'release_extension',
              'release_pos_x', 'release_pos_z', 'pfx_z_inches', 'pfx_x_inches',
              'arm_slot_Vector', 'speed_diff', 'pfx_z_diff', 'pfx_x_diff')

target <- 'target'

# -----------------------------------------------------------------------------
# 9. MODEL TRAINING AND TESTING LOOP
# -----------------------------------------------------------------------------

# Define the years to model
model_years <- 2017:2022
all_test_results <- list()

for(test_year in model_years) {
  
  cat("\n", rep("=", 60), "\n")
  cat("TRAINING MODEL FOR YEAR:", test_year, "\n")
  cat("Training years:", (test_year-2), "-", (test_year-1), "\n")
  cat(rep("=", 60), "\n")
  
  # Prepare training data (previous 2 years)
  train_years <- c(test_year - 2, test_year - 1)
  df_train <- df_processed %>%
    filter(year %in% train_years) %>%
    drop_na(all_of(c(features, target)))
  
  # Prepare test data (target year)
  df_test <- df_processed %>%
    filter(year == test_year) %>%
    drop_na(all_of(c(features, target)))
  
  cat("Training observations:", nrow(df_train), "\n")
  cat("Test observations:", nrow(df_test), "\n")
  
  if(nrow(df_train) == 0 || nrow(df_test) == 0) {
    cat("Insufficient data for year", test_year, "- skipping\n")
    next
  }
  
  # Extract features and targets
  X_train <- df_train[features]
  y_train <- df_train[[target]]
  X_test <- df_test[features]
  y_test <- df_test[[target]]
  
  # Preprocess data
  cat("Preprocessing data...\n")
  preprocess_result <- preprocess_data(X_train, X_test)
  X_train_scaled <- preprocess_result$X_train_scaled
  X_test_scaled <- preprocess_result$X_test_scaled
  preprocess_params <- preprocess_result$preprocess_params
  
  # Convert to matrices
  X_train_matrix <- as.matrix(X_train_scaled)
  X_test_matrix <- as.matrix(X_test_scaled)
  
  # Create XGBoost datasets
  xgb_train <- xgb.DMatrix(data = X_train_matrix, label = y_train)
  xgb_test <- xgb.DMatrix(data = X_test_matrix, label = y_test)
  
  # Run Bayesian optimization
  cat("Running parameter optimization...\n")
  optimal_params <- run_bayesian_optimization(xgb_train, features)
  
  # Train final model
  cat("Training final model...\n")
  model <- xgb.train(
    params = optimal_params,
    data = xgb_train,
    nrounds = 1000,
    verbose = 0
  )
  
  # Make predictions
  cat("Making predictions...\n")
  predictions <- predict(model, X_test_matrix)
  
  # Add predictions to test data
  df_test <- df_test %>%
    mutate(predicted_target = predictions)
  
  # Calculate Stuff+ for this year
  m_year <- mean(df_test$predicted_target, na.rm = TRUE)
  s_year <- sd(df_test$predicted_target, na.rm = TRUE)
  
  df_test <- df_test %>%
    mutate(
      target_zscore = (predicted_target - m_year) / s_year,
      stuff_plus = 100 - target_zscore * 10
    )
  
  # Calculate evaluation metrics
  actual <- df_test$target
  predicted <- df_test$predicted_target
  
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  r_squared <- 1 - (sum((actual - predicted)^2) / sum((actual - mean(actual))^2))
  
  cat("RMSE:", round(rmse, 4), "\n")
  cat("R-squared:", round(r_squared, 4), "\n")
  
  # Save model and preprocessing parameters
  model_filename <- paste0(output_dir, "xgboost_model_", test_year, ".txt")
  preprocess_filename <- paste0(output_dir, "preprocess_params_", test_year, ".rds")
  params_filename <- paste0(output_dir, "model_params_", test_year, ".rds")
  
  xgb.save(model, model_filename)
  saveRDS(preprocess_params, preprocess_filename)
  saveRDS(optimal_params, params_filename)
  
  # Save scaling parameters
  scaling_params <- data.frame(
    year = test_year,
    mean_predicted_target = m_year,
    sd_predicted_target = s_year,
    n_pitches = nrow(df_test),
    rmse = rmse,
    r_squared = r_squared
  )
  
  scaling_filename <- paste0(output_dir, "scaling_params_", test_year, ".csv")
  write_csv(scaling_params, scaling_filename)
  
  cat("Model files saved for", test_year, "\n")
  
  # Store test results
  all_test_results[[as.character(test_year)]] <- df_test
  
  cat("Year", test_year, "complete!\n")
}

# -----------------------------------------------------------------------------
# 10. COMBINE ALL RESULTS AND SAVE
# -----------------------------------------------------------------------------

cat("\n", rep("=", 60), "\n")
cat("COMBINING ALL RESULTS\n")
cat(rep("=", 60), "\n")

# Combine all test results
df_combined <- bind_rows(all_test_results)
cat("Combined results:", nrow(df_combined), "rows\n")
cat("Years included:", paste(sort(unique(df_combined$year)), collapse = ", "), "\n")

# Save the large combined CSV
combined_filename <- paste0(output_dir, "test_results_combined_2017_2022.csv")
write_csv(df_combined, combined_filename)
cat("Combined results saved to:", combined_filename, "\n")

# Create summary statistics
summary_stats <- df_combined %>%
  group_by(year) %>%
  summarise(
    n_pitches = n(),
    mean_stuff_plus = mean(stuff_plus, na.rm = TRUE),
    sd_stuff_plus = sd(stuff_plus, na.rm = TRUE),
    mean_predicted_target = mean(predicted_target, na.rm = TRUE),
    sd_predicted_target = sd(predicted_target, na.rm = TRUE),
    rmse = sqrt(mean((target - predicted_target)^2)),
    r_squared = 1 - (sum((target - predicted_target)^2) / sum((target - mean(target))^2)),
    .groups = 'drop'
  )

summary_filename <- paste0(output_dir, "summary_statistics_2017_2022.csv")
write_csv(summary_stats, summary_filename)
cat("Summary statistics saved to:", summary_filename, "\n")

# -----------------------------------------------------------------------------
# 11. FINAL SUMMARY
# -----------------------------------------------------------------------------

cat("\n", rep("=", 80), "\n")
cat("COMPREHENSIVE STUFF+ MODELING COMPLETE\n")
cat(rep("=", 80), "\n\n")

cat("MODELS CREATED:\n")
for(year in model_years) {
  if(as.character(year) %in% names(all_test_results)) {
    cat("- ", year, ": trained on ", (year-2), "-", (year-1), ", tested on ", year, "\n", sep = "")
  }
}

cat("\nFILES SAVED TO:", output_dir, "\n")
cat("- Combined test results: test_results_combined_2017_2022.csv\n")
cat("- Summary statistics: summary_statistics_2017_2022.csv\n")
cat("- Individual models: xgboost_model_YYYY.txt\n")
cat("- Preprocessing params: preprocess_params_YYYY.rds\n")
cat("- Model parameters: model_params_YYYY.rds\n")
cat("- Scaling parameters: scaling_params_YYYY.csv\n")
cat("- Data backups: statcast_YYYY_backup.csv\n")

cat("\nSUMMARY STATISTICS:\n")
print(summary_stats)

cat("\n", rep("=", 80), "\n")
cat("SCRIPT EXECUTION COMPLETE\n")
cat(rep("=", 80), "\n")