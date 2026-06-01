# =============================================================================
# Stuff+ Model in R - Run Value Prediction - xgboost regression
# Training ONLY on 2022-2024 Data
# Modified to use pfx_x and pfx_z (in inches) instead of ax and az
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGE LOADING AND SETUP
# -----------------------------------------------------------------------------

# Install and load required packages
packages <- c("dplyr", "readr", "tidyr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(dplyr)
library(readr)
library(tidyr)

# Set seed for reproducibility
set.seed(42)

# Set output directory
output_dir <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2025/Model Development/"

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------

# Load the MLB pitch data
file_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Data/2020_24Training.csv"
df <- read_csv(file_path)

# Load run values data
run_values_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Data/run_values.csv"
df_run_values <- read_csv(run_values_path)

# -----------------------------------------------------------------------------
# 3. RUN VALUE ASSIGNMENT
# -----------------------------------------------------------------------------

# Define a dictionary to group pitch outcomes together
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

# Remove any existing delta_run_exp columns if they exist
df <- df %>%
  select(-any_of(c("delta_run_exp", "delta_run_exp.x", "delta_run_exp.y", "delta_run_exp.x.x", "delta_run_exp.y.y")))

# First join: Try to get run values based on events (like "single", "strikeout", etc.)
df <- df %>%
  left_join(df_run_values, by = c("events" = "event", "balls", "strikes"))

# Replace play descriptions with the grouped outcomes from des_dict
df <- df %>%
  mutate(description_mapped = ifelse(description %in% names(des_dict), des_dict[description], description))

# Second join: Try to get run values based on description (like "ball", "called_strike", etc.)
df <- df %>%
  left_join(df_run_values, by = c("description_mapped" = "event", "balls", "strikes"), suffix = c("_event", "_desc"))

# Create target: Use events-based run value if available, otherwise use description-based
df <- df %>%
  mutate(target = coalesce(delta_run_exp_event, delta_run_exp_desc))

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

feature_engineering <- function(df) {
  # Extract the year from the game_date column
  df <- df %>%
    mutate(year = as.integer(substr(game_date, 1, 4)))
  
  # Convert pfx values from feet to inches and mirror horizontal movement for left-handed pitchers
  df <- df %>%
    mutate(
      pfx_x_inches = pfx_x * 12,  # Convert feet to inches
      pfx_z_inches = pfx_z * 12   # Convert feet to inches
    ) %>%
    mutate(pfx_x_inches = ifelse(p_throws == 'L', -pfx_x_inches, pfx_x_inches))
  
  # Mirror horizontal release point for left-handed pitchers
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
  
  # Define pitch types for fastball averages
  pitch_types <- c('SI', 'FF', 'FC')
  
  # Calculate fastball averages
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
  
  # Fill missing fastball values
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
  
  # Calculate pitch differentials
  df <- df %>%
    mutate(
      speed_diff = release_speed - avg_fastball_speed,
      pfx_z_diff = pfx_z_inches - avg_fastball_pfx_z,
      pfx_x_diff = abs(pfx_x_inches - avg_fastball_pfx_x)
    )
  
  # Remove problematic records with infinite or NA fastball values
  df <- df %>%
    filter(!is.infinite(avg_fastball_speed) & 
             !is.infinite(avg_fastball_pfx_z) & 
             !is.infinite(avg_fastball_pfx_x) &
             !is.na(avg_fastball_speed) &
             !is.na(avg_fastball_pfx_z) &
             !is.na(avg_fastball_pfx_x))
  
  return(df)
}

# Apply feature engineering
df_processed <- feature_engineering(df)

# -----------------------------------------------------------------------------
# 5. MODEL TRAINING PREPARATION
# -----------------------------------------------------------------------------

# Filter the dataframe to include only the years 2022, 2023, and 2024
df_train <- df_processed %>%
  filter(year %in% c(2022, 2023, 2024))

# Define the features to be used for training
features <- c('release_speed',
              'release_spin_rate', 
              'release_extension',
              'release_pos_x',
              'release_pos_z',
              'pfx_z_inches',
              'pfx_x_inches',
              'arm_slot_Vector',
              'speed_diff',
              'pfx_z_diff',
              'pfx_x_diff')

# Define the target variable
target <- 'target'

# Drop rows with null values in the specified features and target column
df_train <- df_train %>%
  drop_na(all_of(c(features, target)))

cat("Training data prepared:\n")
cat("Training years: 2022-2024\n")
cat("Training observations:", nrow(df_train), "\n")
cat("Features:", length(features), "\n")
cat("Target variable:", target, "\n")

# =============================================================================
# Bayesian Optimization for xgboost Run Value Model
# =============================================================================

# -----------------------------------------------------------------------------
# 6. INSTALL AND LOAD BAYESIAN OPTIMIZATION PACKAGES 
# -----------------------------------------------------------------------------

# Install required packages with proper error handling
bayesian_packages <- c("rBayesianOptimization", "xgboost", "caret", "foreach", "iterators", "dplyr")
new_bayesian_packages <- bayesian_packages[!(bayesian_packages %in% installed.packages()[,"Package"])]

if(length(new_bayesian_packages)) {
  cat("Installing missing packages:", paste(new_bayesian_packages, collapse = ", "), "\n")
  
  # Install packages one by one with error handling
  for(pkg in new_bayesian_packages) {
    tryCatch({
      install.packages(pkg, dependencies = TRUE)
      cat("Successfully installed:", pkg, "\n")
    }, error = function(e) {
      cat("Failed to install", pkg, "- Error:", e$message, "\n")
    })
  }
}

# Load packages with error handling
required_packages <- c("rBayesianOptimization", "xgboost", "dplyr")

for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    stop(paste("Package", pkg, "is required but not available. Please install it manually."))
  }
}

# Try to load caret, but provide fallback if it fails
caret_available <- FALSE
tryCatch({
  library(caret)
  caret_available <- TRUE
  cat("caret package loaded successfully\n")
}, error = function(e) {
  cat("caret package failed to load. Using manual data splitting instead.\n")
  cat("Error:", e$message, "\n")
})

# Set seed for reproducibility
set.seed(42)

# -----------------------------------------------------------------------------
# 7. PREPARE DATA FOR BAYESIAN OPTIMIZATION (WITH FALLBACK)
# -----------------------------------------------------------------------------

cat("Using training data from 2022-2024 for Bayesian optimization\n")
cat("Training observations:", nrow(df_train), "\n")

# Create train/validation split - with fallback if caret is not available
if(caret_available) {
  # Use caret for data partitioning
  train_indices <- createDataPartition(df_train[[target]], p = 0.8, list = FALSE)
  train_data <- df_train[train_indices, ]
  val_data <- df_train[-train_indices, ]
} else {
  # Manual data splitting fallback
  n <- nrow(df_train)
  train_size <- floor(0.8 * n)
  train_indices <- sample(seq_len(n), size = train_size)
  train_data <- df_train[train_indices, ]
  val_data <- df_train[-train_indices, ]
}

# Extract features and targets
X_train <- train_data[features]
y_train <- train_data[[target]]
X_val <- val_data[features]
y_val <- val_data[[target]]

# Scale features - with fallback if caret is not available
if(caret_available) {
  # Use caret preprocessing
  preprocess_params <- preProcess(X_train, method = c("center", "scale"))
  X_train_scaled <- predict(preprocess_params, X_train)
  X_val_scaled <- predict(preprocess_params, X_val)
} else {
  # Manual scaling fallback
  cat("Using manual scaling (center and scale)\n")
  
  # Calculate scaling parameters
  means <- sapply(X_train, mean, na.rm = TRUE)
  sds <- sapply(X_train, sd, na.rm = TRUE)
  
  # Apply scaling
  X_train_scaled <- scale(X_train, center = means, scale = sds)
  X_val_scaled <- scale(X_val, center = means, scale = sds)
  
  # Convert back to data frame
  X_train_scaled <- as.data.frame(X_train_scaled)
  X_val_scaled <- as.data.frame(X_val_scaled)
  
  # Store parameters for later use
  preprocess_params <- list(means = means, sds = sds)
}

# Convert to matrices
X_train_matrix <- as.matrix(X_train_scaled)
X_val_matrix <- as.matrix(X_val_scaled)

# Create XGBoost matrices
xgb_train <- xgb.DMatrix(data = X_train_matrix, label = y_train)
xgb_val <- xgb.DMatrix(data = X_val_matrix, label = y_val)

# -----------------------------------------------------------------------------
# 8. DEFINE BAYESIAN OPTIMIZATION OBJECTIVE FUNCTION
# -----------------------------------------------------------------------------

xgb_bayesian_opt <- function(max_depth, min_child_weight,
                             colsample_bytree, subsample,
                             alpha, lambda, eta) {
  
  # Convert parameters to appropriate types and ranges
  max_depth <- round(max_depth)
  min_child_weight <- round(min_child_weight)
  
  # Set up parameters
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
  
  # Evaluate parameters using cross-validation
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
    
    # Return negative RMSE (since BayesianOptimization maximizes the score)
    return(list(Score = -rmse, Pred = 0))
    
  }, error = function(e) {
    # Return a very bad score if model training fails
    cat("Model training failed with parameters:", paste(names(params), params, sep="=", collapse=", "), "\n")
    return(list(Score = -999999, Pred = 0))
  })
}

# -----------------------------------------------------------------------------
# 9. DEFINE PARAMETER BOUNDS FOR OPTIMIZATION
# -----------------------------------------------------------------------------

# Define bounds for each parameter
bounds <- list(
  max_depth = c(5L, 15L),              # Tree depth
  min_child_weight = c(10L, 100L),     # Minimum samples per leaf
  colsample_bytree = c(0.6, 1.0),      # Feature sampling
  subsample = c(0.6, 1.0),             # Row sampling
  alpha = c(0.0, 1.0),                 # L1 regularization
  lambda = c(0.0, 1.0),                # L2 regularization
  eta = c(0.01, 0.1)                   # Learning rate
)

# -----------------------------------------------------------------------------
# 10. RUN BAYESIAN OPTIMIZATION
# -----------------------------------------------------------------------------

cat("Starting Bayesian Optimization...\n")
cat("This may take a while.\n\n")

# Run Bayesian optimization with error handling
tryCatch({
  opt_result <- BayesianOptimization(
    FUN = xgb_bayesian_opt,
    bounds = bounds,
    init_points = 5,      # Initial random evaluations
    n_iter = 10,          # Number of optimization iterations
    acq = "ucb",          # Acquisition function (Upper Confidence Bound)
    kappa = 2.576,        # Exploration parameter
    verbose = TRUE
  )
}, error = function(e) {
  stop("Bayesian optimization failed: ", e$message)
})

# -----------------------------------------------------------------------------
# 11. EXTRACT BEST PARAMETERS AND TRAIN FINAL MODEL
# -----------------------------------------------------------------------------

# Get best parameters
best_params <- opt_result$Best_Par
cat("\n", rep("=", 50), "\n")
cat("BEST PARAMETERS FOUND:\n")
print(best_params)
cat("Best RMSE:", -opt_result$Best_Value, "\n")

# Convert to proper format for XGBoost
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

# -----------------------------------------------------------------------------
# 12. TRAIN FINAL MODEL ON FULL TRAINING DATA (2022-2024)
# -----------------------------------------------------------------------------

# Use your existing df_train (already filtered to 2022-2024)
X_full <- df_train[features]
y_full <- df_train[[target]]

# Scale using the same approach as above
if(caret_available) {
  preprocess_params_final <- preProcess(X_full, method = c("center", "scale"))
  X_full_scaled <- predict(preprocess_params_final, X_full)
} else {
  # Manual scaling
  means_final <- sapply(X_full, mean, na.rm = TRUE)
  sds_final <- sapply(X_full, sd, na.rm = TRUE)
  X_full_scaled <- scale(X_full, center = means_final, scale = sds_final)
  X_full_scaled <- as.data.frame(X_full_scaled)
  preprocess_params_final <- list(means = means_final, sds = sds_final)
}

X_full_matrix <- as.matrix(X_full_scaled)

# Create XGBoost dataset
xgb_full_train <- xgb.DMatrix(data = X_full_matrix, label = y_full)

# Train final model
cat("\nTraining final model with optimized parameters...\n")
final_model <- xgb.train(
  params = final_params,
  data = xgb_full_train,
  nrounds = 1000,
  verbose = 1
)

# -----------------------------------------------------------------------------
# 13. SAVE OPTIMIZED MODEL AND PARAMETERS
# -----------------------------------------------------------------------------

# Save the optimized model
xgb.save(final_model, paste0(output_dir, "xgboost_regression_model_runvalue_bayesian_optimized_2022_2024_pfx.txt"))
cat("Optimized model saved to", paste0(output_dir, "xgboost_regression_model_runvalue_bayesian_optimized_2022_2024_pfx.txt"), "\n")

# Save preprocessing parameters
saveRDS(preprocess_params_final, paste0(output_dir, "preprocess_params_xgboost_regression_runvalue_optimized_pfx.rds"))
cat("Preprocess parameters saved to", paste0(output_dir, "preprocess_params_xgboost_regression_runvalue_optimized_pfx.rds"), "\n")

# Save the best parameters for reference
saveRDS(final_params, paste0(output_dir, "best_xgboost_regression_params_runvalue_pfx.rds"))
cat("Best parameters saved to", paste0(output_dir, "best_xgboost_regression_params_runvalue_pfx.rds"), "\n")

# -----------------------------------------------------------------------------
# 14. DISPLAY RESULTS AND FEATURE IMPORTANCE
# -----------------------------------------------------------------------------

# Show feature importance
cat("\n", rep("=", 50), "\n")
cat("FEATURE IMPORTANCE (optimized model):\n")
importance <- xgb.importance(model = final_model)
print(importance)

# Display final parameters
cat("\n", rep("=", 50), "\n")
cat("FINAL OPTIMIZED PARAMETERS:\n")
for(param_name in names(final_params)) {
  if(param_name %in% c("objective", "metric", "seed", "verbose")) next
  cat(sprintf("%s: %s\n", param_name, final_params[[param_name]]))
}

cat("\nBayesian optimization complete\n")

# -----------------------------------------------------------------------------
# 15. TRAINING COMPLETE SUMMARY
# -----------------------------------------------------------------------------

cat("\n", rep("=", 80), "\n")
cat("XGBOOST REGRESSION RUN VALUE STUFF+ MODEL TRAINING COMPLETE\n")
cat(rep("=", 80), "\n\n")

cat("TRAINING SUMMARY:\n")
cat("- Training years: 2022-2024\n")
cat("- Movement variables: pfx_x and pfx_z (converted to inches)\n")
cat("- Training observations:", nrow(df_train), "\n")
cat("- Best CV RMSE:", round(-opt_result$Best_Value, 4), "\n\n")

cat("MODEL FILES SAVED TO:", output_dir, "\n")
cat("- Model: xgboost_regression_model_runvalue_bayesian_optimized_2022_2024_pfx.txt\n")
cat("- Preprocessing parameters: preprocess_params_xgboost_regression_runvalue_optimized_pfx.rds\n")
cat("- Best parameters: best_xgboost_regression_params_runvalue_pfx.rds\n\n")

cat("READY FOR TESTING!\n")
cat("Use the separate testing script to apply this model to 2025 data.\n")
cat(rep("=", 80), "\n")