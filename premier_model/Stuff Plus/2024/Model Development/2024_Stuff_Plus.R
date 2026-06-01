# =============================================================================
# Stuff+ Model in R - Run Value Prediction - xgboost regression
# Training on 2021-2023, Testing on 2024 Only
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
output_dir <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Aaron-David/Regression Model/2024/"

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------

# Load the MLB pitch data
file_path <- r'(C:\Users\aasmi\p3_summer_2025\Pitch Modeling Aaron-David\Get Statcast\2020_24Training.csv)'
df <- read_csv(file_path)

# Load run values data
run_values_path <- r'(C:\Users\aasmi\p3_summer_2025\Pitch Modeling Aaron-David\Get Statcast\run_values.csv)'
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

# Filter the dataframe to include only the years 2021, 2022, and 2023
df_train <- df_processed %>%
  filter(year %in% c(2021, 2022, 2023))

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
cat("Training years: 2021-2023\n")
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

cat("Using training data from 2021-2023 for Bayesian optimization\n")
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
# 12. TRAIN FINAL MODEL ON FULL TRAINING DATA (2021-2023)
# -----------------------------------------------------------------------------

# Use your existing df_train (already filtered to 2021-2023)
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
xgb.save(final_model, paste0(output_dir, "xgboost_regression_model_runvalue_bayesian_optimized_2021_2023_pfx.txt"))
cat("Optimized model saved to", paste0(output_dir, "xgboost_regression_model_runvalue_bayesian_optimized_2021_2023_pfx.txt"), "\n")

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
# 15. TESTING AND PREDICTION PHASE - 2024 ONLY
# -----------------------------------------------------------------------------

# Filter the dataframe to include only 2024 data
df_test <- df_processed %>%
  filter(year == 2024) %>%
  drop_na(all_of(c(features, target)))

cat("Test data prepared:\n")
cat("Test year: 2024\n")
cat("Test observations:", nrow(df_test), "\n")

# Prepare test features using the same preprocessing as training
X_test <- df_test[features]

# Apply the same scaling that was used during training
if(caret_available) {
  X_test_scaled <- predict(preprocess_params_final, X_test)
} else {
  # Manual scaling using the same parameters from training
  X_test_scaled <- scale(X_test, center = preprocess_params_final$means, scale = preprocess_params_final$sds)
  X_test_scaled <- as.data.frame(X_test_scaled)
}

# Convert to matrix for prediction
X_test_matrix <- as.matrix(X_test_scaled)

# Predict the target values for 2024 data using the trained model
predictions <- predict(final_model, X_test_matrix)

# Add predictions to the test dataframe
df_test <- df_test %>%
  mutate(predicted_target = predictions)

cat("Predictions completed\n")

# -----------------------------------------------------------------------------
# 16. PITCH COLOURS AND VISUALIZATION SETUP
# -----------------------------------------------------------------------------

# For help with plotting the pitch data, we will use the following list to map 
pitch_colours <- list(
  ## Fastballs ##
  'FF' = list(colour = '#FF007D', name = '4-Seam Fastball'),
  'FA' = list(colour = '#FF007D', name = 'Fastball'),
  'SI' = list(colour = '#98165D', name = 'Sinker'),
  'FC' = list(colour = '#BE5FA0', name = 'Cutter'),
  
  ## Offspeed ##
  'CH' = list(colour = '#F79E70', name = 'Changeup'),
  'FS' = list(colour = '#FE6100', name = 'Splitter'),
  'SC' = list(colour = '#F08223', name = 'Screwball'),
  'FO' = list(colour = '#FFB000', name = 'Forkball'),
  
  ## Sliders ##
  'SL' = list(colour = '#67E18D', name = 'Slider'),
  'ST' = list(colour = '#1BB999', name = 'Sweeper'),
  'SV' = list(colour = '#376748', name = 'Slurve'),
  
  ## Curveballs ##
  'KC' = list(colour = '#311D8B', name = 'Knuckle Curve'),
  'CU' = list(colour = '#3025CE', name = 'Curveball'),
  'CS' = list(colour = '#274BFC', name = 'Slow Curve'),
  'EP' = list(colour = '#648FFF', name = 'Eephus'),
  
  ## Others ##
  'KN' = list(colour = '#867A08', name = 'Knuckleball'),
  'PO' = list(colour = '#472C30', name = 'Pitch Out'),
  'UN' = list(colour = '#9C8975', name = 'Unknown')
)

# Create dictionaries (named vectors in R) mapping pitch types to their colors
dict_colour <- sapply(pitch_colours, function(x) x$colour)
names(dict_colour) <- names(pitch_colours)

# Create a dictionary mapping pitch types to their names
dict_pitch <- sapply(pitch_colours, function(x) x$name)
names(dict_pitch) <- names(pitch_colours)

# Create a dictionary mapping pitch names back to pitch types
dict_pitch_desc_type <- sapply(pitch_colours, function(x) x$name)
names(dict_pitch_desc_type) <- sapply(pitch_colours, function(x) x$name)
dict_pitch_desc_type <- setNames(names(pitch_colours), sapply(pitch_colours, function(x) x$name))

# Create a dictionary mapping pitch names to their colors
dict_pitch_name <- sapply(pitch_colours, function(x) x$colour)
names(dict_pitch_name) <- sapply(pitch_colours, function(x) x$name)

# -----------------------------------------------------------------------------
# 17. BASIC EVALUATION METRICS
# -----------------------------------------------------------------------------

# Calculate basic evaluation metrics
actual_target <- df_test$target
predicted_target <- df_test$predicted_target

# Calculate RMSE
rmse <- sqrt(mean((actual_target - predicted_target)^2))

# Calculate MAE
mae <- mean(abs(actual_target - predicted_target))

# Calculate R-squared
ss_res <- sum((actual_target - predicted_target)^2)
ss_tot <- sum((actual_target - mean(actual_target))^2)
r_squared <- 1 - (ss_res / ss_tot)

# Display evaluation metrics
cat("\n", rep("=", 50), "\n")
cat("MODEL EVALUATION METRICS (2024 TEST DATA):\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAE:", round(mae, 4), "\n")
cat("R-squared:", round(r_squared, 4), "\n")

# Display summary statistics
cat("\nACTUAL RUN VALUE SUMMARY:\n")
print(summary(actual_target))
cat("\nPREDICTED RUN VALUE SUMMARY:\n")
print(summary(predicted_target))

# -----------------------------------------------------------------------------
# 18. SAVE TEST RESULTS
# -----------------------------------------------------------------------------

# Save the test results
write_csv(df_test, paste0(output_dir, "test_results_xgboost_regression_runvalue_2024_pfx.csv"))
cat("\nTest results saved to", paste0(output_dir, "test_results_xgboost_regression_runvalue_2024_pfx.csv"), "\n")

# Save evaluation metrics
eval_metrics <- data.frame(
  metric = c("RMSE", "MAE", "R_squared"),
  value = c(rmse, mae, r_squared)
)
write_csv(eval_metrics, paste0(output_dir, "xgboost_regression_evaluation_metrics_runvalue_pfx.csv"))
cat("Evaluation metrics saved to", paste0(output_dir, "xgboost_regression_evaluation_metrics_runvalue_pfx.csv"), "\n")

cat("\nTesting and prediction phase complete\n")

# -----------------------------------------------------------------------------
# 19. STUFF+ CALCULATION (2024 only)
# -----------------------------------------------------------------------------

# Calculate mean and SD for 2024
m_2024 <- mean(df_test$predicted_target, na.rm = TRUE)
s_2024 <- sd(df_test$predicted_target, na.rm = TRUE)

# Add Stuff+ to test data
df_test <- df_test %>%
  mutate(
    target_zscore = (predicted_target - m_2024) / s_2024,
    stuff_plus = 100 - target_zscore * 10
  )

# Aggregate by pitcher
df_agg_2024 <- df_test %>%
  group_by(pitcher, year) %>%
  summarise(
    count = n(),
    stuff_plus = mean(stuff_plus, na.rm = TRUE),
    .groups = "drop"
  )

# -----------------------------------------------------------------------------
# 20. FANGRAPHS DATA FETCH FOR 2024
# -----------------------------------------------------------------------------

# Load required packages for API calls
if(!require(httr, quietly = TRUE)) install.packages("httr")
if(!require(jsonlite, quietly = TRUE)) install.packages("jsonlite")
library(httr)
library(jsonlite)

# Fetch data from Fangraphs API for the 2024 MLB season
url <- "https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season=2024&season1=2024&ind=1&qual=0&type=8&month=0&pageitems=500000"

response <- GET(url)
data <- fromJSON(content(response, "text"))

# Create a data frame from the fetched data
df_fg <- data.frame(data$data)

# Convert column types to match the schema
df_fg <- df_fg %>%
  mutate(
    playerid = as.integer(playerid),
    xMLBAMID = as.integer(xMLBAMID),
    PlayerName = as.character(PlayerName),
    Season = as.integer(Season),
    Team = as.character(Team),
    G = as.integer(G),
    IP = as.numeric(IP),
    `K.BB.` = as.numeric(`K.BB.`),
    ERA = as.numeric(ERA),
    FIP = as.numeric(FIP),
    xFIP = as.numeric(xFIP),
    TBF = as.integer(TBF),
    Pitches = as.integer(Pitches)
  )

# Load wOBA data from a CSV file
df_woba <- read_csv(r'(C:\Users\aasmi\p3_summer_2025\Pitch Modeling Aaron-David\Get Statcast\woba_2020_2024.csv)')

# Join the Fangraphs data with the wOBA data on player ID and season
df_fg <- df_fg %>%
  left_join(df_woba, by = c("xMLBAMID" = "player_id", "Season" = "year"))

# -----------------------------------------------------------------------------
# 21. STUFF+ BY PITCH TYPE FOR 2024
# -----------------------------------------------------------------------------

# Create pitch type dataframe for 2024
final_2024 <- df_test %>%
  group_by(pitcher, pitch_type) %>%
  summarise(stuff_plus_pp = mean(stuff_plus, na.rm = TRUE), .groups = 'drop') %>%
  filter(pitch_type %in% c('FF', 'SI', 'FC', 'CH', 'SL', 'CU', 'KC', 'ST')) %>%
  pivot_wider(names_from = pitch_type, values_from = stuff_plus_pp, names_prefix = "stuff_plus_") %>%
  left_join(df_agg_2024 %>% select(pitcher, stuff_plus), by = "pitcher") %>%
  left_join(df_fg %>% filter(Season == 2024) %>% 
              select(xMLBAMID, PlayerName, Team, IP, ERA, FIP, xFIP, `K.BB.`, hard_hit_percent, woba), 
            by = c("pitcher" = "xMLBAMID"))

# Save pitch type results
write_csv(final_2024, paste0(output_dir, "xgboost_regression_pitcher_stuff_plus_by_pitch_2024_pfx.csv"))
cat("Pitch type results saved to", paste0(output_dir, "xgboost_regression_pitcher_stuff_plus_by_pitch_2024_pfx.csv"), "\n")

# -----------------------------------------------------------------------------
# 22. EXTRACT STUFF+ FORMULA PARAMETERS
# -----------------------------------------------------------------------------

cat("\n", rep("=", 50), "\n")
cat("STUFF+ FORMULA PARAMETERS EXTRACTION\n")
cat(rep("=", 50), "\n\n")

# Calculate season-specific means and standard deviations for Stuff+ formula
scaling_params <- df_test %>%
  group_by(year) %>%
  summarise(
    mean_predicted_target = mean(predicted_target, na.rm = TRUE),
    sd_predicted_target = sd(predicted_target, na.rm = TRUE),
    n_pitches = n(),
    .groups = 'drop'
  )

print("Stuff+ Formula Parameters for 2024:")
print(scaling_params)

# Save the parameters
write_csv(scaling_params, paste0(output_dir, "stuff_plus_formula_parameters_2024_pfx.csv"))
cat("\nFormula parameters saved to", paste0(output_dir, "stuff_plus_formula_parameters_2024_pfx.csv"), "\n")

cat("\nStuff+ Formula for 2024:\n")
cat("stuff_plus = 100 - ((predicted_target -", scaling_params$mean_predicted_target, ") /", scaling_params$sd_predicted_target, ") * 10\n")

# Also save as an R object for easy loading later
formula_params <- list(
  year = 2024,
  mean = scaling_params$mean_predicted_target,
  sd = scaling_params$sd_predicted_target,
  n_pitches = scaling_params$n_pitches
)
saveRDS(formula_params, paste0(output_dir, "stuff_plus_formula_params_2024_pfx.rds"))
cat("\nFormula parameters also saved as R object:", paste0(output_dir, "stuff_plus_formula_params_2024_pfx.rds"), "\n")

# -----------------------------------------------------------------------------
# 23. FEATURE IMPORTANCE VISUALIZATION
# -----------------------------------------------------------------------------

# Load ggplot2 for visualization
if(!require(ggplot2, quietly = TRUE)) install.packages("ggplot2")
library(ggplot2)

# Create professional feature importance plot
ggplot(importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#2E86AB", alpha = 0.8) +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(
    title = "XGBoost Regression Model - Feature Importance",
    subtitle = "Stuff+ Run Value Prediction Model (2024) - pfx_x/pfx_z Version",
    x = "Features",
    y = "Importance (Gain)",
    caption = "Higher values indicate more predictive features"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  ) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(importance$Gain) * 1.15))

# Save the plot
ggsave(paste0(output_dir, "xgboost_regression_feature_importance_stuff_plus_2024_pfx.png"), 
       width = 10, height = 6, dpi = 300)

cat("Feature importance plot saved to", paste0(output_dir, "xgboost_regression_feature_importance_stuff_plus_2024_pfx.png"), "\n")

# -----------------------------------------------------------------------------
# 24. SUMMARY REPORT
# -----------------------------------------------------------------------------

cat("\n", rep("=", 80), "\n")
cat("XGBOOST REGRESSION RUN VALUE STUFF+ MODEL COMPLETE - 2024 VERSION (PFX)\n")
cat(rep("=", 80), "\n\n")

cat("SUMMARY:\n")
cat("- Training years: 2021-2023\n")
cat("- Test year: 2024\n")
cat("- Movement variables: pfx_x and pfx_z (converted to inches)\n")
cat("- Training observations:", nrow(df_train), "\n")
cat("- Test observations:", nrow(df_test), "\n")
cat("- Model RMSE:", round(rmse, 4), "\n")
cat("- Model R-squared:", round(r_squared, 4), "\n\n")

cat("STUFF+ FORMULA FOR 2024 PREDICTIONS:\n")
cat("stuff_plus = 100 - ((predicted_target -", round(m_2024, 6), ") /", round(s_2024, 6), ") * 10\n\n")

cat("ALL FILES SAVED TO:", output_dir, "\n")
cat("- Model:", "xgboost_regression_model_runvalue_bayesian_optimized_2021_2023_pfx.txt\n")
cat("- Test results:", "test_results_xgboost_regression_runvalue_2024_pfx.csv\n")
cat("- Formula parameters:", "stuff_plus_formula_parameters_2024_pfx.csv\n")
cat("- Pitch type results:", "xgboost_regression_pitcher_stuff_plus_by_pitch_2024_pfx.csv\n")
cat("- Feature importance plot:", "xgboost_regression_feature_importance_stuff_plus_2024_pfx.png\n")

# -----------------------------------------------------------------------------
# 25. CREATE EXAMPLE PREDICTION FUNCTION
# -----------------------------------------------------------------------------

cat("\n", rep("=", 50), "\n")
cat("EXAMPLE FUNCTION FOR APPLYING STUFF+ TO NEW 2024 DATA:\n")
cat(rep("=", 50), "\n\n")

# Create a function that can be used to calculate Stuff+ for new predictions
calculate_stuff_plus_2024 <- function(predicted_run_values) {
  # Using the parameters we just calculated
  mean_2024 <- formula_params$mean
  sd_2024 <- formula_params$sd
  
  # Calculate Stuff+
  stuff_plus <- 100 - ((predicted_run_values - mean_2024) / sd_2024) * 10
  
  return(stuff_plus)
}

# Example usage
cat("Example function created: calculate_stuff_plus_2024()\n")
cat("Usage: stuff_plus_values <- calculate_stuff_plus_2024(predicted_run_values)\n")
cat("\nTo use with new data:\n")
cat("1. Load the model: model <- xgb.load('", paste0(output_dir, "xgboost_regression_model_runvalue_bayesian_optimized_2021_2023_pfx.txt"), "')\n", sep = "")
cat("2. Load preprocessing params: preprocess_params <- readRDS('", paste0(output_dir, "preprocess_params_xgboost_regression_runvalue_optimized_pfx.rds"), "')\n", sep = "")
cat("3. Load formula params: formula_params <- readRDS('", paste0(output_dir, "stuff_plus_formula_params_2024_pfx.rds"), "')\n", sep = "")
cat("4. Preprocess your new data using the same features and scaling\n")
cat("5. Make predictions: predictions <- predict(model, new_data_matrix)\n")
cat("6. Calculate Stuff+: stuff_plus <- calculate_stuff_plus_2024(predictions)\n")

cat("\n", rep("=", 80), "\n")
cat("SCRIPT EXECUTION COMPLETE\n")
cat(rep("=", 80), "\n")