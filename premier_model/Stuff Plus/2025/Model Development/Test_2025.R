# =============================================================================
# Stuff+ Model Testing Script - 2025 Data Only
# Uses pre-trained model from training script to test on new 2025 data
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGE LOADING AND SETUP
# -----------------------------------------------------------------------------

# Install and load required packages
packages <- c("dplyr", "readr", "tidyr", "xgboost", "ggplot2", "httr", "jsonlite")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(dplyr)
library(readr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(httr)
library(jsonlite)
library(purrr)

# Load environment variables from .env file
env_path <- file.path(dirname(dirname(dirname(dirname(dirname(getwd()))))), ".env")
if (file.exists(env_path)) {
  readRenviron(env_path)
}  

# one time:
remotes::install_github("BillPetti/baseballr")   # needs 'remotes' first time
# every script:
library(baseballr)
library(purrr)     # map_dfr
library(lubridate) # easy dates

season_start <- as.Date("2025-03-20")   # Opening-Day eve; adjust if needed
season_end   <- as.Date("2025-11-05")   # day after WS-G7, safe upper bound

date_vec <- seq(season_start, season_end, by = "1 day")

message("Grabbing 2025 Statcast … this will take a while")

df_2025 <- map_dfr(
  date_vec,
  function(d) {
    tryCatch({
      message(format(d), appendLF = FALSE)
      
      res <- statcast_search(
        start_date  = d,
        end_date    = d,
        player_type = "pitcher"
      )
      
      if (nrow(res) == 0) {
        message(" – no games")
        return(NULL)             # <- drops the empty frame
      }
      
      res                          # <- keep normal days
      
    }, error = function(e) {
      message(" – skipped (", e$message, ")")
      NULL
    })
  }
)

message("\nFinished. Rows pulled: ", nrow(df_2025))


# Set seed for reproducibility
set.seed(42)

# Set paths
output_dir <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2025/Model Development/"
new_data_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Data/2020_25Training.csv"

# Model and preprocessing files (from your training script)
model_path <- paste0(output_dir, "xgboost_regression_model_runvalue_bayesian_optimized_2022_2024_pfx.txt")
preprocess_path <- paste0(output_dir, "preprocess_params_xgboost_regression_runvalue_optimized_pfx.rds")

cat("=== Stuff+ Testing Script for 2025 Data ===\n")
cat("Loading new data from:", new_data_path, "\n")
cat("Loading trained model from:", model_path, "\n")
cat("Output directory:", output_dir, "\n\n")

# -----------------------------------------------------------------------------
# 2. LOAD PRE-TRAINED MODEL AND PREPROCESSING PARAMETERS
# -----------------------------------------------------------------------------

# Check if model files exist
if(!file.exists(model_path)) {
  stop("Trained model not found at: ", model_path, "\nPlease run the training script first.")
}

if(!file.exists(preprocess_path)) {
  stop("Preprocessing parameters not found at: ", preprocess_path, "\nPlease run the training script first.")
}

# Load the trained model
cat("Loading pre-trained model...\n")
trained_model <- xgb.load(model_path)

# Load preprocessing parameters
cat("Loading preprocessing parameters...\n")
preprocess_params <- readRDS(preprocess_path)

# Try to load caret, but provide fallback if it fails
caret_available <- FALSE
tryCatch({
  library(caret)
  caret_available <- TRUE
  cat("caret package loaded successfully\n")
}, error = function(e) {
  cat("caret package failed to load. Using manual preprocessing instead.\n")
})

cat("Model and preprocessing parameters loaded successfully.\n\n")

# -----------------------------------------------------------------------------
# 3. DATA LOADING AND RUN VALUE ASSIGNMENT
# -----------------------------------------------------------------------------

# Load the new MLB pitch data (with 2025 data)
cat("Loading new dataset...\n")
df <- df_2025       # << just use the freshly-scraped frame
cat("Dataset loaded. Total rows:", nrow(df), "\n")

# Load run values data
run_values_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Data/run_values.csv"
df_run_values <- read_csv(run_values_path)

# Define pitch outcome groupings (same as training script)
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

# Remove any existing delta_run_exp columns
df <- df %>%
  select(-any_of(c("delta_run_exp", "delta_run_exp.x", "delta_run_exp.y", "delta_run_exp.x.x", "delta_run_exp.y.y")))

# Join run values (same process as training)
df <- df %>%
  left_join(df_run_values, by = c("events" = "event", "balls", "strikes"))

df <- df %>%
  mutate(description_mapped = ifelse(description %in% names(des_dict), des_dict[description], description))

df <- df %>%
  left_join(df_run_values, by = c("description_mapped" = "event", "balls", "strikes"), suffix = c("_event", "_desc"))

df <- df %>%
  mutate(target = coalesce(delta_run_exp_event, delta_run_exp_desc))

cat("Run value assignment completed.\n\n")

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (IDENTICAL TO TRAINING SCRIPT)
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
cat("Applying feature engineering...\n")
df_processed <- feature_engineering(df)
cat("Feature engineering completed.\n\n")

# -----------------------------------------------------------------------------
# 5. FILTER FOR 2025 TEST DATA
# -----------------------------------------------------------------------------

# Check available years
available_years <- sort(unique(df_processed$year))
cat("Available years in new dataset:", paste(available_years, collapse = ", "), "\n")

# Filter for 2025 data
df_test_2025 <- df_processed %>%
  filter(year == 2025)

cat("2025 data found:", nrow(df_test_2025), "rows\n")

if(nrow(df_test_2025) == 0) {
  stop("No 2025 data found in the dataset. Please check the data file.")
}

# Define the features (same as training script)
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

target <- 'target'

# Drop rows with null values in features and target
df_test_2025 <- df_test_2025 %>%
  drop_na(all_of(c(features, target)))

cat("Final 2025 test data after cleaning:", nrow(df_test_2025), "observations\n\n")

# -----------------------------------------------------------------------------
# 6. PREPARE TEST DATA FOR PREDICTION
# -----------------------------------------------------------------------------

cat("Preparing test data for prediction...\n")

# Extract features
X_test <- df_test_2025[features]

# Apply the same scaling that was used during training
if(caret_available) {
  X_test_scaled <- predict(preprocess_params, X_test)
} else {
  # Manual scaling using saved parameters
  X_test_scaled <- scale(X_test, center = preprocess_params$means, scale = preprocess_params$sds)
  X_test_scaled <- as.data.frame(X_test_scaled)
}

# Convert to matrix for prediction
X_test_matrix <- as.matrix(X_test_scaled)

cat("Test data preprocessing completed.\n\n")

# -----------------------------------------------------------------------------
# 7. MAKE PREDICTIONS
# -----------------------------------------------------------------------------

cat("Making predictions on 2025 data...\n")

# Predict using the trained model
predictions <- predict(trained_model, X_test_matrix)

# Add predictions to the test dataframe
df_test_2025 <- df_test_2025 %>%
  mutate(predicted_target = predictions)

cat("Predictions completed!\n\n")

# -----------------------------------------------------------------------------
# 8. CALCULATE STUFF+ FOR 2025
# -----------------------------------------------------------------------------

cat("Calculating Stuff+ for 2025...\n")

# Calculate mean and SD for 2025 predictions
m_2025 <- mean(df_test_2025$predicted_target, na.rm = TRUE)
s_2025 <- sd(df_test_2025$predicted_target, na.rm = TRUE)

# Add Stuff+ to test data
df_test_2025 <- df_test_2025 %>%
  mutate(
    target_zscore = (predicted_target - m_2025) / s_2025,
    stuff_plus = 100 - target_zscore * 10
  )

# Aggregate by pitcher
df_agg_2025 <- df_test_2025 %>%
  group_by(pitcher, year) %>%
  summarise(
    count = n(),
    stuff_plus = mean(stuff_plus, na.rm = TRUE),
    .groups = "drop"
  )

cat("Stuff+ calculation completed.\n\n")

# -----------------------------------------------------------------------------
# 9. EVALUATION METRICS
# -----------------------------------------------------------------------------

cat("Calculating evaluation metrics...\n")

# Calculate basic evaluation metrics
actual_target <- df_test_2025$target
predicted_target <- df_test_2025$predicted_target

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
cat("MODEL EVALUATION METRICS (2025 TEST DATA):\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAE:", round(mae, 4), "\n")
cat("R-squared:", round(r_squared, 4), "\n")

# Display summary statistics
cat("\nACTUAL RUN VALUE SUMMARY:\n")
print(summary(actual_target))
cat("\nPREDICTED RUN VALUE SUMMARY:\n")
print(summary(predicted_target))

# -----------------------------------------------------------------------------
# 10. FANGRAPHS DATA FETCH FOR 2025
# -----------------------------------------------------------------------------

cat("\nFetching Fangraphs data for 2025...\n")

# Fetch data from Fangraphs API for the 2025 MLB season
url <- "https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season=2025&season1=2025&ind=1&qual=0&type=8&month=0&pageitems=500000"

tryCatch({
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
  
  cat("Fangraphs data fetched successfully.\n")
  
}, error = function(e) {
  cat("Warning: Could not fetch Fangraphs data. Creating empty dataframe.\n")
  df_fg <- data.frame()
})

# Load wOBA data if available
woba_path <- "woba_2020_2025.csv"
if(file.exists(woba_path)) {
  df_woba <- read_csv(woba_path)
  
  # Join the Fangraphs data with the wOBA data
  if(nrow(df_fg) > 0) {
    df_fg <- df_fg %>%
      left_join(df_woba, by = c("xMLBAMID" = "player_id", "Season" = "year"))
  }
  cat("wOBA data joined successfully.\n")
} else {
  cat("Warning: wOBA data file not found at", woba_path, "\n")
}

# -----------------------------------------------------------------------------
# 11. STUFF+ BY PITCH TYPE FOR 2025
# -----------------------------------------------------------------------------

cat("Calculating Stuff+ by pitch type...\n")

# Create pitch type dataframe for 2025
final_2025 <- df_test_2025 %>%
  group_by(pitcher, pitch_type) %>%
  summarise(stuff_plus_pp = mean(stuff_plus, na.rm = TRUE), .groups = 'drop') %>%
  filter(pitch_type %in% c('FF', 'SI', 'FC', 'CH', 'SL', 'CU', 'KC', 'ST')) %>%
  pivot_wider(names_from = pitch_type, values_from = stuff_plus_pp, names_prefix = "stuff_plus_") %>%
  left_join(df_agg_2025 %>% select(pitcher, stuff_plus), by = "pitcher")

# Join with Fangraphs data if available
if(nrow(df_fg) > 0) {
  final_2025 <- final_2025 %>%
    left_join(df_fg %>% filter(Season == 2025) %>% 
                select(xMLBAMID, PlayerName, Team, IP, ERA, FIP, xFIP, `K.BB.`, 
                       any_of("woba")), 
              by = c("pitcher" = "xMLBAMID"))
}

cat("Pitch type analysis completed.\n\n")

# -----------------------------------------------------------------------------
# 12. SAVE ALL RESULTS
# -----------------------------------------------------------------------------

cat("Saving results...\n")

# Save the test results
write_csv(df_test_2025, paste0(output_dir, "test_results_xgboost_regression_runvalue_2025_pfx.csv"))
cat("Test results saved to: test_results_xgboost_regression_runvalue_2025_pfx.csv\n")

# Save evaluation metrics
eval_metrics <- data.frame(
  metric = c("RMSE", "MAE", "R_squared"),
  value = c(rmse, mae, r_squared)
)
write_csv(eval_metrics, paste0(output_dir, "xgboost_regression_evaluation_metrics_runvalue_2025_pfx.csv"))
cat("Evaluation metrics saved to: xgboost_regression_evaluation_metrics_runvalue_2025_pfx.csv\n")

# Save Stuff+ formula parameters
scaling_params <- data.frame(
  year = 2025,
  mean_predicted_target = m_2025,
  sd_predicted_target = s_2025,
  n_pitches = nrow(df_test_2025)
)
write_csv(scaling_params, paste0(output_dir, "stuff_plus_formula_parameters_2025_pfx.csv"))
cat("Formula parameters saved to: stuff_plus_formula_parameters_2025_pfx.csv\n")

# Save as R object
formula_params <- list(
  year = 2025,
  mean = m_2025,
  sd = s_2025,
  n_pitches = nrow(df_test_2025)
)
saveRDS(formula_params, paste0(output_dir, "stuff_plus_formula_params_2025_pfx.rds"))
cat("Formula parameters (R object) saved to: stuff_plus_formula_params_2025_pfx.rds\n")

# Save pitch type results
write_csv(final_2025, paste0(output_dir, "xgboost_regression_pitcher_stuff_plus_by_pitch_2025_pfx.csv"))
cat("Pitch type results saved to: xgboost_regression_pitcher_stuff_plus_by_pitch_2025_pfx.csv\n")

# Save aggregated results
write_csv(df_agg_2025, paste0(output_dir, "xgboost_regression_pitcher_stuff_plus_aggregated_2025_pfx.csv"))
cat("Aggregated pitcher results saved to: xgboost_regression_pitcher_stuff_plus_aggregated_2025_pfx.csv\n")

# -----------------------------------------------------------------------------
# 13. FEATURE IMPORTANCE FROM TRAINED MODEL
# -----------------------------------------------------------------------------

cat("\nExtracting feature importance from trained model...\n")

# Get feature importance with proper feature names
importance <- xgb.importance(feature_names = features, model = trained_model)
cat("Feature importance extracted.\n")

# Create feature importance plot
ggplot(importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#2E86AB", alpha = 0.8) +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(
    title = "XGBoost Regression Model - Feature Importance",
    subtitle = "Stuff+ Run Value Prediction Model (2025 Testing) - pfx_x/pfx_z Version",
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
ggsave(paste0(output_dir, "xgboost_regression_feature_importance_stuff_plus_2025_testing_pfx.png"), 
       width = 10, height = 6, dpi = 300)
cat("Feature importance plot saved to: xgboost_regression_feature_importance_stuff_plus_2025_testing_pfx.png\n")

# Save feature importance data
write_csv(importance, paste0(output_dir, "xgboost_regression_feature_importance_2025_pfx.csv"))
cat("Feature importance data saved to: xgboost_regression_feature_importance_2025_pfx.csv\n")

# -----------------------------------------------------------------------------
# 14. FINAL SUMMARY REPORT
# -----------------------------------------------------------------------------

cat("\n", rep("=", 80), "\n")
cat("STUFF+ MODEL TESTING ON 2025 DATA - COMPLETE\n")
cat(rep("=", 80), "\n\n")

cat("SUMMARY:\n")
cat("- Used pre-trained model from 2022-2024 data\n")
cat("- Tested on 2025 data from:", new_data_path, "\n")
cat("- Test observations:", nrow(df_test_2025), "\n")
cat("- Model RMSE on 2025 data:", round(rmse, 4), "\n")
cat("- Model R-squared on 2025 data:", round(r_squared, 4), "\n\n")

cat("STUFF+ FORMULA FOR 2025:\n")
cat("stuff_plus = 100 - ((predicted_target -", round(m_2025, 6), ") /", round(s_2025, 6), ") * 10\n\n")

cat("FILES SAVED TO:", output_dir, "\n")
cat("- Test results: test_results_xgboost_regression_runvalue_2025_pfx.csv\n")
cat("- Evaluation metrics: xgboost_regression_evaluation_metrics_runvalue_2025_pfx.csv\n")
cat("- Formula parameters: stuff_plus_formula_parameters_2025_pfx.csv\n")
cat("- Pitch type results: xgboost_regression_pitcher_stuff_plus_by_pitch_2025_pfx.csv\n")
cat("- Aggregated results: xgboost_regression_pitcher_stuff_plus_aggregated_2025_pfx.csv\n")
cat("- Feature importance: xgboost_regression_feature_importance_2025_pfx.csv\n")
cat("- Feature importance plot: xgboost_regression_feature_importance_stuff_plus_2025_testing_pfx.png\n\n")

cat("2025 TESTING COMPLETE - Ready for analysis!\n")
cat(rep("=", 80), "\n")



# ---------------------------------------------------------------------------
# 15. PUSH 2025 RESULTS INTO RDS AND REFRESH Stuff+ (fully automated)
# ---------------------------------------------------------------------------

library(DBI)
library(RPostgres)
library(readr)

# ---------- connection credentials (edit once) -----------------------------
pg_con <- dbConnect(
  RPostgres::Postgres(),
  host     = Sys.getenv("DB_HOST"),
  port     = 5432,
  dbname   = "postgres",
  user     = Sys.getenv("DB_USER"),
  password = Sys.getenv("DB_PASSWORD"),
  sslmode  = "require"
)

pred_tbl  <- Id(schema = "public", table = "stuff_plus_predictions")
scale_tbl <- Id(schema = "public", table = "stuff_plus_scaling")

# ---------- 1. Remove stale 2025 rows --------------------------------------
dbExecute(pg_con, "
  DELETE FROM public.stuff_plus_predictions
   WHERE year = 2025;
")
cat("Old 2025 rows removed from stuff_plus_predictions.\n")

# ---------- 2. Load only the needed columns ---------------------------------
keep_cols <- c(
  "game_date","year","pitch_number","pitcher","player_name","pitch_type","p_throws","stand",
  "release_speed","release_spin_rate","release_extension","release_pos_x","release_pos_z",
  "pfx_x","pfx_z","plate_x","plate_z","zone","description","predicted_target",
  "pfx_x_inches","pfx_z_inches","arm_slot_Vector","speed_diff","pfx_z_diff","pfx_x_diff"
)

cat("Copying new 2025 rows into DB…\n")
read_csv_chunked(
  paste0(output_dir, "test_results_xgboost_regression_runvalue_2025_pfx.csv"),
  DataFrameCallback$new(function(df, pos) {
    df <- df[, keep_cols]                           # trim to 26 cols
    dbWriteTable(pg_con, name = pred_tbl, value = df,
                 append = TRUE, row.names = FALSE)
  }),
  chunk_size = 5e4, show_col_types = FALSE, progress = FALSE
)
cat("Insert complete.\n")

# ---------- 3. Update scaling parameters table ------------------------------
dbExecute(pg_con, "
  INSERT INTO public.stuff_plus_scaling (year, mean_predicted_target, sd_predicted_target, n_pitches)
  SELECT 2025,
         AVG(predicted_target),
         STDDEV_POP(predicted_target),
         COUNT(*)
    FROM public.stuff_plus_predictions
   WHERE year = 2025
  ON CONFLICT (year) DO UPDATE
      SET mean_predicted_target = EXCLUDED.mean_predicted_target,
          sd_predicted_target   = EXCLUDED.sd_predicted_target,
          n_pitches             = EXCLUDED.n_pitches,
          updated_at            = NOW();
")
cat("Scaling parameters row for 2025 refreshed.\n")

# ---------- 4. Recompute Stuff+ for 2025 directly in SQL --------------------
dbExecute(pg_con, "
  UPDATE public.stuff_plus_predictions AS sp
     SET stuff_plus =
         100 - 10 * (sp.predicted_target - sc.mean_predicted_target) / sc.sd_predicted_target
    FROM public.stuff_plus_scaling AS sc
   WHERE sp.year = 2025
     AND sc.year = 2025;
")
cat("Stuff+ column recalculated for 2025 rows.\n")

dbDisconnect(pg_con)
cat("Database step finished – daily refresh complete.\n")
