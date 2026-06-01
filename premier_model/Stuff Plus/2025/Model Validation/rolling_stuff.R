# =============================================================================
# XGBoost Run Value Stuff+ Rolling Analysis from 2025 Predictions
# Uses pre-computed predictions from XGBoost model
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGE LOADING AND SETUP
# -----------------------------------------------------------------------------

packages <- c("dplyr", "readr", "ggplot2", "scales", "httr", "jsonlite")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(dplyr)
library(readr)
library(ggplot2)
library(scales)
library(httr)
library(jsonlite)

set.seed(42)

# -----------------------------------------------------------------------------
# 2. LOAD EXISTING PREDICTIONS
# -----------------------------------------------------------------------------

# Load the existing XGBoost model predictions
xgboost_file <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2025/Model Development/test_results_xgboost_regression_runvalue_2025_pfx.csv"
output_folder <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2025/Model Validation/"

df_test <- read_csv(xgboost_file)

cat("XGBoost model predictions loaded:\n")
cat("Total observations:", nrow(df_test), "\n")
cat("Years available:", paste(sort(unique(df_test$year)), collapse = ", "), "\n")
cat("Unique pitchers:", length(unique(df_test$pitcher)), "\n")

# Verify we have the required columns
required_cols <- c("pitcher", "year", "game_date", "at_bat_number", "pitch_number", "predicted_target")
missing_cols <- required_cols[!required_cols %in% colnames(df_test)]

if(length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

cat("Required columns found: ✓\n")

# Check if stuff_plus already exists
if("stuff_plus" %in% colnames(df_test)) {
  cat("Stuff+ column already exists in data ✓\n")
  df_test_with_stuff <- df_test %>%
    rename(xgboost_stuff_plus = stuff_plus)
} else {
  # Calculate Stuff+ for 2025 using that season's mean & SD
  cat("Calculating Stuff+ for 2025...\n")
  m_2025 <- mean(df_test$predicted_target, na.rm = TRUE)
  s_2025 <- sd(df_test$predicted_target, na.rm = TRUE)
  
  df_test_with_stuff <- df_test %>%
    mutate(xgboost_stuff_plus = 100 - ((predicted_target - m_2025) / s_2025) * 10)
  
  cat("XGBoost Stuff+ calculated for 2025\n")
}

# -----------------------------------------------------------------------------
# 3. ROLLING XGBOOST STUFF+ ANALYSIS
# -----------------------------------------------------------------------------

create_xgboost_rolling_stuff_plus <- function(df, max_pitches = 300) {
  
  cat("Creating XGBoost rolling Stuff+ up to", max_pitches, "pitches...\n")
  
  # Order pitches chronologically for each pitcher-year
  df_ordered <- df %>%
    arrange(pitcher, year, game_date, at_bat_number, pitch_number) %>%
    group_by(pitcher, year) %>%
    mutate(pitch_order = row_number(),
           trained_pitches = n()) %>%
    ungroup()
  
  # Filter to only first 300 pitches per pitcher-year (for rolling calculations)
  df_filtered <- df_ordered %>%
    filter(pitch_order <= max_pitches)
  
  # Calculate rolling averages for each pitcher-year
  final_rolling <- df_filtered %>%
    group_by(pitcher, year) %>%
    arrange(pitch_order) %>%
    summarise(
      trained_pitches = first(trained_pitches),
      
      # Calculate rolling averages correctly
      xgboost_stuff_plus_1 = ifelse(n() >= 1, mean(xgboost_stuff_plus[1:min(1, n())]), NA),
      xgboost_stuff_plus_10 = ifelse(n() >= 10, mean(xgboost_stuff_plus[1:min(10, n())]), NA),
      xgboost_stuff_plus_25 = ifelse(n() >= 25, mean(xgboost_stuff_plus[1:min(25, n())]), NA),
      xgboost_stuff_plus_50 = ifelse(n() >= 50, mean(xgboost_stuff_plus[1:min(50, n())]), NA),
      xgboost_stuff_plus_75 = ifelse(n() >= 75, mean(xgboost_stuff_plus[1:min(75, n())]), NA),
      xgboost_stuff_plus_100 = ifelse(n() >= 100, mean(xgboost_stuff_plus[1:min(100, n())]), NA),
      xgboost_stuff_plus_150 = ifelse(n() >= 150, mean(xgboost_stuff_plus[1:min(150, n())]), NA),
      xgboost_stuff_plus_200 = ifelse(n() >= 200, mean(xgboost_stuff_plus[1:min(200, n())]), NA),
      xgboost_stuff_plus_250 = ifelse(n() >= 250, mean(xgboost_stuff_plus[1:min(250, n())]), NA),
      xgboost_stuff_plus_300 = ifelse(n() >= 300, mean(xgboost_stuff_plus[1:min(300, n())]), NA),
      
      # Also calculate final season average for comparison
      final_season_xgboost_stuff_plus = mean(xgboost_stuff_plus, na.rm = TRUE),
      
      .groups = 'drop'
    )
  
  return(final_rolling)
}

# Create the rolling dataframe
rolling_xgboost_df <- create_xgboost_rolling_stuff_plus(df_test_with_stuff, max_pitches = 300)

cat("XGBoost rolling Stuff+ dataframe created:\n")
cat("Rows (pitcher-year combinations):", nrow(rolling_xgboost_df), "\n")
cat("Columns:", ncol(rolling_xgboost_df), "\n")

# Show sample of the data
cat("\nSample of XGBoost rolling Stuff+ data:\n")
print(head(rolling_xgboost_df %>% 
             select(pitcher, year, trained_pitches, xgboost_stuff_plus_1, xgboost_stuff_plus_10, 
                    xgboost_stuff_plus_25, xgboost_stuff_plus_50, xgboost_stuff_plus_100, 
                    final_season_xgboost_stuff_plus), 10))

# -----------------------------------------------------------------------------
# 4. ADD SEASON STATS FROM FANGRAPHS
# -----------------------------------------------------------------------------

cat("Adding season performance statistics...\n")

# Fetch data from Fangraphs API for 2025
url <- "https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season=2025&season1=2025&ind=1&qual=0&type=8&month=0&pageitems=500000"

response <- GET(url)
data <- fromJSON(content(response, "text"))

df_fg <- data.frame(data$data)

# Convert column types
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

# Load wOBA data (using the updated 2025 data)
df_woba <- read_csv("C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Data/woba_2020_2025.csv")

# Join the Fangraphs data with the wOBA data
df_fg <- df_fg %>%
  left_join(df_woba, by = c("xMLBAMID" = "player_id", "Season" = "year"))

# Join rolling Stuff+ with season performance stats
rolling_xgboost_with_stats <- rolling_xgboost_df %>%
  left_join(
    df_fg %>% 
      select(xMLBAMID, Season, PlayerName, Team, G, IP, ERA, FIP, xFIP, `K.BB.`, 
             TBF, Pitches, woba) %>%
      rename(pitcher = xMLBAMID, year = Season, fangraphs_pitches = Pitches),
    by = c("pitcher", "year")
  ) %>%
  # Convert K-BB% to percentage format for consistency
  mutate(`K.BB.` = `K.BB.` * 100)

# Save the enhanced dataframe
output_file <- file.path(output_folder, "xgboost_rolling_stuff_plus_with_season_stats_2025.csv")
write_csv(rolling_xgboost_with_stats, output_file)

cat("\nXGBoost rolling Stuff+ with season stats saved to:", output_file, "\n")

# Show updated sample with stats
cat("\nSample of XGBoost data with season stats:\n")
print(head(rolling_xgboost_with_stats %>% 
             select(pitcher, PlayerName, year, trained_pitches, fangraphs_pitches, 
                    xgboost_stuff_plus_1, xgboost_stuff_plus_10, xgboost_stuff_plus_25, 
                    xgboost_stuff_plus_50, xgboost_stuff_plus_100, xgboost_stuff_plus_150,
                    final_season_xgboost_stuff_plus, ERA, FIP), 10))

# Show basic stats
cat("\nBasic Statistics:\n")
cat("Pitchers with at least 300 trained pitches:", sum(rolling_xgboost_with_stats$trained_pitches >= 300, na.rm = TRUE), "\n")
cat("Pitchers with at least 200 trained pitches:", sum(rolling_xgboost_with_stats$trained_pitches >= 200, na.rm = TRUE), "\n")
cat("Pitchers with at least 100 trained pitches:", sum(rolling_xgboost_with_stats$trained_pitches >= 100, na.rm = TRUE), "\n")
cat("Average trained pitches per pitcher-year:", round(mean(rolling_xgboost_with_stats$trained_pitches, na.rm = TRUE)), "\n")
cat("Pitchers with performance stats:", sum(!is.na(rolling_xgboost_with_stats$ERA)), "\n")
cat("Average Fangraphs pitches per pitcher-year:", round(mean(rolling_xgboost_with_stats$fangraphs_pitches, na.rm = TRUE)), "\n")

# =============================================================================
# 5. XGBOOST ROLLING STUFF+ vs wOBA CORRELATION ANALYSIS
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("XGBOOST ROLLING STUFF+ vs wOBA CORRELATION ANALYSIS\n")
cat(rep("=", 60), "\n")

# Filter to pitchers with valid wOBA data AND at least 1000 Fangraphs pitches
correlation_data <- rolling_xgboost_with_stats %>%
  filter(!is.na(woba) & !is.na(fangraphs_pitches) & fangraphs_pitches >= 1000)

cat("Pitchers with valid wOBA data and at least 1000 Fangraphs pitches:", nrow(correlation_data), "\n")
cat("Original sample size (before filtering):", sum(!is.na(rolling_xgboost_with_stats$woba)), "\n")
cat("Pitchers filtered out (< 1000 pitches):", sum(!is.na(rolling_xgboost_with_stats$woba)) - nrow(correlation_data), "\n\n")

# Calculate correlations between each rolling Stuff+ and wOBA
rolling_xgboost_columns <- c("xgboost_stuff_plus_1", "xgboost_stuff_plus_10", "xgboost_stuff_plus_25", 
                             "xgboost_stuff_plus_50", "xgboost_stuff_plus_75", "xgboost_stuff_plus_100", 
                             "xgboost_stuff_plus_150", "xgboost_stuff_plus_200", "xgboost_stuff_plus_250", 
                             "xgboost_stuff_plus_300", "final_season_xgboost_stuff_plus")

xgboost_correlations <- data.frame(
  pitch_count = c(1, 10, 25, 50, 75, 100, 150, 200, 250, 300, "Full Season"),
  correlation = numeric(length(rolling_xgboost_columns)),
  n_pitchers = numeric(length(rolling_xgboost_columns))
)

for(i in 1:length(rolling_xgboost_columns)) {
  col_name <- rolling_xgboost_columns[i]
  
  # Filter to pitchers who have this rolling average (reached enough pitches)
  valid_data <- correlation_data %>%
    filter(!is.na(.data[[col_name]]))
  
  if(nrow(valid_data) > 0) {
    xgboost_correlations$correlation[i] <- cor(valid_data[[col_name]], valid_data$woba, use = "complete.obs")
    xgboost_correlations$n_pitchers[i] <- nrow(valid_data)
  } else {
    xgboost_correlations$correlation[i] <- NA
    xgboost_correlations$n_pitchers[i] <- 0
  }
}

# Display correlation results
cat("XGBoost Correlation between Rolling Stuff+ and wOBA:\n")
cat("(Negative correlation = better Stuff+ leads to lower wOBA)\n")
cat("(Filtered to pitchers with >= 1000 Fangraphs pitches)\n\n")
print(xgboost_correlations)

# Find when correlation stabilizes
final_xgboost_correlation <- xgboost_correlations$correlation[xgboost_correlations$pitch_count == "Full Season"]
stabilization_threshold <- 0.01

stabilized_xgboost_correlations <- xgboost_correlations %>%
  filter(pitch_count != "Full Season") %>%
  mutate(
    pitch_count_num = as.numeric(pitch_count),
    diff_from_final = abs(correlation - final_xgboost_correlation),
    stabilized = diff_from_final <= stabilization_threshold
  )

first_xgboost_stabilized <- stabilized_xgboost_correlations %>%
  filter(stabilized == TRUE) %>%
  slice_head(n = 1)

cat("\n", rep("-", 40), "\n")
cat("XGBOOST STABILIZATION ANALYSIS:\n")
cat("Final season correlation with wOBA:", round(final_xgboost_correlation, 4), "\n")

if(nrow(first_xgboost_stabilized) > 0) {
  cat("XGBoost correlation stabilizes (within 0.01) at:", first_xgboost_stabilized$pitch_count_num, "pitches\n")
  cat("Correlation at stabilization point:", round(first_xgboost_stabilized$correlation, 4), "\n")
} else {
  cat("XGBoost correlation does not stabilize within 300 pitches\n")
}

# Save correlation results
xgboost_correlation_file <- file.path(output_folder, "xgboost_rolling_stuff_plus_woba_correlations_1000plus_pitches_2025.csv")
write_csv(xgboost_correlations, xgboost_correlation_file)

cat("\nXGBoost correlation results saved to:", xgboost_correlation_file, "\n")

# =============================================================================
# 6. XGBOOST STABILIZATION VISUALIZATION
# =============================================================================

# Create overall XGBoost stabilization plot
xgboost_corr_data <- xgboost_correlations %>%
  filter(pitch_count != "Full Season") %>%
  mutate(pitch_count = as.numeric(pitch_count)) %>%
  filter(!is.na(correlation))

p_xgboost <- ggplot(xgboost_corr_data, aes(x = pitch_count, y = abs(correlation))) +
  geom_line(size = 3, alpha = 0.9, color = "#1f77b4") +
  geom_point(size = 4, alpha = 0.9, color = "#1f77b4") +
  
  # Reference lines
  geom_hline(yintercept = 0.15, linetype = "dashed", color = "#666666", size = 0.8, alpha = 0.7) +
  geom_hline(yintercept = 0.20, linetype = "dashed", color = "#333333", size = 0.8, alpha = 0.7) +
  
  # Scales
  scale_x_continuous(
    breaks = c(25, 50, 100, 150, 200, 300),
    labels = c("25", "50", "100", "150", "200", "300")
  ) +
  scale_y_continuous(
    limits = c(0, max(abs(xgboost_corr_data$correlation)) * 1.1),
    expand = c(0, 0)
  ) +
  
  # Labels
  labs(
    title = "XGBoost Run Value Stuff+ Predictive Power Stabilization (2025)",
    subtitle = "Correlation strength between rolling Stuff+ and season wOBA",
    x = "Number of Pitches",
    y = "Correlation Strength (absolute value)",
    caption = "XGBoost model with pfx_x/pfx_z features | Filtered to pitchers with ≥1000 Fangraphs pitches"
  ) +
  
  # Theme
  theme_minimal(base_family = "Arial", base_size = 12) +
  theme(
    plot.title = element_text(size = 20, face = "bold", color = "#222222", margin = margin(b = 5)),
    plot.subtitle = element_text(size = 14, color = "#666666", margin = margin(b = 20)),
    axis.title.x = element_text(size = 13, color = "#222222", margin = margin(t = 15)),
    axis.title.y = element_text(size = 13, color = "#222222", margin = margin(r = 15)),
    axis.text = element_text(size = 11, color = "#666666"),
    panel.grid.major = element_line(color = "#f0f0f0", size = 0.5),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 10, color = "#999999", hjust = 0, margin = margin(t = 15)),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(25, 25, 25, 25)
  )

# Save the plot
ggsave(file.path(output_folder, "xgboost_stuff_plus_stabilization_1000plus_pitches_2025.png"), p_xgboost, 
       width = 12, height = 8, dpi = 300, bg = "white")

# Display plot
print(p_xgboost)

# =============================================================================
# 7. SUMMARY REPORT
# =============================================================================

cat("\n", rep("=", 80), "\n")
cat("XGBOOST RUN VALUE STUFF+ ROLLING ANALYSIS COMPLETE - 2025\n")
cat(rep("=", 80), "\n")

cat("\nSUMMARY:\n")
cat("- Model: XGBoost Run Value with pfx_x/pfx_z features\n")
cat("- Year analyzed: 2025\n")
cat("- Total pitcher observations:", nrow(rolling_xgboost_with_stats), "\n")
cat("- Pitchers with valid performance data:", sum(!is.na(rolling_xgboost_with_stats$woba)), "\n")
cat("- Pitchers meeting 1000+ pitch threshold:", nrow(correlation_data), "\n")

if(nrow(first_xgboost_stabilized) > 0) {
  cat("- Correlation stabilizes at:", first_xgboost_stabilized$pitch_count_num, "pitches\n")
  cat("- Final correlation with wOBA:", round(final_xgboost_correlation, 4), "\n")
} else {
  cat("- Correlation does not stabilize within 300 pitches\n")
  cat("- Final correlation with wOBA:", round(final_xgboost_correlation, 4), "\n")
}

cat("\nFILES SAVED:\n")
cat("- xgboost_rolling_stuff_plus_with_season_stats_2025.csv\n")
cat("- xgboost_rolling_stuff_plus_woba_correlations_1000plus_pitches_2025.csv\n")
cat("- xgboost_stuff_plus_stabilization_1000plus_pitches_2025.png\n")

cat("\n", rep("=", 80), "\n")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n")