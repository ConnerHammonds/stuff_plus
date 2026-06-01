# =============================================================================
# XGBoost Run Value Model Stabilization Analysis - 2025 Only
# =============================================================================

library(dplyr)
library(readr)
library(ggplot2)
library(scales)

# Load the test results from the XGBoost model
# This should be the output from the XGBoost script: test_results_xgboost_regression_runvalue_2025_pfx.csv
file_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2025/Model Development/test_results_xgboost_regression_runvalue_2025_pfx.csv"
output_folder <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2025/Model Validation/"

df <- read_csv(file_path)

# The df should already have:
# - predicted_target (from XGBoost model)
# - stuff_plus (calculated from predicted_target)
# - year, pitcher, pitch_type, game_date, at_bat_number, pitch_number

# If stuff_plus is not in the data, calculate it using the same method as the XGBoost script
if(!"stuff_plus" %in% colnames(df)) {
  # Calculate stuff_plus for 2025 using that season's mean & SD
  m_2025 <- mean(df$predicted_target, na.rm = TRUE)
  s_2025 <- sd(df$predicted_target, na.rm = TRUE)
  
  df <- df %>%
    mutate(stuff_plus = 100 - ((predicted_target - m_2025) / s_2025) * 10)
}

# Analyze stabilization by pitch type for 2025
analyze_stabilization <- function(df, min_season_pitches = 50) {
  results <- data.frame()
  pitch_thresholds <- c(10, 20, 30, 40, 50, 75, 100, 150, 200)
  major_pitch_types <- c('FF', 'SI', 'FC', 'CH', 'SL', 'CU', 'KC')
  
  for(pitch_type in major_pitch_types) {
    year_pitch_data <- df %>% 
      filter(pitch_type == !!pitch_type) %>%
      arrange(pitcher, game_date, at_bat_number, pitch_number) %>%
      group_by(pitcher) %>%
      mutate(pitch_order = row_number(), season_total = n()) %>%
      ungroup() %>%
      filter(season_total >= min_season_pitches)
    
    if(nrow(year_pitch_data) == 0) next
    
    # Calculate final season stuff_plus for each pitcher
    final_stuff <- year_pitch_data %>%
      group_by(pitcher) %>%
      summarise(final_stuff_plus = mean(stuff_plus, na.rm = TRUE), .groups = 'drop')
    
    for(threshold in pitch_thresholds) {
      running_stuff <- year_pitch_data %>%
        filter(pitch_order <= threshold) %>%
        group_by(pitcher) %>%
        summarise(running_stuff_plus = mean(stuff_plus, na.rm = TRUE),
                  pitches_used = n(), .groups = 'drop') %>%
        filter(pitches_used == threshold)
      
      comparison <- running_stuff %>%
        inner_join(final_stuff, by = "pitcher")
      
      if(nrow(comparison) >= 5) {
        correlation <- cor(comparison$running_stuff_plus, comparison$final_stuff_plus)
        
        results <- rbind(results, data.frame(
          pitch_type = pitch_type,
          pitches = threshold,
          correlation = correlation
        ))
      }
    }
  }
  return(results)
}

# Get results for 2025
results_2025 <- analyze_stabilization(df)

# Create NYT-style plot for pitch type stabilization
create_nyt_plot <- function(results) {
  
  # NYT color palette
  colors <- c(
    "FF" = "#d62728",    # Red
    "SI" = "#ff7f0e",    # Orange  
    "FC" = "#2ca02c",    # Green
    "CH" = "#1f77b4",    # Blue
    "SL" = "#9467bd",    # Purple
    "CU" = "#8c564b",    # Brown
    "KC" = "#e377c2"     # Pink
  )
  
  ggplot(results, aes(x = pitches, y = correlation, color = pitch_type)) +
    geom_line(size = 2.5, alpha = 0.9) +
    geom_point(size = 3, alpha = 0.9) +
    
    # Reference lines
    geom_hline(yintercept = 0.8, linetype = "dashed", color = "#666666", size = 0.8, alpha = 0.7) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "#333333", size = 0.8, alpha = 0.7) +
    
    # Colors
    scale_color_manual(values = colors) +
    
    # Scales
    scale_x_continuous(
      breaks = c(10, 25, 50, 75, 100, 150, 200),
      labels = c("10", "25", "50", "75", "100", "150", "200")
    ) +
    scale_y_continuous(
      limits = c(0.55, 1.02),
      breaks = seq(0.6, 1.0, 0.1),
      labels = percent_format(accuracy = 1),
      expand = c(0, 0)
    ) +
    
    # Labels
    labs(
      title = "XGBoost Run Value Stuff+ Stabilization by Pitch Type: 2025",
      subtitle = "How many pitches of each type are needed for a reliable rating?",
      x = "Number of Pitches",
      y = "Correlation with Final Season Rating",
      color = "Pitch Type",
      caption = "Dashed lines show 80% and 90% correlation thresholds"
    ) +
    
    # NYT Theme
    theme_minimal(base_family = "Arial", base_size = 12) +
    theme(
      # Title and subtitle
      plot.title = element_text(
        size = 20, 
        face = "bold", 
        color = "#222222",
        margin = margin(b = 5)
      ),
      plot.subtitle = element_text(
        size = 14, 
        color = "#666666",
        margin = margin(b = 20)
      ),
      
      # Axes
      axis.title.x = element_text(size = 13, color = "#222222", margin = margin(t = 15)),
      axis.title.y = element_text(size = 13, color = "#222222", margin = margin(r = 15)),
      axis.text = element_text(size = 11, color = "#666666"),
      
      # Legend
      legend.title = element_text(size = 12, face = "bold", color = "#222222"),
      legend.text = element_text(size = 11, color = "#666666"),
      legend.position = "bottom",
      legend.margin = margin(t = 20),
      
      # Grid
      panel.grid.major = element_line(color = "#f0f0f0", size = 0.5),
      panel.grid.minor = element_blank(),
      
      # Caption
      plot.caption = element_text(
        size = 10, 
        color = "#999999", 
        hjust = 0,
        margin = margin(t = 15)
      ),
      
      # Background
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      
      # Margins
      plot.margin = margin(25, 25, 25, 25)
    )
}

# Create pitch type plot
p_2025 <- create_nyt_plot(results_2025)

# Save pitch type plot
ggsave(file.path(output_folder, "xgboost_runvalue_stuff_plus_stabilization_2025_nyt.png"), p_2025, 
       width = 12, height = 8, dpi = 300, bg = "white")

# Overall Stuff+ stabilization (all pitches combined)
analyze_overall_stabilization <- function(df, min_season_pitches = 200) {
  results <- data.frame()
  pitch_thresholds <- c(25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
  
  year_data <- df %>%
    arrange(pitcher, game_date, at_bat_number, pitch_number) %>%
    group_by(pitcher) %>%
    mutate(pitch_order = row_number(), season_total = n()) %>%
    ungroup() %>%
    filter(season_total >= min_season_pitches)
  
  final_stuff <- year_data %>%
    group_by(pitcher) %>%
    summarise(final_stuff_plus = mean(stuff_plus, na.rm = TRUE), .groups = 'drop')
  
  for(threshold in pitch_thresholds) {
    running_stuff <- year_data %>%
      filter(pitch_order <= threshold) %>%
      group_by(pitcher) %>%
      summarise(running_stuff_plus = mean(stuff_plus, na.rm = TRUE),
                pitches_used = n(), .groups = 'drop') %>%
      filter(pitches_used == threshold)
    
    comparison <- running_stuff %>%
      inner_join(final_stuff, by = "pitcher")
    
    if(nrow(comparison) >= 10) {
      correlation <- cor(comparison$running_stuff_plus, comparison$final_stuff_plus)
      
      results <- rbind(results, data.frame(
        pitches = threshold,
        correlation = correlation
      ))
    }
  }
  return(results)
}

# Get overall results for 2025
overall_2025 <- analyze_overall_stabilization(df)

# Create overall Stuff+ plot
p_overall <- ggplot(overall_2025, aes(x = pitches, y = correlation)) +
  geom_line(size = 3, alpha = 0.9, color = "#1f77b4") +
  geom_point(size = 4, alpha = 0.9, color = "#1f77b4") +
  
  # Reference lines
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "#666666", size = 0.8, alpha = 0.7) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "#333333", size = 0.8, alpha = 0.7) +
  
  # Scales
  scale_x_continuous(
    breaks = c(25, 50, 100, 150, 200, 300, 500),
    labels = c("25", "50", "100", "150", "200", "300", "500")
  ) +
  scale_y_continuous(
    limits = c(0.55, 1.02),
    breaks = seq(0.6, 1.0, 0.1),
    labels = percent_format(accuracy = 1),
    expand = c(0, 0)
  ) +
  
  # Labels
  labs(
    title = "XGBoost Run Value Model - Overall Stuff+ Stabilization (2025)",
    subtitle = "How many total pitches are needed for a reliable overall rating?",
    x = "Number of Pitches",
    y = "Correlation with Final Season Rating",
    caption = "Dashed lines show 80% and 90% correlation thresholds"
  ) +
  
  # NYT Theme
  theme_minimal(base_family = "Arial", base_size = 12) +
  theme(
    # Title and subtitle
    plot.title = element_text(
      size = 20, 
      face = "bold", 
      color = "#222222",
      margin = margin(b = 5)
    ),
    plot.subtitle = element_text(
      size = 14, 
      color = "#666666",
      margin = margin(b = 20)
    ),
    
    # Axes
    axis.title.x = element_text(size = 13, color = "#222222", margin = margin(t = 15)),
    axis.title.y = element_text(size = 13, color = "#222222", margin = margin(r = 15)),
    axis.text = element_text(size = 11, color = "#666666"),
    
    # Grid
    panel.grid.major = element_line(color = "#f0f0f0", size = 0.5),
    panel.grid.minor = element_blank(),
    
    # Caption
    plot.caption = element_text(
      size = 10, 
      color = "#999999", 
      hjust = 0,
      margin = margin(t = 15)
    ),
    
    # Background
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    
    # Margins
    plot.margin = margin(25, 25, 25, 25)
  )

# Save overall plot
ggsave(file.path(output_folder, "xgboost_runvalue_stuff_plus_stabilization_overall_2025_nyt.png"), p_overall, 
       width = 12, height = 8, dpi = 300, bg = "white")

# Raw run value stabilization (not Stuff+)
analyze_raw_runvalue_stabilization <- function(df, min_season_pitches = 200) {
  results <- data.frame()
  pitch_thresholds <- c(25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
  
  year_data <- df %>%
    arrange(pitcher, game_date, at_bat_number, pitch_number) %>%
    group_by(pitcher) %>%
    mutate(pitch_order = row_number(), season_total = n()) %>%
    ungroup() %>%
    filter(season_total >= min_season_pitches)
  
  final_runvalue <- year_data %>%
    group_by(pitcher) %>%
    summarise(final_predicted_target = mean(predicted_target, na.rm = TRUE), .groups = 'drop')
  
  for(threshold in pitch_thresholds) {
    running_runvalue <- year_data %>%
      filter(pitch_order <= threshold) %>%
      group_by(pitcher) %>%
      summarise(running_predicted_target = mean(predicted_target, na.rm = TRUE),
                pitches_used = n(), .groups = 'drop') %>%
      filter(pitches_used == threshold)
    
    comparison <- running_runvalue %>%
      inner_join(final_runvalue, by = "pitcher")
    
    if(nrow(comparison) >= 10) {
      correlation <- cor(comparison$running_predicted_target, comparison$final_predicted_target)
      
      results <- rbind(results, data.frame(
        pitches = threshold,
        correlation = correlation
      ))
    }
  }
  return(results)
}

# Get raw run value results for 2025
runvalue_2025 <- analyze_raw_runvalue_stabilization(df)

# Create raw run value plot
p_runvalue <- ggplot(runvalue_2025, aes(x = pitches, y = correlation)) +
  geom_line(size = 3, alpha = 0.9, color = "#ff7f0e") +
  geom_point(size = 4, alpha = 0.9, color = "#ff7f0e") +
  
  # Reference lines
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "#666666", size = 0.8, alpha = 0.7) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "#333333", size = 0.8, alpha = 0.7) +
  
  # Scales
  scale_x_continuous(
    breaks = c(25, 50, 100, 150, 200, 300, 500),
    labels = c("25", "50", "100", "150", "200", "300", "500")
  ) +
  scale_y_continuous(
    limits = c(0.55, 1.02),
    breaks = seq(0.6, 1.0, 0.1),
    labels = percent_format(accuracy = 1),
    expand = c(0, 0)
  ) +
  
  # Labels
  labs(
    title = "XGBoost Run Value Model - Raw Predicted Run Value Stabilization (2025)",
    subtitle = "How many total pitches are needed for a reliable raw run value rating?",
    x = "Number of Pitches",
    y = "Correlation with Final Season Rating",
    caption = "Dashed lines show 80% and 90% correlation thresholds"
  ) +
  
  # NYT Theme
  theme_minimal(base_family = "Arial", base_size = 12) +
  theme(
    # Title and subtitle
    plot.title = element_text(
      size = 20, 
      face = "bold", 
      color = "#222222",
      margin = margin(b = 5)
    ),
    plot.subtitle = element_text(
      size = 14, 
      color = "#666666",
      margin = margin(b = 20)
    ),
    
    # Axes
    axis.title.x = element_text(size = 13, color = "#222222", margin = margin(t = 15)),
    axis.title.y = element_text(size = 13, color = "#222222", margin = margin(r = 15)),
    axis.text = element_text(size = 11, color = "#666666"),
    
    # Grid
    panel.grid.major = element_line(color = "#f0f0f0", size = 0.5),
    panel.grid.minor = element_blank(),
    
    # Caption
    plot.caption = element_text(
      size = 10, 
      color = "#999999", 
      hjust = 0,
      margin = margin(t = 15)
    ),
    
    # Background
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    
    # Margins
    plot.margin = margin(25, 25, 25, 25)
  )

# Save raw run value plot
ggsave(file.path(output_folder, "xgboost_runvalue_predicted_target_stabilization_2025_nyt.png"), p_runvalue, 
       width = 12, height = 8, dpi = 300, bg = "white")

# Save stabilization results as CSV files
write_csv(results_2025, file.path(output_folder, "pitch_type_stabilization_results_2025.csv"))
write_csv(overall_2025, file.path(output_folder, "overall_stabilization_results_2025.csv"))
write_csv(runvalue_2025, file.path(output_folder, "runvalue_stabilization_results_2025.csv"))

# Display plots
print(p_2025)
print(p_overall)
print(p_runvalue)

# Summary statistics
cat("\n", rep("=", 80), "\n")
cat("XGBOOST RUN VALUE MODEL STABILIZATION ANALYSIS SUMMARY - 2025\n")
cat(rep("=", 80), "\n")

cat("\nStuff+ Stabilization Results (2025):\n")
cat("Pitch Type Data Points:", nrow(results_2025), "\n")

if(nrow(results_2025) > 0) {
  # Find correlations at 100 pitches
  corr_100 <- results_2025$correlation[results_2025$pitches == 100]
  if(length(corr_100) > 0) {
    cat("Average Correlation at 100 pitches:", round(mean(corr_100, na.rm = TRUE), 3), "\n")
  }
  
  # Show pitch type specific results at 100 pitches
  cat("\nPitch Type Correlations at 100 pitches:\n")
  pitch_100 <- results_2025 %>% filter(pitches == 100)
  if(nrow(pitch_100) > 0) {
    for(i in 1:nrow(pitch_100)) {
      cat(paste0(pitch_100$pitch_type[i], ": ", round(pitch_100$correlation[i], 3)), "\n")
    }
  }
}

cat("\nOverall Stabilization Results (2025):\n")
cat("Overall Data Points:", nrow(overall_2025), "\n")

if(nrow(overall_2025) > 0 && 200 %in% overall_2025$pitches) {
  cat("Overall Correlation at 200 pitches:", 
      round(overall_2025$correlation[overall_2025$pitches == 200], 3), "\n")
}

cat("\nRaw Run Value Stabilization Results (2025):\n")
cat("Run Value Data Points:", nrow(runvalue_2025), "\n")

if(nrow(runvalue_2025) > 0 && 200 %in% runvalue_2025$pitches) {
  cat("Run Value Correlation at 200 pitches:", 
      round(runvalue_2025$correlation[runvalue_2025$pitches == 200], 3), "\n")
}

cat("\nFiles saved to:", output_folder, "\n")
cat("Plots:\n")
cat("- xgboost_runvalue_stuff_plus_stabilization_2025_nyt.png\n")
cat("- xgboost_runvalue_stuff_plus_stabilization_overall_2025_nyt.png\n")
cat("- xgboost_runvalue_predicted_target_stabilization_2025_nyt.png\n")
cat("Data:\n")
cat("- pitch_type_stabilization_results_2025.csv\n")
cat("- overall_stabilization_results_2025.csv\n")
cat("- runvalue_stabilization_results_2025.csv\n")

cat("\n", rep("=", 80), "\n")
cat("XGBOOST RUN VALUE MODEL STABILIZATION ANALYSIS COMPLETE - 2025\n")
cat(rep("=", 80), "\n")