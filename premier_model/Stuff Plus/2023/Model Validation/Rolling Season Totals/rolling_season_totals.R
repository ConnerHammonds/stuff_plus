library(httr)
library(jsonlite)
library(purrr)
library(glue)
library(dplyr)
library(readr)

# ========== DATA COLLECTION FUNCTIONS ==========

# Function to get all MLB teams (filter to only major league teams)
get_all_teams <- function(season = 2023) {
  url <- glue("https://statsapi.mlb.com/api/v1/teams?season={season}&sportId=1")
  response <- GET(url)
  parsed <- fromJSON(content(response, as = "text", encoding = "UTF-8"), flatten = TRUE)
  
  if("teams" %in% names(parsed)) {
    mlb_teams <- parsed$teams %>% 
      filter(sport.id == 1, active == TRUE) %>%
      select(id, name, abbreviation)
    return(mlb_teams)
  }
  return(NULL)
}

# Function to get all pitchers from a team
get_team_pitchers <- function(team_id, season = 2023) {
  url <- glue("https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}")
  
  tryCatch({
    response <- GET(url)
    parsed <- fromJSON(content(response, as = "text", encoding = "UTF-8"), flatten = TRUE)
    
    if("roster" %in% names(parsed)) {
      pitchers <- parsed$roster %>% 
        filter(position.code == "1") %>%
        select(person.id, person.fullName, jerseyNumber) %>%
        mutate(team_id = team_id)
      
      return(pitchers)
    }
    return(NULL)
  }, error = function(e) {
    message("Error getting pitchers for team ", team_id, ": ", e$message)
    return(NULL)
  })
}

# Function to get pitcher game logs
get_pitcher_game_logs <- function(player_id, season = 2023) {
  url <- glue(
    "https://statsapi.mlb.com/api/v1/people/{player_id}",
    "/stats?stats=gameLog&group=pitching&season={season}&language=en"
  )
  
  tryCatch({
    response <- GET(url)
    parsed <- fromJSON(content(response, as = "text", encoding = "UTF-8"), flatten = TRUE)
    
    if("stats" %in% names(parsed) && length(parsed$stats) > 0) {
      if("splits" %in% names(parsed$stats) && length(parsed$stats$splits) > 0) {
        splits_data <- parsed$stats$splits[[1]]
        
        if(nrow(splits_data) > 0) {
          splits_data$player_id <- player_id
          return(splits_data)
        }
      }
    }
    return(NULL)
  }, error = function(e) {
    message("Error getting game logs for player ", player_id, ": ", e$message)
    return(NULL)
  })
}

# ========== MAIN DATA COLLECTION ==========

cat("Step 1: Getting all MLB teams...\n")
teams <- get_all_teams(2023)
cat("Found", nrow(teams), "teams\n")

cat("\nStep 2: Getting all pitchers from all teams...\n")
all_pitchers <- map_dfr(teams$id, ~get_team_pitchers(.x, 2023))
cat("Found", nrow(all_pitchers), "total pitchers\n")

unique_pitchers <- all_pitchers %>% 
  distinct(person.id, .keep_all = TRUE)
cat("Unique pitchers:", nrow(unique_pitchers), "\n")

cat("\nStep 3: Getting game logs for all pitchers...\n")
cat("This will take several minutes...\n")

all_game_logs <- map_dfr(unique_pitchers$person.id, 
                         ~get_pitcher_game_logs(.x, 2023),
                         .progress = TRUE)

pitcher_logs <- all_game_logs %>%
  left_join(unique_pitchers %>% select(person.id, person.fullName), 
            by = c("player_id" = "person.id"))

# ========== DATA PROCESSING & CALCULATIONS ==========

cat("\nStep 4: Processing data and calculating advanced stats...\n")

pitcher_logs_with_fip <- pitcher_logs %>%
  mutate(
    date = as.Date(date),
    hr = as.numeric(stat.homeRuns),
    bb = as.numeric(stat.baseOnBalls),
    hbp = as.numeric(stat.hitBatsmen),
    k = as.numeric(stat.strikeOuts),
    ip_raw = as.numeric(stat.inningsPitched),
    ip = floor(ip_raw) + (ip_raw - floor(ip_raw)) * 10 / 3,
    er = as.numeric(stat.earnedRuns),
    doubles = as.numeric(stat.doubles),
    triples = as.numeric(stat.triples),
    hits = as.numeric(stat.hits),
    ab = as.numeric(stat.atBats),
    ibb = as.numeric(stat.intentionalWalks),
    sf = as.numeric(stat.sacFlies),
    ubb = bb - ibb,
    singles = hits - doubles - triples - hr,
    pitches = as.numeric(stat.numberOfPitches),
    strikes = as.numeric(stat.strikes)
  ) %>%
  arrange(player_id, date) %>%
  group_by(player_id) %>%
  mutate(
    game_fip = case_when(
      ip > 0 ~ ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + 3.166,
      TRUE ~ NA_real_
    ),
    game_number = row_number(),
    cum_hr = cumsum(coalesce(hr, 0)),
    cum_bb = cumsum(coalesce(bb, 0)),
    cum_hbp = cumsum(coalesce(hbp, 0)),
    cum_k = cumsum(coalesce(k, 0)),
    cum_ip = cumsum(coalesce(ip, 0)),
    cum_er = cumsum(coalesce(er, 0)),
    cum_ubb = cumsum(coalesce(ubb, 0)),
    cum_singles = cumsum(coalesce(singles, 0)),
    cum_doubles = cumsum(coalesce(doubles, 0)),
    cum_triples = cumsum(coalesce(triples, 0)),
    cum_ab = cumsum(coalesce(ab, 0)),
    cum_sf = cumsum(coalesce(sf, 0)),
    cum_ibb = cumsum(coalesce(ibb, 0)),
    season_fip_to_date = case_when(
      cum_ip > 0 ~ ((13 * cum_hr) + (3 * (cum_bb + cum_hbp)) - (2 * cum_k)) / cum_ip + 3.166,
      TRUE ~ NA_real_
    ),
    season_era_to_date = case_when(
      cum_ip > 0 ~ (cum_er * 9) / cum_ip,
      TRUE ~ NA_real_
    ),
    game_k_bb_ratio = case_when(
      bb > 0 ~ k / bb,
      bb == 0 & k > 0 ~ Inf,
      TRUE ~ NA_real_
    ),
    season_k_bb_ratio_to_date = case_when(
      cum_bb > 0 ~ cum_k / cum_bb,
      cum_bb == 0 & cum_k > 0 ~ Inf,
      TRUE ~ NA_real_
    ),
    game_woba = case_when(
      (ab + bb - ibb + sf + hbp) > 0 ~ 
        (.689 * ubb + .720 * hbp + .882 * singles + 1.254 * doubles + 1.590 * triples + 2.050 * hr) / 
        (ab + bb - ibb + sf + hbp),
      TRUE ~ NA_real_
    ),
    season_woba_to_date = case_when(
      (cum_ab + cum_bb - cum_ibb + cum_sf + cum_hbp) > 0 ~ 
        (.689 * cum_ubb + .720 * cum_hbp + .882 * cum_singles + 1.254 * cum_doubles + 1.590 * cum_triples + 2.050 * cum_hr) / 
        (cum_ab + cum_bb - cum_ibb + cum_sf + cum_hbp),
      TRUE ~ NA_real_
    )
  ) %>%
  ungroup()

# Create final dataframe
final_df <- pitcher_logs_with_fip %>%
  select(
    player_id,
    person.fullName,
    date,
    game_number,
    # Per-game stats
    game_ip = ip,
    game_hr = hr,
    game_bb = bb,
    game_k = k,
    game_hits = hits,
    game_er = er,
    game_pitches = pitches,
    game_strikes = strikes,
    game_fip,
    game_k_bb_ratio,
    game_woba,
    # Season cumulative stats
    season_ip = cum_ip,
    season_era = season_era_to_date,
    season_fip = season_fip_to_date,
    season_k_bb = season_k_bb_ratio_to_date,
    season_woba = season_woba_to_date
  )

# ========== SAVE DATA ==========

output_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2023/Model Validation/pitcher_performance_2023.csv"
write_csv(final_df, output_path)

cat("\nData saved to:", output_path, "\n")

# ========== CORRELATION ANALYSIS ==========

cat("\nStep 5: Running correlation analysis...\n")

# Get stats after 1 outing and final season stats
correlation_data <- final_df %>%
  group_by(player_id) %>%
  filter(n() >= 10) %>%
  filter(mean(game_ip, na.rm = TRUE) > 3) %>%
  summarise(
    player_name = first(person.fullName),
    # Stats after 1st outing
    era_after_1 = nth(season_era, 1),
    fip_after_1 = nth(season_fip, 1),
    k_bb_after_1 = nth(season_k_bb, 1),
    woba_after_1 = nth(season_woba, 1),
    # Final season stats
    final_woba = last(season_woba),
    total_outings = n(),
    avg_innings_per_outing = mean(game_ip, na.rm = TRUE),
    total_innings = last(season_ip),
    .groups = "drop"
  ) %>%
  filter(!is.na(era_after_1) & !is.na(fip_after_1) & 
           !is.na(k_bb_after_1) & !is.na(woba_after_1) & 
           !is.na(final_woba) &
           is.finite(k_bb_after_1) &
           total_innings > 100)

# Calculate correlations
era_cor <- cor(correlation_data$era_after_1, correlation_data$final_woba, use = "complete.obs")
fip_cor <- cor(correlation_data$fip_after_1, correlation_data$final_woba, use = "complete.obs")
k_bb_cor <- cor(correlation_data$k_bb_after_1, correlation_data$final_woba, use = "complete.obs")
woba_cor <- cor(correlation_data$woba_after_1, correlation_data$final_woba, use = "complete.obs")

# Get average pitches in first outing
avg_pitches_after_1 <- final_df %>%
  group_by(player_id) %>%
  filter(n() >= 10) %>%
  filter(mean(game_ip, na.rm = TRUE) > 3) %>%
  filter(last(season_ip) > 100) %>%
  summarise(total_pitches_after_1 = nth(game_pitches, 1)) %>%
  summarise(avg_total_pitches = mean(total_pitches_after_1, na.rm = TRUE)) %>%
  pull(avg_total_pitches)

# Create results table
results_table <- data.frame(
  Metric = c("ERA after 1 outing", "FIP after 1 outing", "K/BB after 1 outing", "wOBA after 1 outing"),
  Correlation_with_Final_wOBA = c(era_cor, fip_cor, k_bb_cor, woba_cor)
)

# Save correlation results
write_csv(correlation_data, "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2023/Model Validation/correlation_analysis_data_2023.csv")
write_csv(results_table, "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2023/Model Validation/correlation_results_2023.csv")

# Display results
cat("\n========== CORRELATION ANALYSIS RESULTS - 2023 ==========\n")
cat("Sample size:", nrow(correlation_data), "pitchers")
cat("\nPitchers with at least 10 outings, >3 innings/outing avg, >100 total innings")
cat("\nAverage pitches thrown in first outing:", round(avg_pitches_after_1, 1))

cat("\n\nCorrelations with Final Season wOBA:\n")
print(results_table)

cat("\n========== ANALYSIS COMPLETE - 2023 ==========\n")