# =============================================================================
# Historical Stuff+ Database Upload Script
# Uploads combined 2017-2022 CSV to PostgreSQL database
# Creates scaling parameters and calculates Stuff+ for all historical years
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGE LOADING AND SETUP
# -----------------------------------------------------------------------------

# Load required packages
packages <- c("DBI", "RPostgres", "readr", "dplyr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages)
}

library(DBI)
library(RPostgres)
library(readr)
library(dplyr)

# Load environment variables from .env file
env_path <- file.path(dirname(dirname(dirname(dirname(getwd())))), ".env")
if (file.exists(env_path)) {
  readRenviron(env_path)
}

cat("=== Historical Stuff+ Database Upload Script ===\n")
cat("Loading 2017-2022 combined results into PostgreSQL\n\n")

# -----------------------------------------------------------------------------
# 2. DATABASE CONNECTION
# -----------------------------------------------------------------------------

# Database connection credentials
pg_con <- dbConnect(
  RPostgres::Postgres(),
  host     = Sys.getenv("DB_HOST"),
  port     = 5432,
  dbname   = "postgres",
  user     = Sys.getenv("DB_USER"),
  password = Sys.getenv("DB_PASSWORD"),
  sslmode  = "require"
)

# Define table references
pred_tbl  <- Id(schema = "public", table = "stuff_plus_predictions")
scale_tbl <- Id(schema = "public", table = "stuff_plus_scaling")

cat("Database connection established\n")

# -----------------------------------------------------------------------------
# 3. FILE PATHS AND VALIDATION
# -----------------------------------------------------------------------------

# Path to the combined historical CSV
historical_csv_path <- "C:/Users/aasmi/p3_summer_2025/Pitch Modeling Organaized/Stuff Plus/2017on/test_results_combined_2017_2022.csv"

# Check if file exists
if(!file.exists(historical_csv_path)) {
  stop("Historical CSV not found at: ", historical_csv_path, 
       "\nPlease run the comprehensive modeling script first.")
}

cat("Historical CSV found:", historical_csv_path, "\n")

# -----------------------------------------------------------------------------
# 4. DEFINE COLUMNS TO KEEP
# -----------------------------------------------------------------------------

# Define the columns we want to keep (same as other scripts)
keep_cols <- c(
  "game_date","year","pitch_number","pitcher","player_name","pitch_type","p_throws","stand",
  "release_speed","release_spin_rate","release_extension","release_pos_x","release_pos_z",
  "pfx_x","pfx_z","plate_x","plate_z","zone","description","predicted_target",
  "pfx_x_inches","pfx_z_inches","arm_slot_Vector","speed_diff","pfx_z_diff","pfx_x_diff"
)

cat("Columns to upload:", length(keep_cols), "\n")

# -----------------------------------------------------------------------------
# 5. REMOVE EXISTING HISTORICAL DATA
# -----------------------------------------------------------------------------

cat("\nRemoving existing historical data (2017-2022)...\n")

# Remove historical years from predictions table
historical_years <- 2017:2022
for(year in historical_years) {
  rows_deleted <- dbExecute(pg_con, paste0("
    DELETE FROM public.stuff_plus_predictions 
     WHERE year = ", year, ";
  "))
  cat("Deleted", rows_deleted, "rows for year", year, "\n")
}

# Remove historical years from scaling table
for(year in historical_years) {
  rows_deleted <- dbExecute(pg_con, paste0("
    DELETE FROM public.stuff_plus_scaling 
     WHERE year = ", year, ";
  "))
  cat("Deleted scaling parameters for year", year, "\n")
}

cat("Historical data cleanup complete\n")

# -----------------------------------------------------------------------------
# 6. CHUNKED DATA UPLOAD
# -----------------------------------------------------------------------------

cat("\nStarting chunked upload of historical data...\n")

# Define chunk size for reading/uploading
chunk_size <- 50000

# Initialize counters
total_rows_uploaded <- 0
chunk_counter <- 0

# Chunked upload function
upload_chunk <- function(df, pos) {
  # Increment chunk counter
  chunk_counter <<- chunk_counter + 1
  
  # Filter to keep only desired columns
  df <- df[, keep_cols, drop = FALSE]
  
  # Upload chunk to database
  dbWriteTable(pg_con, name = pred_tbl, value = df,
               append = TRUE, row.names = FALSE)
  
  # Update total counter
  rows_in_chunk <- nrow(df)
  total_rows_uploaded <<- total_rows_uploaded + rows_in_chunk
  
  # Progress update
  cat("Chunk", chunk_counter, "uploaded:", rows_in_chunk, "rows", 
      "(Total:", total_rows_uploaded, ")\n")
}

# Read and upload the CSV in chunks
cat("Reading and uploading CSV in chunks of", chunk_size, "rows...\n")

read_csv_chunked(
  historical_csv_path,
  DataFrameCallback$new(upload_chunk),
  chunk_size = chunk_size,
  show_col_types = FALSE,
  progress = TRUE
)

cat("\nData upload complete! Total rows uploaded:", total_rows_uploaded, "\n")

# -----------------------------------------------------------------------------
# 7. VERIFY DATA UPLOAD
# -----------------------------------------------------------------------------

cat("\nVerifying data upload...\n")

# Check row counts by year
year_counts <- dbGetQuery(pg_con, "
  SELECT year, COUNT(*) as row_count
    FROM public.stuff_plus_predictions
   WHERE year BETWEEN 2017 AND 2022
GROUP BY year
ORDER BY year;
")

cat("Rows uploaded by year:\n")
print(year_counts)

total_historical_rows <- sum(year_counts$row_count)
cat("Total historical rows in database:", total_historical_rows, "\n")

# -----------------------------------------------------------------------------
# 8. CREATE SCALING PARAMETERS
# -----------------------------------------------------------------------------

cat("\nCreating scaling parameters for historical years...\n")

# Calculate and insert scaling parameters for each year
dbExecute(pg_con, "
  INSERT INTO public.stuff_plus_scaling (year, mean_predicted_target, sd_predicted_target, n_pitches)
  SELECT year,
         AVG(predicted_target) AS mean_predicted_target,
         STDDEV_POP(predicted_target) AS sd_predicted_target,
         COUNT(*) AS n_pitches
    FROM public.stuff_plus_predictions
   WHERE year BETWEEN 2017 AND 2022
GROUP BY year
  ON CONFLICT (year) DO UPDATE
      SET mean_predicted_target = EXCLUDED.mean_predicted_target,
          sd_predicted_target   = EXCLUDED.sd_predicted_target,
          n_pitches             = EXCLUDED.n_pitches,
          updated_at            = NOW();
")

cat("Scaling parameters created for all historical years\n")

# Verify scaling parameters
scaling_params <- dbGetQuery(pg_con, "
  SELECT year, 
         ROUND(mean_predicted_target::numeric, 6) as mean_target,
         ROUND(sd_predicted_target::numeric, 6) as sd_target,
         n_pitches,
         updated_at
    FROM public.stuff_plus_scaling
   WHERE year BETWEEN 2017 AND 2022
ORDER BY year;
")

cat("\nScaling parameters by year:\n")
print(scaling_params)

# -----------------------------------------------------------------------------
# 9. CALCULATE STUFF+ FOR ALL HISTORICAL YEARS
# -----------------------------------------------------------------------------

cat("\nCalculating Stuff+ for all historical years...\n")

# Update Stuff+ for all historical records
rows_updated <- dbExecute(pg_con, "
  UPDATE public.stuff_plus_predictions AS sp
     SET stuff_plus = 
         100 - 10 * (sp.predicted_target - sc.mean_predicted_target) / sc.sd_predicted_target
    FROM public.stuff_plus_scaling AS sc
   WHERE sp.year = sc.year
     AND sp.year BETWEEN 2017 AND 2022;
")

cat("Stuff+ calculated for", rows_updated, "historical records\n")

# -----------------------------------------------------------------------------
# 10. FINAL VERIFICATION
# -----------------------------------------------------------------------------

cat("\nPerforming final verification...\n")

# Check for any records with missing Stuff+ values
missing_stuff_plus <- dbGetQuery(pg_con, "
  SELECT year, COUNT(*) as missing_count
    FROM public.stuff_plus_predictions
   WHERE year BETWEEN 2017 AND 2022
     AND stuff_plus IS NULL
GROUP BY year
ORDER BY year;
")

if(nrow(missing_stuff_plus) > 0) {
  cat("WARNING: Found records with missing Stuff+ values:\n")
  print(missing_stuff_plus)
} else {
  cat("✓ All historical records have Stuff+ values calculated\n")
}

# Sample some records to verify
sample_records <- dbGetQuery(pg_con, "
  SELECT year, 
         COUNT(*) as total_pitches,
         ROUND(AVG(stuff_plus)::numeric, 2) as avg_stuff_plus,
         ROUND(MIN(stuff_plus)::numeric, 2) as min_stuff_plus,
         ROUND(MAX(stuff_plus)::numeric, 2) as max_stuff_plus
    FROM public.stuff_plus_predictions
   WHERE year BETWEEN 2017 AND 2022
     AND stuff_plus IS NOT NULL
GROUP BY year
ORDER BY year;
")

cat("\nStuff+ summary statistics by year:\n")
print(sample_records)

# -----------------------------------------------------------------------------
# 11. CHECK FOREIGN KEY RELATIONSHIPS
# -----------------------------------------------------------------------------

cat("\nVerifying foreign key relationships...\n")

# Check that all prediction years have corresponding scaling parameters
orphaned_predictions <- dbGetQuery(pg_con, "
  SELECT p.year, COUNT(*) as orphaned_count
    FROM public.stuff_plus_predictions p
    LEFT JOIN public.stuff_plus_scaling s ON p.year = s.year
   WHERE p.year BETWEEN 2017 AND 2022
     AND s.year IS NULL
GROUP BY p.year
ORDER BY p.year;
")

if(nrow(orphaned_predictions) > 0) {
  cat("WARNING: Found orphaned predictions (no scaling parameters):\n")
  print(orphaned_predictions)
} else {
  cat("✓ All historical predictions have corresponding scaling parameters\n")
}

# Check scaling parameters without predictions
orphaned_scaling <- dbGetQuery(pg_con, "
  SELECT s.year, s.n_pitches as recorded_pitches,
         COALESCE(COUNT(p.year), 0) as actual_pitches
    FROM public.stuff_plus_scaling s
    LEFT JOIN public.stuff_plus_predictions p ON s.year = p.year
   WHERE s.year BETWEEN 2017 AND 2022
GROUP BY s.year, s.n_pitches
  HAVING s.n_pitches != COALESCE(COUNT(p.year), 0)
ORDER BY s.year;
")

if(nrow(orphaned_scaling) > 0) {
  cat("WARNING: Scaling parameter counts don't match actual predictions:\n")
  print(orphaned_scaling)
} else {
  cat("✓ All scaling parameters match actual prediction counts\n")
}

# -----------------------------------------------------------------------------
# 12. CREATE SUMMARY REPORT
# -----------------------------------------------------------------------------

cat("\n", rep("=", 80), "\n")
cat("HISTORICAL STUFF+ DATABASE UPLOAD COMPLETE\n")
cat(rep("=", 80), "\n\n")

cat("UPLOAD SUMMARY:\n")
cat("- Source file:", historical_csv_path, "\n")
cat("- Years uploaded: 2017-2022\n")
cat("- Total rows uploaded:", total_historical_rows, "\n")
cat("- Columns per row:", length(keep_cols), "\n")
cat("- Chunk size used:", chunk_size, "\n\n")

cat("DATABASE TABLES UPDATED:\n")
cat("- stuff_plus_predictions: Historical pitch data with predictions and Stuff+\n")
cat("- stuff_plus_scaling: Year-specific scaling parameters (2017-2022)\n\n")

cat("SCALING PARAMETERS CREATED FOR YEARS:\n")
for(i in 1:nrow(scaling_params)) {
  year_data <- scaling_params[i, ]
  cat("- ", year_data$year, ": mean=", year_data$mean_target, 
      ", sd=", year_data$sd_target, ", n=", year_data$n_pitches, "\n")
}

cat("\nSTUFF+ CALCULATION:\n")
cat("Formula: stuff_plus = 100 - 10 * (predicted_target - year_mean) / year_sd\n")
cat("Applied to", rows_updated, "historical records\n\n")

cat("NEXT STEPS:\n")
cat("1. Verify data quality by spot-checking some records\n")
cat("2. Run your individual year scripts for 2023-2025 to complete the database\n")
cat("3. Set up the 2025 daily refresh script for ongoing updates\n")
cat("4. Consider creating indexes for better query performance\n\n")

cat("DATABASE READY FOR ANALYSIS!\n")
cat(rep("=", 80), "\n")

# -----------------------------------------------------------------------------
# 13. CLEANUP AND DISCONNECT
# -----------------------------------------------------------------------------

# Disconnect from database
dbDisconnect(pg_con)
cat("\nDatabase connection closed\n")
cat("Historical upload script complete!\n")