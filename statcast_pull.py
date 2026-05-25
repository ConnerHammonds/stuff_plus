# This file contains the script that pulled the StatCast data from MLB
# games from 2024 through present day (May 2026)

# The data was exported as a CSV called "mlb_2024-2026" in the csv's directory

from pybaseball import statcast
import pandas as pd

COLS = [
    # Grouping / Informational
    'game_date',
    'player_name',       # Pitcher name
    'pitcher',           # Pitcher ID number
    'p_throws',
    'pitch_type',

    # Features
    'release_speed',         # velocity
    'release_spin_rate',     # spin_rate
    'pfx_z',                 # induced_vertical_break (feet, catcher's POV)
    'pfx_x',                 # horizontal_break (feet, positive = 1B side)
    'release_extension',     # extension
    'release_pos_z',         # release_height
    'release_pos_x',         # release_side
    'effective_speed',       # effective velocity (factors in extension)
    'release_spin_axis',     # spin_axis

    # Target variable
    'delta_run_exp',         # ΔRV
]

# Pull each season up to the present day
df_2024 = statcast(start_dt='2024-03-01', end_dt='2024-11-30')
df_2025 = statcast(start_dt='2025-03-01', end_dt='2025-11-30')
df_2026 = statcast(start_dt='2026-03-01', end_dt='2026-05-19')

df_mlb = pd.concat([df_2024, df_2025, df_2026], ignore_index=True)

df_mlb = df_mlb[COLS]

df_mlb.to_csv("mlb_2024-2026.csv", index=False)