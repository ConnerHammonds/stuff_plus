"""
model_validation.py
-------------------
Validates the per-pitch-type XGBoost models against a held-out test set.
Run from the project root:  python notebooks/model_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PITCH_TYPES = ['FB', 'SI', 'SL', 'CB', 'CH', 'CT', 'SP', 'SW', 'KN']
FB_TYPES          = {'FB', 'SI'}

FEATURES = [
    'release_speed', 'spin_rate', 'induced_vertical_break',
    'horizontal_break', 'extension', 'release_height',
    'release_side', 'spin_axis',
    'velo_diff', 'ivb_diff', 'hb_diff', 'spin_diff',
]

TARGET = 'delta_run_value'   # ← match whatever your training target column is called
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(BASE_DIR / "csv's" / 'tm_stuff_plus.csv')
print(f"Loaded {len(df):,} rows")

# ── Cleaning (mirror stuff.ipynb) ──────────────────────────────────────────
FLIP_COLS     = ['horizontal_break', 'release_side']
REQUIRED_COLS = ['pitch_type', 'pitcher_throws', 'release_speed',
                 'spin_rate', 'induced_vertical_break', 'horizontal_break', TARGET]

if 'is_warmup' in df.columns:
    df = df[df['is_warmup'] != 1]

df = df.dropna(subset=REQUIRED_COLS).copy()
df = df[df['pitcher_throws'].isin(['Right', 'Left'])].copy()

lhp = df['pitcher_throws'] == 'Left'
df.loc[lhp, FLIP_COLS]    = df.loc[lhp, FLIP_COLS] * -1
df.loc[lhp, 'spin_axis']  = (360 - df.loc[lhp, 'spin_axis']) % 360

# ── Differential features ──────────────────────────────────────────────────
fb_avg = (
    df[df['pitch_type'].isin(FB_TYPES)]
    .groupby('pitcher_name')[['release_speed', 'induced_vertical_break',
                               'horizontal_break', 'spin_rate']]
    .mean()
    .rename(columns={
        'release_speed':          'fb_velo',
        'induced_vertical_break': 'fb_ivb',
        'horizontal_break':       'fb_hb',
        'spin_rate':              'fb_spin',
    })
)

df = df.join(fb_avg, on='pitcher_name')
df['velo_diff'] = df['fb_velo'] - df['release_speed']
df['ivb_diff']  = df['fb_ivb']  - df['induced_vertical_break']
df['hb_diff']   = df['fb_hb']   - df['horizontal_break']
df['spin_diff'] = df['fb_spin'] - df['spin_rate']

fb_mask = df['pitch_type'].isin(FB_TYPES)
df.loc[fb_mask, ['velo_diff', 'ivb_diff', 'hb_diff', 'spin_diff']] = 0
df.drop(columns=['fb_velo', 'fb_ivb', 'fb_hb', 'fb_spin'], inplace=True)

# ── Validation loop ────────────────────────────────────────────────────────
results = []

for pt in MODEL_PITCH_TYPES:
    model_path = BASE_DIR / 'models' / f'{pt}.ubj'
    if not model_path.exists():
        print(f"[{pt}] model file not found — skipping")
        continue

    subset = df[df['pitch_type'] == pt].copy()
    if len(subset) < 50:
        print(f"[{pt}] not enough rows ({len(subset)}) — skipping")
        continue

    # Fill missing optional features with median
    X = subset[FEATURES].copy()
    for col in FEATURES:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    y = subset[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = XGBRegressor()
    model.load_model(model_path)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results.append({'pitch_type': pt, 'n_test': len(y_test),
                    'RMSE': rmse, 'MAE': mae, 'R2': r2})

    print(f"[{pt}]  n={len(y_test):>5,}  RMSE={rmse:.5f}  MAE={mae:.5f}  R²={r2:.4f}")

# ── Summary table ──────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
summary = pd.DataFrame(results).set_index('pitch_type')
print(summary.round(5).to_string())

avg = summary[['RMSE', 'MAE', 'R2']].mean()
print(f"\nMacro avg  RMSE={avg['RMSE']:.5f}  MAE={avg['MAE']:.5f}  R²={avg['R2']:.4f}")