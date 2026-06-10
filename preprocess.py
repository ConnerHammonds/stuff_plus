#Stuff+ preprocessing, inference, and scoring pipeline.

#Supports two input formats:
#  - 'trackman': TrackMan CSV format (pitcher_throws='Right'/'Left', columns already renamed)
#  - 'statcast': Raw MLB Statcast format (pitcher_throws='R'/'L', Statcast column names)

#Typical usage:
#    import preprocess as sp
#    pitch_df = sp.run_pipeline(raw_df, model_dir='models/v3')
#    summary_df = sp.build_summary(pitch_df)


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PITCH_TYPES = ['CB', 'CH', 'CT', 'FB', 'KN', 'SI', 'SL', 'SP', 'SW']
FB_TYPES          = {'FB', 'SI'}
FLIP_COLS         = ['horizontal_break', 'release_side']

FEATURES = [
    'release_speed',
    'spin_rate',
    'induced_vertical_break',
    'horizontal_break',
    'extension',
    'release_height',
    'release_side',
    'spin_axis',
    'velo_diff',
    'ivb_diff',
    'hb_diff',
    'spin_diff',
]

# Map StatCast pitch types to Trackman Pitch types
PITCH_TYPE_MAP = {
    'FF': 'FB',
    'FC': 'CT',
    'CU': 'CB',
    'ST': 'SW',
    'FS': 'SP',
}

# Map StatCast columns to TrackMan columns
STATCAST_COLUMN_MAP = {
    'player_name':       'pitcher_name',
    'p_throws':          'pitcher_throws',
    'release_spin_rate': 'spin_rate',
    'pfx_z':             'induced_vertical_break',
    'pfx_x':             'horizontal_break',
    'release_extension': 'extension',
    'release_pos_z':     'release_height',
    'release_pos_x':     'release_side',
}

COLLEGE_KEYWORDS = [
    'college', 'university', 'state', 'stix', 'vcu', 'fiu', 'mizzou',
    'ucm', 'siu', 'radford', 'lipscomb', 'bradley', 'gaston', 'brower', 'iowa',
    'memphis', 'jacksonville', 'nc state', 'west virg',
]

# ── Statcast-specific transforms ───────────────────────────────────────────────

def rename_statcast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Statcast columns to TM feature names."""
    return df.rename(columns=STATCAST_COLUMN_MAP)


def remap_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    """Map Statcast pitch type codes (FF, CU, …) to model codes (FB, CB, …)."""
    df = df.copy()
    df['pitch_type'] = df['pitch_type'].replace(PITCH_TYPE_MAP)
    return df


# ── Shared transforms ──────────────────────────────────────────────────────────

def normalize_level(level: str) -> str:
    """Bucket granular level labels into High School / College / Pro."""
    if pd.isna(level):
        return 'Unknown'
    lower = level.strip().lower()
    if lower == 'high school':
        return 'High School'
    if lower == 'pro':
        return 'Pro'
    if any(kw in lower for kw in COLLEGE_KEYWORDS):
        return 'College'
    return 'College'


def normalize_lhp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip horizontal metrics for LHP so all pitchers read as RHP convention.
    Accepts pitcher_throws values of either 'L' (Statcast) or 'Left' (TrackMan).
    """
    df = df.copy()
    lhp = df['pitcher_throws'].isin(['L', 'Left'])
    df.loc[lhp, FLIP_COLS]   = df.loc[lhp, FLIP_COLS] * -1
    df.loc[lhp, 'spin_axis'] = (360 - df.loc[lhp, 'spin_axis']) % 360
    return df


def compute_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add velo_diff, ivb_diff, hb_diff, spin_diff relative to each pitcher's
    mean FB/SI profile. Fastball rows are zeroed; pitchers with no FB get NaN
    (filled with column median at inference time).
    """
    df = df.copy()

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

    return df


# ── Preprocessing entry points ─────────────────────────────────────────────────

def preprocess_trackman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and featurize a TrackMan-format DataFrame.
    Expects pitcher_throws in {'Right', 'Left'} and TM column names.
    """
    df = df.copy()

    if 'is_warmup' in df.columns:
        df = df[df['is_warmup'] != 1]

    if 'pitcher_level' in df.columns:
        df['pitcher_level'] = df['pitcher_level'].apply(normalize_level)

    required = ['pitch_type', 'pitcher_throws', 'release_speed',
                'spin_rate', 'induced_vertical_break', 'horizontal_break']
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    df = df[df['pitcher_throws'].isin(['Right', 'Left'])].copy()

    df = normalize_lhp(df)
    df = compute_diff_features(df)
    return df


def preprocess_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and featurize a raw Statcast-format DataFrame.
    Expects pitcher_throws in {'R', 'L'} and Statcast column names.
    """
    df = rename_statcast_columns(df)
    df = remap_pitch_types(df)

    required = ['pitch_type', 'pitcher_throws', 'release_speed',
                'spin_rate', 'induced_vertical_break', 'horizontal_break']
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    df = df[df['pitcher_throws'].isin(['R', 'L'])].copy()

    df = normalize_lhp(df)
    df = compute_diff_features(df)
    return df


# ── Model loading and inference ────────────────────────────────────────────────

def load_models(model_dir: str) -> dict:
    """Load all .ubj XGBoost models from model_dir. Returns {pitch_type: model}."""
    models = {}
    for pt in MODEL_PITCH_TYPES:
        path = os.path.join(model_dir, f'{pt}.ubj')
        if not os.path.exists(path):
            print(f"Warning: model not found for '{pt}' at {path}")
            continue
        m = XGBRegressor()
        m.load_model(path)
        models[pt] = m
    return models


def predict_stuff_plus(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run XGBoost inference and add predicted_rv column.
    models is a dict from load_models(). Unrecognized pitch types get NaN.
    """
    df = df.copy()
    df['predicted_rv'] = np.nan

    for pt, model in models.items():
        mask = df['pitch_type'] == pt
        if mask.sum() == 0:
            continue

        X = df.loc[mask, FEATURES].copy()
        for col in FEATURES:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        df.loc[mask, 'predicted_rv'] = model.predict(X)

    return df


def scale_stuff_plus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert predicted_rv to Stuff+ scale (100 = average, ±10 per std dev).
    Scaling is applied independently within each (pitcher_level × pitch_type) group.
    Requires 'pitcher_level' column; falls back to a single global group if absent.
    """
    df = df.copy()
    df['stuff_plus'] = np.nan

    group_cols = ['pitcher_level', 'pitch_type'] if 'pitcher_level' in df.columns else ['pitch_type']

    for keys, grp in df.groupby(group_cols):
        raw = df.loc[grp.index, 'predicted_rv'].dropna()
        if len(raw) < 2 or raw.std() == 0:
            continue
        pop_mean = raw.mean()
        pop_std  = raw.std()
        scaled = 100 + ((pop_mean - raw) / pop_std) * 10
        df.loc[scaled.index, 'stuff_plus'] = scaled

    return df


# Data Pipeline
def run_pipeline(df: pd.DataFrame, model_dir: str, source: str = 'trackman') -> pd.DataFrame:
    """
    End-to-end pipeline: preprocess → predict → scale → return pitch-level df.

    Args:
        df:        Raw input DataFrame (TrackMan or Statcast format).
        model_dir: Path to directory containing .ubj model files.
        source:    'trackman' (default) or 'statcast'.

    Returns:
        DataFrame with all original columns plus predicted_rv and stuff_plus.
    """
    if source == 'statcast':
        df = preprocess_statcast(df)
    else:
        df = preprocess_trackman(df)

    models = load_models(model_dir)
    df = predict_stuff_plus(df, models)
    df = scale_stuff_plus(df)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level df to one row per pitcher × pitch type.
    Adds usage_pct and arsenal_stuff_plus (usage-weighted mean Stuff+).
    """
    summary = (
        df.dropna(subset=['stuff_plus'])
        .groupby(
            ['pitcher_name', 'pitcher_level', 'pitcher_throws', 'pitch_type']
            if 'pitcher_level' in df.columns
            else ['pitcher_name', 'pitcher_throws', 'pitch_type'],
            as_index=False,
        )
        .agg(
            pitches    = ('stuff_plus',             'count'),
            stuff_plus = ('stuff_plus',             'mean'),
            velocity   = ('release_speed',          'mean'),
            spin_rate  = ('spin_rate',              'mean'),
            ivb        = ('induced_vertical_break', 'mean'),
            h_break    = ('horizontal_break',       'mean'),
            extension  = ('extension',              'mean'),
        )
        .round(1)
        .sort_values(['pitcher_name', 'stuff_plus'], ascending=[True, False])
        .reset_index(drop=True)
    )

    total = summary.groupby('pitcher_name')['pitches'].transform('sum')
    summary['usage_pct'] = (summary['pitches'] / total).round(3)

    arsenal = (
        summary
        .groupby('pitcher_name')
        .apply(lambda g: round((g['stuff_plus'] * g['usage_pct']).sum(), 1))
        .rename('arsenal_stuff_plus')
        .reset_index()
    )
    summary = summary.merge(arsenal, on='pitcher_name')

    return summary
