"""
Feature engineering pipeline for the ERCOT load-forecasting project.
=====================================================================

Provides two complementary functions:

prepare_features
    Full offline preprocessing for the R&D / training pipeline.
    Accepts the raw merged DataFrame from :func:`~src.extraction.fetch_raw_data`
    and returns a clean, feature-rich DataFrame ready for model training.

build_live_features
    Inference-ready feature matrix builder for the production Streamlit
    dashboard.  Consumes the two live DataFrames returned by
    :func:`~src.extraction.fetch_live_data` and yields the future and past
    blocks the LightGBM model needs for prediction and quality control.

COLUMNS_ORDER
    The canonical list of feature names in the exact order expected by the
    serialised LightGBM booster.  Changing this list breaks inference.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature order — must match the column order used during training.
# WARNING: Do not reorder.  Any change here requires retraining the model.
# ---------------------------------------------------------------------------
COLUMNS_ORDER: List[str] = [
    "houston_temp", "houston_humidity", "houston_apparent_temp", "houston_wind_speed",
    "dallas_temp", "dallas_humidity", "dallas_apparent_temp", "dallas_wind_speed",
    "austin_temp", "austin_humidity", "austin_apparent_temp", "austin_wind_speed",
    "texas_avg_temp", "hour", "day_of_week", "month", "is_weekend",
    "load_lag_24", "load_lag_48", "load_lag_168",
    "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_max_24h",
    "is_holiday", "temp_delta_24h", "CDD", "HDD",
]

# EIA metadata columns that carry no predictive signal
_EIA_METADATA_COLS: List[str] = [
    "respondent", "respondent_name", "type", "type_name",
    "value_units", "timestamp_utc",
]

# Comfort base temperature for Degree Day calculations (65 °F = 18.3 °C)
_COMFORT_BASE_TEMP_C: float = 18.3


# ---------------------------------------------------------------------------
# Offline feature engineering (training pipeline)
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a raw merged DataFrame.

    The function is designed to be called on the output of
    :func:`~src.extraction.fetch_raw_data` and covers the following steps
    in order:

    1. Column name normalisation to ``snake_case``.
    2. Removal of EIA API metadata columns.
    3. Conversion of ``period`` to a ``DatetimeIndex``.
    4. Chronological sorting (prerequisite for all time-based operations).
    5. Computation of ``texas_avg_temp`` if not already present.
    6. Calendar variables: ``hour``, ``day_of_week``, ``month``, ``is_weekend``.
    7. Seasonal lags with a 24 h shift to prevent future data leakage:
       ``load_lag_24``, ``load_lag_48``, ``load_lag_168`` (1 week).
    8. Rolling statistics over the prior 24 h window (also shifted 24 h):
       ``load_rolling_mean_24h``, ``load_rolling_std_24h``,
       ``load_rolling_max_24h``.
    9. US Federal Holiday flag: ``is_holiday``.
    10. 24-hour thermal delta (detects sudden cold/heat fronts):
        ``temp_delta_24h``.
    11. Cooling and Heating Degree Days: ``CDD``, ``HDD``.
    12. Removal of the leading NaN rows introduced by the 168 h lag
        (< 0.5 % of data).

    Parameters
    ----------
    df:
        Raw merged DataFrame returned by
        :func:`~src.extraction.fetch_raw_data`.

    Returns
    -------
    pd.DataFrame
        Fully processed, NaN-free DataFrame with a ``DatetimeIndex``
        (column ``period``), containing both the target column ``value``
        and all engineered features.
    """
    df = df.copy()

    # 1. Normalise column names to snake_case
    df.columns = df.columns.str.replace("-", "_").str.lower()

    # 2. Drop EIA metadata columns (constants / duplicates, no predictive value)
    cols_to_drop = [c for c in _EIA_METADATA_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Set DatetimeIndex on the temporal key
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")

    # 4. Sort chronologically — critical for all shift/rolling ops
    df = df.sort_index()

 # 5. Compute/Fill texas_avg_temp if missing or contains NaNs
    _city_temp_cols = ["houston_temp", "dallas_temp", "austin_temp"]
    if all(c in df.columns for c in _city_temp_cols):
        avg_temp = df[_city_temp_cols].mean(axis=1)
        if "texas_avg_temp" not in df.columns:
            df["texas_avg_temp"] = avg_temp
        else:
            # Rellena únicamente los NaNs de 2026 con el promedio de las 3 ciudades
            df["texas_avg_temp"] = df["texas_avg_temp"].fillna(avg_temp)

    # 6. Calendar variables
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek   # 0 = Monday, 6 = Sunday
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 7. Seasonal lags — all shifted 24 h to respect the forecast horizon
    df["load_lag_24"] = df["value"].shift(24)
    df["load_lag_48"] = df["value"].shift(48)
    df["load_lag_168"] = df["value"].shift(168)   # Weekly pattern anchor

    # 8. Rolling statistics anchored at the close of the prior day (shift 24 h)
    _shifted = df["value"].shift(24)
    df["load_rolling_mean_24h"] = _shifted.rolling(window=24).mean()
    df["load_rolling_std_24h"] = _shifted.rolling(window=24).std()
    df["load_rolling_max_24h"] = _shifted.rolling(window=24).max()

    # 9. US Federal Holiday flag
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)

    # 10. 24-hour thermal delta — rate of temperature change (cold-front sensor)
    df["temp_delta_24h"] = df["texas_avg_temp"] - df["texas_avg_temp"].shift(24)

    # 11. Cooling / Heating Degree Days (linearise the U-shaped temp-load curve)
    df["CDD"] = np.maximum(0, df["texas_avg_temp"] - _COMFORT_BASE_TEMP_C)
    df["HDD"] = np.maximum(0, _COMFORT_BASE_TEMP_C - df["texas_avg_temp"])

    # ── DIAGNÓSTICO TEMPORAL ──
    print("\n" + "=" * 50)
    print("🔍 DIAGNÓSTICO DE NaNs EN 2026 (antes de dropna):")
    print(df.loc["2026-01-01":].isna().sum()[lambda x: x > 0])
    print("=" * 50 + "\n")
    #df.to_csv("diagnostico_nans_2026.csv", index=True)

    # 12. Remove leading NaN rows introduced by the 168 h lag
    n_before = len(df)
    df = df.dropna()
    logger.info(
        "prepare_features: %d → %d rows after dropna (removed %d leading NaNs).",
        n_before, len(df), n_before - len(df),
    )

    return df


# ---------------------------------------------------------------------------
# Live inference feature builder (dashboard pipeline)
# ---------------------------------------------------------------------------


def build_live_features(
    df_eia: pd.DataFrame,
    df_clima: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build inference-ready feature matrices from live API data.

    Joins demand (``df_eia``) and weather (``df_clima``) on their shared
    UTC index, applies the same feature engineering as
    :func:`prepare_features`, then splits the result into:

    * **Future block** — the 24 h immediately *after* the last confirmed EIA
      reading: the input to the 24-hour-ahead forecast.
    * **Past block** — the 24 h immediately *before* (and including) the
      last confirmed reading: used for real-time quality control (MAPE/MAE).

    All lag and rolling computations are shifted by 24 h so the model never
    receives demand values it would not have access to at inference time.
    A forward/back-fill safety pass is applied to guard against residual
    NaNs near the boundary between historical and forecast weather data.

    Parameters
    ----------
    df_eia:
        Hourly ERCOT demand indexed by UTC timestamp (``value`` column, MW).
        Returned by :func:`~src.extraction.fetch_live_data`.
    df_clima:
        Hourly weather for Houston, Dallas and Austin, indexed by UTC
        timestamp.  Returned by :func:`~src.extraction.fetch_live_data`.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        ``(X_future, X_past, y_past_real)`` where:

        * ``X_future``     — Feature matrix for the 24-h forecast horizon
          (columns in :data:`COLUMNS_ORDER` order).
        * ``X_past``       — Feature matrix for the last 24 h of known demand.
        * ``y_past_real``  — Actual MW demand series for the past 24 h
          (ground truth for quality-control metrics).
    """
    df_live = df_clima.join(df_eia["value"], how="left")

    # Calendar variables
    df_live["hour"] = df_live.index.hour
    df_live["day_of_week"] = df_live.index.dayofweek
    df_live["month"] = df_live.index.month
    df_live["is_weekend"] = df_live["day_of_week"].isin([5, 6]).astype(int)

    # Demand lags (24 h shift — no leakage into forecast horizon)
    df_live["load_lag_24"] = df_live["value"].shift(24)
    df_live["load_lag_48"] = df_live["value"].shift(48)
    df_live["load_lag_168"] = df_live["value"].shift(168)

    # Rolling statistics
    _shifted = df_live["value"].shift(24)
    df_live["load_rolling_mean_24h"] = _shifted.rolling(window=24).mean()
    df_live["load_rolling_std_24h"] = _shifted.rolling(window=24).std()
    df_live["load_rolling_max_24h"] = _shifted.rolling(window=24).max()

    # Derived weather features
    df_live["texas_avg_temp"] = (
        df_live[["houston_temp", "dallas_temp", "austin_temp"]].mean(axis=1)
    )
    df_live["temp_delta_24h"] = (
        df_live["texas_avg_temp"] - df_live["texas_avg_temp"].shift(24)
    )
    df_live["CDD"] = np.maximum(0, df_live["texas_avg_temp"] - _COMFORT_BASE_TEMP_C)
    df_live["HDD"] = np.maximum(0, _COMFORT_BASE_TEMP_C - df_live["texas_avg_temp"])

    # US Federal Holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df_live.index.min(), end=df_live.index.max())
    df_live["is_holiday"] = df_live.index.normalize().isin(holidays).astype(int)

    # Forward/back-fill safety pass — guards residual NaNs at the
    # historical/forecast boundary in weather data
    for col in COLUMNS_ORDER:
        if col in df_live.columns:
            df_live[col] = df_live[col].ffill().bfill()

    # Deterministic split anchored on the last confirmed EIA timestamp
    ultimo_ts_real = df_eia.index.max()
    df_future = df_live[df_live.index > ultimo_ts_real].head(24)
    df_past = df_live[df_live.index <= ultimo_ts_real].tail(24)

    return df_future[COLUMNS_ORDER], df_past[COLUMNS_ORDER], df_past["value"]
