"""
Data acquisition layer for the ERCOT load-forecasting pipeline.
================================================================

Provides three entry points:

fetch_raw_data
    Loads and merges the static historical CSV snapshots (2022-2025) used by
    the offline R&D / training pipeline (``ercot_exploration.ipynb``).

fetch_historical_data
    Downloads ERCOT demand (EIA) and Texas weather (Open-Meteo ERA5 archive)
    for an arbitrary date range.  Used by ``retrain.py`` to extend the static
    dataset with recent data (2026-present) before retraining.

fetch_live_data
    Pulls recent ERCOT demand (EIA API) and Texas weather
    (Open-Meteo → Visual Crossing → local backup) for the production
    Streamlit dashboard.  All Streamlit dependencies are intentionally
    absent from this module; status is emitted via Python's standard
    ``logging`` facility so callers can route messages anywhere.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_VISUAL_CROSSING_BASE_URL = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices"
    "/rest/services/timeline"
)

_HTTP_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Static data loader (offline / training pipeline)
# ---------------------------------------------------------------------------


def fetch_raw_data(
    ercot_csv_path: str = "data/ercot_load_2022_2025_static.csv",
    weather_csv_path: str = "data/texas_weather_2022_2025_static.csv",
) -> pd.DataFrame:
    """Load and merge static ERCOT load and Texas weather CSV snapshots.

    Performs an inner join on the shared hourly timestamp so that only
    records with **both** demand *and* weather data are retained.  Column
    names, index, and null values are left untouched; pass the result to
    :func:`~src.processing.prepare_features` for the full processing
    pipeline.

    Parameters
    ----------
    ercot_csv_path:
        Path to the ERCOT load CSV.  The column ``period`` must contain
        UTC hourly timestamps and ``value`` must hold demand in MW.
    weather_csv_path:
        Path to the Texas weather CSV.  The column ``timestamp_utc`` must
        contain UTC hourly timestamps.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed merged DataFrame ready for
        :func:`~src.processing.prepare_features`.

    Raises
    ------
    FileNotFoundError
        If either CSV file does not exist at the given path.
    """
    for path in (ercot_csv_path, weather_csv_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    df_electricity = pd.read_csv(ercot_csv_path, parse_dates=["period"])
    df_weather = pd.read_csv(weather_csv_path, parse_dates=["timestamp_utc"])

    df_full = pd.merge(
        df_electricity,
        df_weather,
        left_on="period",
        right_on="timestamp_utc",
        how="inner",
    )

    logger.info(
        "fetch_raw_data: merged %d hourly records with %d columns.",
        len(df_full),
        df_full.shape[1],
    )
    return df_full


# ---------------------------------------------------------------------------
# City coordinates used by both historical and live weather fetchers
# ---------------------------------------------------------------------------
_CITIES: List[Tuple[str, float, float]] = [
    ("houston", 29.7604, -95.3698),
    ("dallas",  32.7767, -96.7970),
    ("austin",  30.2672, -97.7431),
]

_WEATHER_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"
)


# ---------------------------------------------------------------------------
# Historical range fetcher (retraining pipeline)
# ---------------------------------------------------------------------------


def fetch_historical_data(
    start_date: str,
    end_date: str,
    eia_api_key: str,
) -> pd.DataFrame:
    """Download ERCOT demand and Texas weather for an arbitrary date range.

    Intended for extending the static 2022-2025 dataset with recent data
    (e.g. 2026-present) before periodic model retraining.

    Uses the **EIA v2 API** for hourly demand (paginated, handles ranges
    longer than 5,000 hours) and the **Open-Meteo ERA5 reanalysis archive**
    for weather.  The archive endpoint typically lags real-time by 3-5 days;
    set ``end_date`` accordingly.

    Parameters
    ----------
    start_date:
        First day of the desired range, inclusive.  Format: ``"YYYY-MM-DD"``.
    end_date:
        Last day of the desired range, inclusive.  Format: ``"YYYY-MM-DD"``.
        Should be at least 5 days before today to ensure archive availability.
    eia_api_key:
        EIA API authentication key.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``period`` (UTC datetime), ``value`` (MW) and
        twelve weather columns (``{city}_{var}`` for Houston, Dallas, Austin).
        Compatible with :func:`~src.processing.prepare_features`.

    Raises
    ------
    requests.HTTPError
        If any API call fails with a non-2xx HTTP status.
    """
    logger.info(
        "fetch_historical_data: fetching %s → %s from EIA + Open-Meteo archive.",
        start_date, end_date,
    )
    df_eia = _fetch_eia_range(start_date, end_date, eia_api_key)
    df_weather = _fetch_weather_archive(start_date, end_date)

    df = pd.merge(df_eia, df_weather, on="period", how="inner")
    logger.info(
        "fetch_historical_data: merged %d hourly records (%s → %s).",
        len(df), start_date, end_date,
    )
    return df


def _fetch_eia_range(
    start_date: str,
    end_date: str,
    api_key: str,
    page_size: int = 5000,
) -> pd.DataFrame:
    """Paginated EIA demand fetch for an arbitrary date range.

    Iterates through pages until all records are collected, so the function
    handles ranges longer than 5,000 hours without data loss.
    """
    start_str = f"{start_date}T00"
    end_str = f"{end_date}T23"
    all_records: list = []
    offset = 0

    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "facets[type][]": "D",
            "start": start_str,
            "end": end_str,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": page_size,
            "offset": offset,
        }
        resp = requests.get(
            _EIA_URL, params=params, headers=_HTTP_HEADERS, timeout=(5, 30)
        )
        resp.raise_for_status()
        page = resp.json()["response"]["data"]
        all_records.extend(page)
        logger.debug(
            "_fetch_eia_range: offset=%d fetched %d records (total %d).",
            offset, len(page), len(all_records),
        )
        if len(page) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_records)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (
        df[["period", "value"]]
        .dropna()
        .sort_values("period")
        .reset_index(drop=True)
    )


def _fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly ERA5 reanalysis weather for Houston, Dallas and Austin.

    Uses the Open-Meteo historical archive endpoint, which provides data
    from 1940 up to approximately 5 days before the current date.
    """
    df_weather: Optional[pd.DataFrame] = None

    for ciudad, lat, lon in _CITIES:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": _WEATHER_HOURLY_VARS,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        resp = requests.get(
            _OPEN_METEO_ARCHIVE_URL,
            params=params,
            headers=_HTTP_HEADERS,
            timeout=(5, 30),
        )
        resp.raise_for_status()
        data = resp.json()

        df_city = pd.DataFrame({
            "period": pd.to_datetime(data["hourly"]["time"]),
            f"{ciudad}_temp": data["hourly"]["temperature_2m"],
            f"{ciudad}_humidity": data["hourly"]["relative_humidity_2m"],
            f"{ciudad}_apparent_temp": data["hourly"]["apparent_temperature"],
            f"{ciudad}_wind_speed": data["hourly"]["wind_speed_10m"],
        })
        df_weather = (
            df_city if df_weather is None
            else df_weather.merge(df_city, on="period", how="inner")
        )
        logger.debug("_fetch_weather_archive: fetched %s (%d rows).", ciudad, len(df_city))

    return df_weather


# ---------------------------------------------------------------------------
# Live data fetcher (production / dashboard pipeline)
# ---------------------------------------------------------------------------


def fetch_live_data(
    eia_api_key: str,
    visual_crossing_key: str,
    backup_eia_path: str = "data/backup_live_eia.csv",
    backup_clima_path: str = "data/backup_live_clima.csv",
    lookback_days: int = 8,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
    """Pull recent ERCOT demand and Texas weather with a three-level fallback.

    Climate fallback order:

    1. **Open-Meteo** (primary — free, no key required)
    2. **Visual Crossing** (secondary — requires ``visual_crossing_key``)
    3. **Local backup CSV** (tertiary — last successful response cached on disk)

    EIA demand uses the same two-level strategy (live API → local backup).

    After each successful live call the response is persisted to the backup
    paths so contingency data stays fresh.  Timestamps are floored to the
    hour to prevent microsecond misalignments between the two sources.

    Parameters
    ----------
    eia_api_key:
        EIA API authentication key for the demand endpoint.
    visual_crossing_key:
        Visual Crossing API key (only used when Open-Meteo is unavailable).
    backup_eia_path:
        Path where the latest EIA snapshot is cached.
    backup_clima_path:
        Path where the latest climate snapshot is cached.
    lookback_days:
        How many past days to request.  Must be ≥ 7 to satisfy the 168-hour
        lag requirement of the feature engineering pipeline.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]
        A three-element tuple ``(df_eia, df_clima, using_backup)`` where:

        * ``df_eia``       — Hourly ERCOT demand indexed by UTC timestamp
          (column ``value`` in MW).
        * ``df_clima``     — Hourly weather for Houston, Dallas and Austin,
          indexed by UTC timestamp.
        * ``using_backup`` — ``True`` if *any* source fell back to cached data.

        Returns ``(None, None, False)`` when all sources are unavailable
        and no backup exists.
    """
    hoy = datetime.utcnow()
    hace_n_dias = hoy - timedelta(days=lookback_days)
    usando_backup = False

    # ------------------------------------------------------------------ EIA --
    df_eia: Optional[pd.DataFrame] = None

    params_eia = {
        "api_key": eia_api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "D",
        "start": hace_n_dias.strftime("%Y-%m-%dT%H"),
        "end": hoy.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    try:
        response_eia = requests.get(
            _EIA_URL, params=params_eia, headers=_HTTP_HEADERS, timeout=(5, 15)
        )
        response_eia.raise_for_status()
        raw_eia = response_eia.json()

        df_eia = pd.DataFrame(raw_eia["response"]["data"])
        df_eia["timestamp"] = pd.to_datetime(df_eia["period"])
        df_eia = df_eia.set_index("timestamp").sort_index()
        df_eia["value"] = df_eia["value"].astype(float)

        os.makedirs(os.path.dirname(backup_eia_path) or ".", exist_ok=True)
        df_eia.to_csv(backup_eia_path)
        logger.info("fetch_live_data: EIA demand fetched successfully (%d rows).", len(df_eia))

    except Exception as exc:
        logger.warning(
            "fetch_live_data: EIA API unavailable — %s. Loading local backup.", exc
        )
        if os.path.exists(backup_eia_path):
            df_eia = pd.read_csv(backup_eia_path, index_col="timestamp", parse_dates=True)
            usando_backup = True
        else:
            logger.error(
                "fetch_live_data: EIA backup not found at '%s'. Cannot proceed.", backup_eia_path
            )
            return None, None, False

    # --------------------------------------------------------------- Climate --
    df_clima: Optional[pd.DataFrame] = None
    api_clima_exito = False

    # Level 1: Open-Meteo (primary — no API key required)
    params_om = {
        "latitude": [29.7604, 32.7767, 30.2672],
        "longitude": [-95.3698, -96.7970, -97.7431],
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "past_days": lookback_days,
        "forecast_days": 2,
        "timezone": "UTC",
    }
    try:
        response_om = requests.get(
            _OPEN_METEO_URL, params=params_om, headers=_HTTP_HEADERS, timeout=(5, 15)
        )
        response_om.raise_for_status()
        res_om = response_om.json()

        ciudades = ["houston", "dallas", "austin"]
        df_clima = pd.DataFrame(
            {"timestamp": pd.to_datetime(res_om[0]["hourly"]["time"])}
        )
        for i, ciudad in enumerate(ciudades):
            h = res_om[i]["hourly"]
            df_clima[f"{ciudad}_temp"] = h["temperature_2m"]
            df_clima[f"{ciudad}_humidity"] = h["relative_humidity_2m"]
            df_clima[f"{ciudad}_apparent_temp"] = h["apparent_temperature"]
            df_clima[f"{ciudad}_wind_speed"] = h["wind_speed_10m"]

        df_clima = df_clima.set_index("timestamp").sort_index()
        api_clima_exito = True
        logger.info("fetch_live_data: climate fetched via Open-Meteo (primary).")

    except Exception as exc_om:
        logger.warning(
            "fetch_live_data: Open-Meteo failed — %s. Trying Visual Crossing.", exc_om
        )

        # Level 2: Visual Crossing (secondary fallback)
        start_str = hace_n_dias.strftime("%Y-%m-%d")
        end_str = (hoy + timedelta(days=2)).strftime("%Y-%m-%d")
        ciudades = ["houston", "dallas", "austin"]
        df_clima_vc: Optional[pd.DataFrame] = None
        vc_exito = True

        for ciudad in ciudades:
            url_vc = f"{_VISUAL_CROSSING_BASE_URL}/{ciudad}/{start_str}/{end_str}"
            params_vc = {
                "key": visual_crossing_key,
                "unitGroup": "metric",
                "include": "hours",
                "contentType": "json",
                "timezone": "Z",
            }
            try:
                response_vc = requests.get(
                    url_vc, params=params_vc, headers=_HTTP_HEADERS, timeout=(5, 15)
                )
                response_vc.raise_for_status()
                res_vc = response_vc.json()

                hours_data = []
                for day in res_vc["days"]:
                    for hour in day["hours"]:
                        ts_str = f"{day['datetime']} {hour['datetime']}"
                        hours_data.append(
                            {
                                "timestamp": pd.to_datetime(ts_str),
                                f"{ciudad}_temp": float(hour["temp"]),
                                f"{ciudad}_humidity": float(hour["humidity"]),
                                f"{ciudad}_apparent_temp": float(hour["feelslike"]),
                                f"{ciudad}_wind_speed": float(hour["windspeed"]),
                            }
                        )

                df_ciudad = pd.DataFrame(hours_data).set_index("timestamp")
                df_clima_vc = (
                    df_ciudad if df_clima_vc is None else df_clima_vc.join(df_ciudad, how="outer")
                )
                time.sleep(1)  # Respect Visual Crossing rate limits

            except Exception as exc_vc:
                logger.error(
                    "fetch_live_data: Visual Crossing failed for %s — %s", ciudad, exc_vc
                )
                vc_exito = False
                break

        if vc_exito and df_clima_vc is not None:
            df_clima = df_clima_vc
            api_clima_exito = True
            logger.info("fetch_live_data: climate fetched via Visual Crossing (secondary).")

    # Level 3: Local backup
    if not api_clima_exito:
        if os.path.exists(backup_clima_path):
            df_clima = pd.read_csv(
                backup_clima_path, index_col="timestamp", parse_dates=True
            )
            usando_backup = True
            logger.warning("fetch_live_data: using local climate backup from '%s'.", backup_clima_path)
        else:
            logger.error(
                "fetch_live_data: all climate sources failed and no backup found at '%s'.",
                backup_clima_path,
            )
            return None, None, False
    else:
        # Persist a fresh copy for future contingency use
        os.makedirs(os.path.dirname(backup_clima_path) or ".", exist_ok=True)
        df_clima.to_csv(backup_clima_path)

    # Floor timestamps to prevent microsecond misalignment between sources
    df_eia.index = df_eia.index.floor("h")
    df_clima.index = df_clima.index.floor("h")

    return df_eia, df_clima, usando_backup
