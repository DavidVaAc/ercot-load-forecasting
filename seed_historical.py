"""
seed_historical.py — Generación de los datasets estáticos históricos (2022-2025).

Uso:
    python seed_historical.py

Reemplaza a los notebooks ``EIA_API.ipynb`` y ``OPEN_METEO_API.ipynb``,
modularizando en un único script la generación de los dos CSV maestros que
alimentan todo el pipeline de R&D y producción:

    1. data/ercot_load_2022_2025_static.csv     ← Demanda ERCOT (API de la EIA)
    2. data/texas_weather_2022_2025_static.csv  ← Clima (Open-Meteo ERA5 archive)

Ambos archivos conservan exactamente el mismo esquema de columnas que producían
los notebooks originales, por lo que son intercambiables sin tocar
``fetch_raw_data()`` ni ``prepare_features()``.
"""

from __future__ import annotations

import logging
import os
import time
import tomllib
from typing import Dict, List, Tuple

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
SECRETS_PATH = ".streamlit/secrets.toml"
DATA_DIR = "data"

# Endpoints
_EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Rango temporal del dataset estático maestro
LOAD_START = "2022-01-01"
LOAD_END = "2026-01-01"        # exclusivo: cubre hasta 2025-12-31T23
WEATHER_START = "2022-01-01"
WEATHER_END = "2025-12-31"

# Ciudades que concentran la carga energética de Texas (nombres capitalizados
# para reproducir el esquema de columnas del notebook original)
CITIES: Dict[str, Dict[str, float]] = {
    "Houston": {"lat": 29.7604, "lon": -95.3698},
    "Dallas":  {"lat": 32.7767, "lon": -96.7970},
    "Austin":  {"lat": 30.2672, "lon": -97.7431},
}

_WEATHER_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"
)

# Rutas de salida (idénticas a las que generaban los notebooks)
ERCOT_LOAD_CSV = os.path.join(DATA_DIR, "ercot_load_2022_2025_static.csv")
TEXAS_WEATHER_CSV = os.path.join(DATA_DIR, "texas_weather_2022_2025_static.csv")


def _read_eia_key() -> str:
    """Lee la clave EIA desde ``.streamlit/secrets.toml``.

    Returns
    -------
    str
        El valor de ``EIA_API_KEY``.

    Raises
    ------
    FileNotFoundError
        Si el archivo de secretos no existe.
    KeyError
        Si ``EIA_API_KEY`` no está definida en el archivo.
    """
    try:
        with open(SECRETS_PATH, "rb") as f:
            secrets = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró '{SECRETS_PATH}'. "
            "Asegúrate de que el archivo existe y contiene EIA_API_KEY."
        )
    key = secrets.get("EIA_API_KEY")
    if not key:
        raise KeyError("EIA_API_KEY no está definida en secrets.toml.")
    return key


def _build_time_blocks(start: str, end: str) -> List[Tuple[str, str]]:
    """Divide un rango de fechas en bloques consecutivos de 6 meses.

    La EIA limita cada respuesta a 5,000 registros; un bloque semestral
    (~4,344 horas) cabe holgadamente en una sola petición.

    Parameters
    ----------
    start:
        Fecha inicial inclusiva (``"YYYY-MM-DD"``).
    end:
        Fecha final exclusiva (``"YYYY-MM-DD"``).

    Returns
    -------
    List[Tuple[str, str]]
        Lista de pares ``(inicio, fin)`` con formato ``"YYYY-MM-DDTHH"``.
    """
    boundaries = pd.date_range(start=start, end=end, freq="6MS")
    return [
        (boundaries[i].strftime("%Y-%m-%dT%H"), boundaries[i + 1].strftime("%Y-%m-%dT%H"))
        for i in range(len(boundaries) - 1)
    ]


def seed_ercot_load(api_key: str, output_path: str = ERCOT_LOAD_CSV) -> pd.DataFrame:
    """Descarga la demanda horaria de ERCOT (2022-2025) y la persiste como CSV.

    Reproduce la lógica de ``EIA_API.ipynb``: itera por bloques semestrales,
    conserva el volcado crudo de la API (incluyendo los metadatos) y elimina
    duplicados en la marca temporal ``period``.

    Parameters
    ----------
    api_key:
        Clave de autenticación de la API de la EIA.
    output_path:
        Ruta donde se guardará el CSV resultante.

    Returns
    -------
    pd.DataFrame
        El histórico consolidado de demanda con el esquema crudo de la EIA.

    Raises
    ------
    RuntimeError
        Si ningún bloque temporal pudo descargarse con éxito.
    """
    blocks = _build_time_blocks(LOAD_START, LOAD_END)
    logger.info("EIA: descargando %d bloques semestrales (%s → %s)...",
                len(blocks), LOAD_START, LOAD_END)

    dataframes: List[pd.DataFrame] = []
    for inicio, fin in blocks:
        logger.info("EIA: bloque %s → %s", inicio, fin)
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "facets[type][]": "D",
            "start": inicio,
            "end": fin,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        try:
            response = requests.get(_EIA_URL, params=params, timeout=(5, 30))
            response.raise_for_status()
            records = response.json()["response"]["data"]
            dataframes.append(pd.DataFrame(records))
            time.sleep(1)  # Pausa de cortesía para no saturar la API
        except Exception as exc:
            logger.error("EIA: error en el bloque %s-%s: %s", inicio, fin, exc)

    if not dataframes:
        raise RuntimeError(
            "EIA: no se pudo descargar ningún bloque. Revisa la clave y la conexión."
        )

    df = (
        pd.concat(dataframes, ignore_index=True)
        .drop_duplicates(subset=["period"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("EIA: %d registros guardados en '%s'.", len(df), output_path)
    return df


def seed_texas_weather(output_path: str = TEXAS_WEATHER_CSV) -> pd.DataFrame:
    """Descarga el clima horario de Houston, Dallas y Austin y lo persiste como CSV.

    Reproduce la lógica de ``OPEN_METEO_API.ipynb``: consulta el archivo ERA5
    de Open-Meteo (sin clave), fusiona las tres ciudades sobre ``timestamp_utc``
    y añade la columna promedio ``texas_avg_temp``.

    Parameters
    ----------
    output_path:
        Ruta donde se guardará el CSV resultante.

    Returns
    -------
    pd.DataFrame
        La matriz climática consolidada de las tres ciudades.

    Raises
    ------
    RuntimeError
        Si no se pudieron recolectar los datos de las tres ciudades.
    """
    logger.info("Clima: descargando %s → %s para %d ciudades...",
                WEATHER_START, WEATHER_END, len(CITIES))

    dataframes: List[pd.DataFrame] = []
    for nombre, coords in CITIES.items():
        logger.info("Clima: extrayendo %s...", nombre)
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": WEATHER_START,
            "end_date": WEATHER_END,
            "hourly": _WEATHER_HOURLY_VARS,
            "timezone": "UTC",  # Alineación directa con el 'period' de la EIA
        }
        try:
            response = requests.get(_OPEN_METEO_ARCHIVE_URL, params=params, timeout=(5, 30))
            response.raise_for_status()
            hourly = response.json()["hourly"]

            df_ciudad = pd.DataFrame({
                "timestamp_utc": hourly["time"],
                f"{nombre}_temp": hourly["temperature_2m"],
                f"{nombre}_humidity": hourly["relative_humidity_2m"],
                f"{nombre}_apparent_temp": hourly["apparent_temperature"],
                f"{nombre}_wind_speed": hourly["wind_speed_10m"],
            })
            dataframes.append(df_ciudad)
            time.sleep(1)  # Pausa de cortesía
        except Exception as exc:
            logger.error("Clima: error al descargar %s: %s", nombre, exc)

    if len(dataframes) != len(CITIES):
        raise RuntimeError(
            "Clima: no se pudieron recolectar los datos de las tres ciudades."
        )

    # Fusión de las tres ciudades sobre la llave temporal
    df = dataframes[0]
    for df_ciudad in dataframes[1:]:
        df = pd.merge(df, df_ciudad, on="timestamp_utc")

    # Feature engineering preliminar: temperatura promedio del estado
    df["texas_avg_temp"] = df[["Houston_temp", "Dallas_temp", "Austin_temp"]].mean(axis=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Clima: %d registros guardados en '%s'.", len(df), output_path)
    return df


def main() -> None:
    """Genera ambos CSV estáticos de forma secuencial."""
    eia_api_key = _read_eia_key()

    logger.info("━━━ PASO 1/2: Demanda ERCOT (API de la EIA) ━━━")
    df_load = seed_ercot_load(eia_api_key)

    logger.info("━━━ PASO 2/2: Clima de Texas (Open-Meteo archive) ━━━")
    df_weather = seed_texas_weather()

    logger.info("━━━ SEED COMPLETADO ━━━")
    logger.info("  Demanda : %d filas → %s", len(df_load), ERCOT_LOAD_CSV)
    logger.info("  Clima   : %d filas → %s", len(df_weather), TEXAS_WEATHER_CSV)


if __name__ == "__main__":
    main()
