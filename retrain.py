"""
retrain.py — Punto de entrada para el reentrenamiento periódico del modelo.

Uso:
    python retrain.py

Pipeline completo en 6 pasos:
    1. Carga los CSV estáticos (2022-2025)  ← fetch_raw_data()
    2. Descarga datos recientes desde APIs  ← fetch_historical_data()
    3. Fusiona y deduplica ambas fuentes
    4. Ingeniería de características        ← prepare_features()
    5. Evalúa en el benchmark 2025          ← LGBMRegressor (solo métricas)
    6. Reentrena en el 100% (2022-hoy)      ← train_lightgbm() y guarda
"""

import logging
import os
import tomllib

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import metrics as sk_metrics

from src.extraction import fetch_historical_data, fetch_raw_data
from src.processing import prepare_features
from src.training import PRODUCTION_LGBM_PARAMS, train_lightgbm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

TARGET_COL = "value"
EVAL_CUTOFF = "2025-01-01"       # Límite fijo para el benchmark de evaluación
MODEL_PATH = "models/modelo_final_ercot_lgb_retrained.json"
BASE_MODEL_PATH = "models/modelo_final_ercot_lgb.json"
SECRETS_PATH = ".streamlit/secrets.toml"

# Columnas a conservar del dataset estático antes de concatenar
_BASE_COLS = [
    "period", "value",
    "houston_temp", "houston_humidity", "houston_apparent_temp", "houston_wind_speed",
    "dallas_temp",  "dallas_humidity",  "dallas_apparent_temp",  "dallas_wind_speed",
    "austin_temp",  "austin_humidity",  "austin_apparent_temp",  "austin_wind_speed",
    "texas_avg_temp",
]


def _read_eia_key() -> str:
    """Lee la clave EIA desde .streamlit/secrets.toml."""
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


def main() -> None:
    eia_api_key = _read_eia_key()

    # ─────────────────────────────────────────────────────────────────
    # PASO 1: Datos históricos estáticos (2022-2025)
    # ─────────────────────────────────────────────────────────────────
    logging.info("━━━ PASO 1/6: Cargando CSV estáticos (2022-2025) ━━━")
    df_static = fetch_raw_data()
    df_static.columns = df_static.columns.str.replace("-", "_").str.lower()
    df_static = df_static[[c for c in _BASE_COLS if c in df_static.columns]].copy()
    df_static["period"] = pd.to_datetime(df_static["period"])
    last_static = df_static["period"].max()
    logging.info(
        "CSV estáticos: %d filas  |  rango: %s → %s",
        len(df_static),
        df_static["period"].min().date(),
        last_static.date(),
    )

    # ─────────────────────────────────────────────────────────────────
    # PASO 2: Datos recientes desde APIs (último día del CSV+1 → hoy-5d)
    # ─────────────────────────────────────────────────────────────────
    start_recent = (last_static + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    # Open-Meteo archive tiene un retraso de ~5 días; usamos ese margen
    end_recent = (pd.Timestamp.utcnow() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    logging.info(
        "━━━ PASO 2/6: Descargando datos recientes de APIs (%s → %s) ━━━",
        start_recent, end_recent,
    )
    df_recent = fetch_historical_data(start_recent, end_recent, eia_api_key)
    logging.info("Datos recientes: %d filas", len(df_recent))

    # ─────────────────────────────────────────────────────────────────
    # PASO 3: Fusión y deduplicación
    # ─────────────────────────────────────────────────────────────────
    logging.info("━━━ PASO 3/6: Unificando datasets ━━━")
    df_full = (
        pd.concat([df_static, df_recent], ignore_index=True)
        .drop_duplicates(subset="period")
        .sort_values("period")
        .reset_index(drop=True)
    )
    logging.info(
        "Dataset unificado: %d filas  |  rango: %s → %s",
        len(df_full),
        df_full["period"].min().date(),
        df_full["period"].max().date(),
    )

    # ─────────────────────────────────────────────────────────────────
    # PASO 4: Ingeniería de características
    # ─────────────────────────────────────────────────────────────────
    logging.info("━━━ PASO 4/6: Ingeniería de características ━━━")
    df = prepare_features(df_full)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X, y = df[feature_cols], df[TARGET_COL]
    logging.info("Dataset final: %d filas × %d features", *X.shape)

    # ─────────────────────────────────────────────────────────────────
    # PASO 5: Evaluación sobre el benchmark fijo 2025
    # (entrena en 2022-2024, evalúa en 2025 — igual que en el notebook)
    # ─────────────────────────────────────────────────────────────────
    logging.info("━━━ PASO 5/6: Evaluación en benchmark 2025 ━━━")
    eval_train = (X.index < EVAL_CUTOFF)
    eval_test  = (X.index >= EVAL_CUTOFF) & (X.index < "2026-01-01")

    eval_model = LGBMRegressor(**PRODUCTION_LGBM_PARAMS)
    eval_model.fit(X[eval_train], y[eval_train])
    y_pred_2025 = eval_model.predict(X[eval_test])

    mape = sk_metrics.mean_absolute_percentage_error(y[eval_test], y_pred_2025) * 100
    mae  = sk_metrics.mean_absolute_error(y[eval_test], y_pred_2025)
    rmse = float(np.sqrt(sk_metrics.mean_squared_error(y[eval_test], y_pred_2025)))
    logging.info(
        "Benchmark 2025 → MAPE: %.2f%%  MAE: %.0f MW  RMSE: %.0f MW",
        mape, mae, rmse,
    )

    # 💡 MODO MANTENIMIENTO: Descomenta las siguientes líneas si deseas regenerar 
    # estrictamente el modelo base de R&D (2022-2025).
    # ─────────────────────────────────────────────────────────────────────────────────
    # logging.info("⚠️ [MANTENIMIENTO] Regenerando estrictamente el MODELO BASE de R&D (2022-2025)...")
    #eval_model.fit(X, y)
    #if BASE_MODEL_PATH is not None:
    #    os.makedirs(os.path.dirname(BASE_MODEL_PATH) or ".", exist_ok=True)
    #    eval_model.booster_.save_model(BASE_MODEL_PATH)
    #    logging.info(
    #        "train_lightgbm: production model saved to '%s'.", BASE_MODEL_PATH
    #    )
    # ─────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # PASO 6: Reentrenamiento en 100% de datos (2022-hoy) y guardado
    # ─────────────────────────────────────────────────────────────────
    logging.info(
        "━━━ PASO 6/6: Reentrenando modelo de producción en %d filas (%s → %s) ━━━",
        len(X),
        X.index.min().date(),
        X.index.max().date(),
    )
    # Se llama sin X_test para que train_lightgbm entrene directamente
    # sobre el dataset completo (Fase 2 con X_full = X)
    train_lightgbm(X, y, model_output_path=MODEL_PATH)

    logging.info("━━━ RESUMEN FINAL ━━━")
    logging.info("  Benchmark 2025 MAPE : %.2f %%", mape)
    logging.info("  Benchmark 2025 MAE  : %.0f MW", mae)
    logging.info("  Benchmark 2025 RMSE : %.0f MW", rmse)
    logging.info("  Filas en producción : %d  (%s → %s)", len(X), X.index.min().date(), X.index.max().date())
    logging.info("  Modelo guardado     : %s", MODEL_PATH)


if __name__ == "__main__":
    main()
