"""
LightGBM training pipeline for the ERCOT load-forecasting project.
===================================================================

Provides the single entry point for model training:

train_lightgbm
    Trains a LightGBM regressor using the locked production hyperparameters,
    optionally runs TimeSeriesSplit cross-validation for stability diagnostics,
    optionally runs RandomizedSearchCV to discover improved hyperparameters,
    evaluates on a held-out test set, and serialises the trained booster.

PRODUCTION_LGBM_PARAMS
    The optimal hyperparameter dictionary locked after the R&D phase.
    Do NOT modify without rerunning the full optimisation in the notebook.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters — locked after RandomizedSearchCV in ercot_exploration.ipynb
# WARNING: Do NOT change these values.  Any modification alters the model's
#          predictive behaviour and invalidates the reported MAPE of 3.11 %.
# ---------------------------------------------------------------------------
PRODUCTION_LGBM_PARAMS: Dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Hyperparameter search space (kept for reference and optional re-optimisation)
_HYPERPARAM_SEARCH_SPACE: Dict = {
    "n_estimators": [300, 500, 700],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [4, 6, 8, -1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    n_cv_splits: int = 3,
    n_search_iter: int = 15,
    run_hyperparameter_search: bool = False,
    model_output_path: Optional[str] = "models/modelo_final_ercot_lgb.json",
) -> Tuple[LGBMRegressor, Dict]:
    """Train a LightGBM regressor and save a production model on the full dataset.

    Mirrors the two-phase workflow from the R&D notebook exactly:

    **Phase 1 — Evaluation (optional)**
    If ``X_test`` and ``y_test`` are provided, a temporary model is first
    fitted on ``X_train`` / ``y_train`` only and evaluated on the held-out
    test period.  This yields the honest out-of-sample metrics (MAPE, MAE,
    RMSE) without any future leakage.

    **Phase 2 — Production retraining (always)**
    After evaluation, the final model is retrained on the **full dataset**
    (``X_train + X_test`` when a test set is provided, otherwise ``X_train``
    alone).  Only this production model is serialised and returned.  This
    ensures the deployed artefact has seen every available data point before
    going live.

    The temporal validation always respects chronological order:
    ``TimeSeriesSplit`` produces expanding training windows so each fold
    trains on all past data and validates on the immediately following block.

    Parameters
    ----------
    X_train:
        Feature matrix for the training period (chronologically before the
        test cutoff — typically 2022–2024 in the standard split).
    y_train:
        Hourly demand target in MW for the training period.
    X_test:
        Feature matrix for the held-out test period (e.g., 2025).
        When provided, out-of-sample metrics are computed **before** the
        production retraining step.
    y_test:
        Hourly demand target in MW for the test period.
    n_cv_splits:
        Number of ``TimeSeriesSplit`` folds used for cross-validation
        diagnostics (default 3, matching the R&D notebook).
    n_search_iter:
        Number of random hyperparameter combinations sampled by
        ``RandomizedSearchCV`` (only used when
        ``run_hyperparameter_search=True``).
    run_hyperparameter_search:
        When ``True``, override :data:`PRODUCTION_LGBM_PARAMS` with the best
        configuration found by ``RandomizedSearchCV``.  Computationally
        expensive — disable for routine retraining.
    model_output_path:
        Path where the production booster is saved as a LightGBM JSON file
        compatible with ``lgb.Booster(model_file=...)`` in the Streamlit
        dashboard.  Parent directories are created automatically.
        Pass ``None`` to skip serialisation.

    Returns
    -------
    Tuple[LGBMRegressor, Dict]
        A two-element tuple ``(model, metrics_dict)`` where:

        * ``model``        — Production ``LGBMRegressor`` fitted on the
          **full** dataset (train + test).
        * ``metrics_dict`` — Dictionary with keys ``mape`` (%), ``mae`` (MW)
          and ``rmse`` (MW) computed on the held-out ``X_test / y_test``.
          Empty dict when no test set is provided.
    """
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    # ------------------------------------------------ Hyperparameter search --
    if run_hyperparameter_search:
        logger.info(
            "train_lightgbm: running RandomizedSearchCV (%d iterations × %d folds)...",
            n_search_iter,
            n_cv_splits,
        )
        search = RandomizedSearchCV(
            estimator=LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            param_distributions=_HYPERPARAM_SEARCH_SPACE,
            n_iter=n_search_iter,
            cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        fit_params = search.best_params_
        best_cv_mape = abs(search.best_score_) * 100
        logger.info(
            "train_lightgbm: best CV MAPE = %.2f%% | params = %s",
            best_cv_mape,
            fit_params,
        )
    else:
        fit_params = PRODUCTION_LGBM_PARAMS

    # --------------------------------- Phase 1: evaluation on held-out test --
    eval_metrics: Dict = {}
    if X_test is not None and y_test is not None:
        logger.info(
            "train_lightgbm: [Phase 1] evaluating on held-out test set "
            "(%d samples)...",
            len(X_test),
        )
        eval_model = LGBMRegressor(**fit_params)
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        eval_metrics["mae"] = float(metrics.mean_absolute_error(y_test, y_pred))
        eval_metrics["rmse"] = float(
            np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        )
        eval_metrics["mape"] = float(
            metrics.mean_absolute_percentage_error(y_test, y_pred) * 100
        )
        logger.info(
            "train_lightgbm: [Phase 1] MAPE=%.2f%% | MAE=%.0f MW | RMSE=%.0f MW",
            eval_metrics["mape"],
            eval_metrics["mae"],
            eval_metrics["rmse"],
        )

    # --------------- Phase 2: production retraining on 100 % of the data --
    if X_test is not None and y_test is not None:
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
    else:
        X_full, y_full = X_train, y_train

    logger.info(
        "train_lightgbm: [Phase 2] retraining production model on full dataset "
        "(%d samples × %d features)...",
        X_full.shape[0],
        X_full.shape[1],
    )
    model = LGBMRegressor(**fit_params)
    model.fit(X_full, y_full)
    logger.info("train_lightgbm: [Phase 2] production training complete.")

    # ---------------------------------------------------- Serialisation --
    if model_output_path is not None:
        os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
        model.booster_.save_model(model_output_path)
        logger.info(
            "train_lightgbm: production model saved to '%s'.", model_output_path
        )

    return model, eval_metrics
