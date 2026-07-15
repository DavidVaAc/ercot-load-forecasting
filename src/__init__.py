"""
ERCOT Load Forecasting — Production Package
============================================

Exports the public API of the src package so callers can do:

    from src import fetch_raw_data, prepare_features, train_lightgbm

instead of drilling into individual sub-modules.
"""

from .extraction import fetch_historical_data, fetch_live_data, fetch_raw_data
from .processing import COLUMNS_ORDER, build_live_features, prepare_features
from .training import PRODUCTION_LGBM_PARAMS, train_lightgbm

__all__ = [
    # extraction
    "fetch_raw_data",
    "fetch_historical_data",
    "fetch_live_data",
    # processing
    "prepare_features",
    "build_live_features",
    "COLUMNS_ORDER",
    # training
    "train_lightgbm",
    "PRODUCTION_LGBM_PARAMS",
]
