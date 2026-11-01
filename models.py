from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor


def _get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y from aggregated driver df."""
    target_col = "LapTime_min" if "LapTime_min" in df.columns else "LapTime"
    if target_col not in df.columns:
        raise ValueError("No LapTime or LapTime_min column found.")

    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        raise ValueError(f"Target {target_col} not in numeric columns.")
    y = numeric[target_col]
    X = numeric.drop(columns=[target_col])

    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")
    # Fill remaining NaN with column median (XGBoost handles NaN but this helps)
    X = X.fillna(X.median())
    return X, y


def train_qualifying_model(df: pd.DataFrame) -> Tuple[XGBRegressor, float, np.ndarray, np.ndarray]:
    """
    Train XGBRegressor on practice features to predict lap time.
    Returns (model, MAE, y_actual, y_pred) from cross-validation for accuracy analysis.
    """
    X, y = _get_X_y(df)
    n = len(X)
    if n < 3:
        raise ValueError("Need at least 3 drivers for training.")

    xgb_reg = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    # Cross-validated predictions for MAE and scatter plot (avoids overfitting)
    cv_folds = min(5, n - 1)
    y_pred_cv = cross_val_predict(xgb_reg, X, y, cv=cv_folds)
    mae = mean_absolute_error(y, y_pred_cv)

    # Train final model on full data for predictions
    xgb_reg.fit(X, y)

    return xgb_reg, mae, y.values, y_pred_cv


def predict_qualifying(model: XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict qualifying lap times using the trained XGBoost model.
    """
    X, _ = _get_X_y(df)
    preds = model.predict(X)

    result = df.copy()
    result["PredictedLapTime"] = preds
    result = result.sort_values("PredictedLapTime").reset_index(drop=True)
    return result


def train_race_model(df: pd.DataFrame) -> XGBRegressor:
    """
    Train XGBRegressor (same as qualifying); race order = ranked by predicted lap time.
    """
    model, _, _, _ = train_qualifying_model(df)
    return model


def predict_race(model: XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict race order from XGBoost-predicted lap times.
    """
    result = predict_qualifying(model, df)
    result["PredictedPosition"] = range(1, len(result) + 1)
    return result


def compute_race_accuracy(
    predicted: pd.DataFrame, actual: pd.DataFrame, driver_map: dict = None
) -> Tuple[float, float, pd.DataFrame]:
    """
    Compare predicted race order with actual race results.
    predicted has Driver (full name or abbrev) and PredictedPosition.
    actual has Driver (abbrev) and Position.
    driver_map: abbrev -> full name, used to match pred (full) to actual (abbrev).

    Returns
    -------
    Tuple[float, float, pd.DataFrame]
        (MAE of positions, Spearman correlation, merged comparison df)
    """
    from scipy.stats import spearmanr

    pred = predicted[["Driver", "PredictedPosition"]].copy()
    act = actual[["Driver", "Position"]].copy()

    # Map pred Driver (full name) -> abbrev for merge with actual
    full_to_abbrev = {v: k for k, v in (driver_map or {}).items()}
    pred["DriverAbbrev"] = pred["Driver"].map(lambda x: full_to_abbrev.get(x, x))
    act = act.rename(columns={"Driver": "DriverAbbrev"})

    merged = pred.merge(act, on="DriverAbbrev", how="inner")
    if len(merged) < 2:
        return float("nan"), float("nan"), merged

    mae = (merged["PredictedPosition"] - merged["Position"]).abs().mean()
    corr, _ = spearmanr(merged["PredictedPosition"], merged["Position"])
    return float(mae), float(corr) if not np.isnan(corr) else 0.0, merged
