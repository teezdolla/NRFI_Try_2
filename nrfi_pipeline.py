import pandas as pd
import numpy as np
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from typing import Tuple, List


def combine_half_innings(df: pd.DataFrame) -> pd.DataFrame:
    """Combine consecutive half-innings into a single row with a total label."""
    # Ensure even number of rows
    if len(df) % 2 == 1:
        df = df.iloc[:-1]
    top = df.iloc[::2].reset_index(drop=True)
    bot = df.iloc[1::2].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in {"label", "pitcher"}]
    top_features = top[feature_cols].add_prefix("top_")
    bot_features = bot[feature_cols].add_prefix("bot_")

    combined = pd.concat([top_features, bot_features], axis=1)
    combined["label"] = ((top["label"] == 1) | (bot["label"] == 1)).astype(int)
    return combined


def load_data(path: str) -> pd.DataFrame:
    """Load training data."""
    return pd.read_csv(path)


def add_pitcher_form(df: pd.DataFrame, window: int = 5, short_window: int = 3) -> pd.DataFrame:
    """Add leak-free rolling pitcher stats for recent form."""
    if "game_date" not in df.columns:
        # Dataset already has aggregated features
        return df
    df = df.sort_values(["pitcher", "game_date"])

    rolling_cols = ["hits_allowed", "walks", "strikeouts", "batters_faced", "runs_allowed"]
    for col in rolling_cols:
        df[f"{col}_rolling{window}"] = (
            df.groupby("pitcher")[col]
            .shift()
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"{col}_rolling{short_window}"] = (
            df.groupby("pitcher")[col]
            .shift()
            .rolling(window=short_window, min_periods=1)
            .mean()
        )

    df["K_pct"] = (
        df.groupby("pitcher")
        .apply(
            lambda x: (
                x["strikeouts"].shift() / x["batters_faced"].shift()
            ).rolling(window, min_periods=1).mean()
        )
        .reset_index(level=0, drop=True)
    )
    df["BB_pct"] = (
        df.groupby("pitcher")
        .apply(
            lambda x: (
                x["walks"].shift() / x["batters_faced"].shift()
            ).rolling(window, min_periods=1).mean()
        )
        .reset_index(level=0, drop=True)
    )

    # Rolling xERA to capture recent effectiveness
    if "xERA_season" in df.columns:
        df["xERA_rolling"] = (
            df.groupby("pitcher")["xERA_season"].shift().rolling(window, min_periods=1).mean()
        )

    return df


def add_team_offense(df: pd.DataFrame, window: int = 10, short_window: int = 7) -> pd.DataFrame:
    """Add leak-free rolling team offense stats."""
    if "game_date" not in df.columns or "team" not in df.columns:
        return df
    df = df.sort_values(["team", "game_date"])

    rolling_cols = {
        "runs_1st": "runs_rolling",
        "hits": "hits_rolling",
        "walks": "walks_rolling",
        "total_bases": "tb_rolling",
        "pa": "pa_rolling",
        "abs": "ab_rolling",
        "strikeouts": "k_rolling",
    }

    for col, new in rolling_cols.items():
        df[new] = (
            df.groupby("team")[col]
            .shift()
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"{new}{short_window}"] = (
            df.groupby("team")[col]
            .shift()
            .rolling(window=short_window, min_periods=1)
            .mean()
        )

    df["OBP_team"] = (df["hits_rolling"] + df["walks_rolling"]) / df["pa_rolling"]
    df["SLG_team"] = df["tb_rolling"] / df["ab_rolling"]
    df["K_rate_team"] = df["k_rolling"] / df["pa_rolling"]
    df["BB_rate_team"] = df["walks_rolling"] / df["pa_rolling"]
    df["ISO_team"] = df["SLG_team"] - df["hits_rolling"] / df["ab_rolling"]
    df["OPS_team"] = df["OBP_team"] + df["SLG_team"]

    # Simple wOBA approximation using OBP and SLG
    df["wOBA_team"] = 0.5 * df["OBP_team"] + 0.5 * df["SLG_team"]

    return df

def merge_ballpark_factors(df: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    """Merge ballpark factors."""
    return df.merge(park_factors, on="park_id", how="left")


def target_encode_pitcher(df: pd.DataFrame, target_col: str = "label"):
    """Add a target-encoded pitcher column and return the mapping and global mean."""
    global_mean = df[target_col].mean()
    mapping = df.groupby("pitcher")[target_col].mean().to_dict()
    df["pitcher_te"] = df["pitcher"].map(mapping).fillna(global_mean)
    return df, mapping, global_mean


def time_series_cv(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Tuple[List[float], List[float]]:
    """Perform time-series cross validation for logistic and XGBoost models."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    lr_metrics: List[List[float]] = []
    xgb_metrics: List[List[float]] = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Logistic regression with isotonic calibration
        lr_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
        calibrated = CalibratedClassifierCV(lr_model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)

        lr_preds = calibrated.predict(X_test)
        lr_proba = calibrated.predict_proba(X_test)[:, 1]
        lr_metrics.append(
            [
                accuracy_score(y_test, lr_preds),
                roc_auc_score(y_test, lr_proba),
                f1_score(y_test, lr_preds),
                precision_score(y_test, lr_preds),
                recall_score(y_test, lr_preds),
                log_loss(y_test, lr_proba),
            ]
        )

        # Tuned XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        cv_res = xgb.cv(
            params,
            dtrain,
            num_boost_round=500,
            nfold=3,
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        best_round = len(cv_res)
        booster = xgb.train(params, dtrain, num_boost_round=best_round)
        proba = booster.predict(dtest)
        preds = (proba >= 0.5).astype(int)
        xgb_metrics.append(
            [
                accuracy_score(y_test, preds),
                roc_auc_score(y_test, proba),
                f1_score(y_test, preds),
                precision_score(y_test, preds),
                recall_score(y_test, preds),
                log_loss(y_test, proba),
            ]
        )

    return np.mean(lr_metrics, axis=0).tolist(), np.mean(xgb_metrics, axis=0).tolist()


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.Booster:
    """Train a tuned XGBoost model using cross-validation."""
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    cv_res = xgb.cv(
        params,
        dtrain,
        num_boost_round=500,
        nfold=5,
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    best_round = len(cv_res)
    model = xgb.train(params, dtrain, num_boost_round=best_round)
    return model


def get_weather_features(game_date: str, latitude: float, longitude: float) -> dict:
    """Fetch weather data from Open-Meteo API."""
    import requests

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,"
        "relative_humidity_2m,pressure_msl,windspeed_10m,winddirection_10m"
        f"&start_date={game_date}&end_date={game_date}"
    )
    resp = requests.get(url, timeout=10)
    if resp.ok:
        data = resp.json()["hourly"]
        return {
            "temperature": np.mean(data.get("temperature_2m", [np.nan])),
            "humidity": np.mean(data.get("relative_humidity_2m", [np.nan])),
            "pressure": np.mean(data.get("pressure_msl", [np.nan])),
            "wind_speed": np.mean(data.get("windspeed_10m", [np.nan])),
            "wind_dir": np.mean(data.get("winddirection_10m", [np.nan])),
        }
    return {
        "temperature": np.nan,
        "humidity": np.nan,
        "pressure": np.nan,
        "wind_speed": np.nan,
        "wind_dir": np.nan,
    }


def predict_today(model: xgb.Booster, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    dtest = xgb.DMatrix(X)
    proba = model.predict(dtest)
    predictions = (proba >= threshold).astype(int)
    return pd.DataFrame({
        "P_YRFI": proba,
        "P_NRFI": 1 - proba,
        "Prediction": np.where(predictions == 1, "YRFI", "NRFI"),
        "Confidence": np.abs(proba - 0.5) * 2,
        "Threshold": threshold,
    })


def main():
    df = load_data("final_training_data_clean_final.csv")
    df = add_pitcher_form(df)
    df = add_team_offense(df)
    df, mapping, global_mean = target_encode_pitcher(df)
    with open("pitcher_encoding.json", "w") as f:
        json.dump({"mapping": mapping, "global_mean": global_mean}, f)

    feature_cols = [
        col
        for col in df.columns
        if col not in {"label", "game_date", "pitcher"}
    ]
    X = df[feature_cols].fillna(0)
    y = df["label"]

    lr_metrics, xgb_metrics = time_series_cv(X, y, n_splits=5)
    print(
        "LogReg CV -> Acc: {:.4f}, AUC: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, LogLoss: {:.4f}".format(
            *lr_metrics
        )
    )
    print(
        "XGB CV -> Acc: {:.4f}, AUC: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, LogLoss: {:.4f}".format(
            *xgb_metrics
        )
    )

    model = train_xgboost(X, y)
    model.save_model("xgboost_nrfi_model.json")

    # Train a model on full inning data
    full_df = combine_half_innings(df)
    full_feature_cols = [c for c in full_df.columns if c != "label"]
    X_full = full_df[full_feature_cols].fillna(0)
    y_full = full_df["label"]

    lr_full, xgb_full = time_series_cv(X_full, y_full, n_splits=5)
    print(
        "Full Inning LogReg CV -> Acc: {:.4f}, AUC: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, LogLoss: {:.4f}".format(
            *lr_full
        )
    )
    print(
        "Full Inning XGB CV -> Acc: {:.4f}, AUC: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, LogLoss: {:.4f}".format(
            *xgb_full
        )
    )

    full_model = train_xgboost(X_full, y_full)
    full_model.save_model("xgboost_yrfi_full_inning.json")


if __name__ == "__main__":
    main()
