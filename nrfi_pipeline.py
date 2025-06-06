import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from typing import Tuple


def load_data(path: str) -> pd.DataFrame:
    """Load training data."""
    return pd.read_csv(path)


def add_pitcher_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add leak-free rolling pitcher stats."""
    df = df.sort_values(["pitcher", "game_date"])
    rolling_cols = ["hits_allowed", "walks", "strikeouts", "batters_faced", "runs_allowed"]
    for col in rolling_cols:
        df[f"{col}_rolling{window}"] = (
            df.groupby("pitcher")[col]
            .shift()  # exclude current game
            .rolling(window=window, min_periods=1)
            .mean()
        )
    df["K_pct"] = (
        df.groupby("pitcher")
        .apply(lambda x: (x["strikeouts"].shift() / x["batters_faced"].shift()).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    df["BB_pct"] = (
        df.groupby("pitcher")
        .apply(lambda x: (x["walks"].shift() / x["batters_faced"].shift()).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_team_offense(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add leak-free rolling team offense stats."""
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
    df["OBP_team"] = (df["hits_rolling"] + df["walks_rolling"]) / df["pa_rolling"]
    df["SLG_team"] = df["tb_rolling"] / df["ab_rolling"]
    df["K_rate_team"] = df["k_rolling"] / df["pa_rolling"]
    df["BB_rate_team"] = df["walks_rolling"] / df["pa_rolling"]
    df["ISO_team"] = (df["SLG_team"] - df["hits_rolling"] / df["ab_rolling"])
    df["OPS_team"] = df["OBP_team"] + df["SLG_team"]

    return df

def merge_ballpark_factors(df: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    """Merge ballpark factors."""
    return df.merge(park_factors, on="park_id", how="left")


def time_series_cv(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Tuple[float, float, float, float, float]:
    """Perform time-series cross validation and return averaged metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        )
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)

        preds = calibrated.predict(X_test)
        proba = calibrated.predict_proba(X_test)[:, 1]
        metrics.append(
            [
                accuracy_score(y_test, preds),
                roc_auc_score(y_test, proba),
                f1_score(y_test, preds),
                precision_score(y_test, preds),
                recall_score(y_test, preds),
                log_loss(y_test, proba),
            ]
        )

    metrics = np.array(metrics)
    return metrics.mean(axis=0)


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=300)
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
    })


def main():
    df = load_data("final_training_data_clean_final.csv")
    df = add_pitcher_form(df)
    df = add_team_offense(df)

    feature_cols = [
        col for col in df.columns if col not in {"label", "game_date"}
    ]
    X = df[feature_cols].fillna(0)
    y = df["label"]

    metrics = time_series_cv(X, y, n_splits=5)
    print(
        "CV Metrics -> Accuracy: {:.4f}, AUC: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, LogLoss: {:.4f}".format(
            *metrics
        )
    )

    model = train_xgboost(X, y)
    model.save_model("xgboost_nrfi_model.json")


if __name__ == "__main__":
    main()
