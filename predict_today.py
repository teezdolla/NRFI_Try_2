import pandas as pd
import numpy as np
import json
import requests
import statsapi
from pybaseball import playerid_lookup, playerid_reverse_lookup, pitching_stats
import xgboost as xgb
from datetime import datetime
from functools import lru_cache

# Mapping from 2-3 letter abbreviations to MLBAM team ids
TEAM_ID_MAP = {
    'ATH': 133, 'PIT': 134, 'SD': 135, 'SEA': 136, 'SF': 137, 'STL': 138,
    'TB': 139, 'TEX': 140, 'TOR': 141, 'MIN': 142, 'PHI': 143, 'ATL': 144,
    'CWS': 145, 'MIA': 146, 'NYY': 147, 'MIL': 158, 'LAA': 108, 'AZ': 109,
    'BAL': 110, 'BOS': 111, 'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115,
    'DET': 116, 'HOU': 117, 'KC': 118, 'LAD': 119, 'WSH': 120, 'NYM': 121,
}


def get_pitcher_id(name: str) -> int | None:
    """Return MLBAM id for a pitcher name using pybaseball lookup."""
    if not name:
        return None
    parts = name.split()
    if len(parts) < 2:
        return None
    try:
        lookup = playerid_lookup(parts[-1], " ".join(parts[:-1]))
        if not lookup.empty:
            return int(lookup.key_mlbam.iloc[0])
    except Exception:
        pass
    return None


@lru_cache(maxsize=None)
def _fg_table(season: int) -> pd.DataFrame:
    """Cache Fangraphs pitching stats for a season."""
    return pitching_stats(season, season, qual=0)


@lru_cache(maxsize=None)
def _fg_id(mlbam: int) -> int | None:
    """Map MLBAM id to Fangraphs id."""
    try:
        return int(playerid_reverse_lookup([mlbam]).key_fangraphs.iloc[0])
    except Exception:
        return None


@lru_cache(maxsize=None)
def _team_id(name: str) -> int:
    if name in TEAM_ID_MAP:
        return TEAM_ID_MAP[name]
    lookup = statsapi.lookup_team(name)
    if not lookup:
        raise ValueError(f"Unknown team {name}")
    return lookup[0]["id"]


def pitcher_form_features(pitcher: int, season: int, n: int = 5) -> dict:
    """Rolling averages for recent pitcher games."""
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    resp = requests.get(url, params=params, timeout=10)
    splits = resp.json().get("stats", [{}])[0].get("splits", [])
    if not splits:
        return {k: np.nan for k in ["hits_allowed", "walks", "strikeouts", "batters_faced", "runs_allowed"]}
    df = pd.json_normalize(splits)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(n)
    return {
        "hits_allowed": df["stat.hits"].mean(),
        "walks": df["stat.baseOnBalls"].mean(),
        "strikeouts": df["stat.strikeOuts"].mean(),
        "batters_faced": df["stat.battersFaced"].mean(),
        "runs_allowed": df["stat.runs"].mean(),
    }


def pitcher_season_features(pitcher: int, season: int) -> dict:
    """Season-long pitching stats including Fangraphs metrics."""
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher}/stats"
    params = {"stats": "season", "group": "pitching", "season": season}
    resp = requests.get(url, params=params, timeout=10)
    split = resp.json().get("stats", [{}])[0].get("splits", [])
    stat = split[0]["stat"] if split else {}
    feats = {
        "ERA_season": float(stat.get("era", np.nan)),
        "WHIP_season": float(stat.get("whip", np.nan)),
        "K/9_season": float(stat.get("strikeoutsPer9Inn", np.nan)),
        "BB/9_season": float(stat.get("walksPer9Inn", np.nan)),
    }
    fgid = _fg_id(pitcher)
    table = _fg_table(season)
    if fgid and fgid in table["IDfg"].values:
        row = table.loc[table["IDfg"] == fgid].iloc[0]
        feats.update(
            {
                "FIP_season": float(row.get("FIP", np.nan)),
                "xFIP_season": float(row.get("xFIP", np.nan)),
                "CSW%_season": float(row.get("CSW%", np.nan)),
                "xERA_season": float(row.get("xERA", np.nan)),
            }
        )
    else:
        feats.update(
            {
                "FIP_season": np.nan,
                "xFIP_season": np.nan,
                "CSW%_season": np.nan,
                "xERA_season": np.nan,
            }
        )
    return feats


def team_offense_features(team: str, season: int, date_str: str) -> dict:
    """Approximate team offense metrics using recent games and season stats."""
    tid = _team_id(team)
    schedule = statsapi.schedule(team=tid, end_date=date_str)
    schedule = [g for g in schedule if g.get("status") == "Final"][-10:]
    runs = []
    for g in schedule:
        gid = g["game_id"]
        line = requests.get(f"https://statsapi.mlb.com/api/v1/game/{gid}/linescore", timeout=10).json()
        half = "home" if g["home_id"] == tid else "away"
        try:
            runs.append(line["innings"][0][half]["runs"])
        except Exception:
            pass
    run_avg = np.mean(runs) if runs else np.nan

    ts = statsapi.get(
        "team_stats",
        {"teamId": tid, "season": season, "stats": "season", "group": "hitting"},
    )
    stat = ts["stats"][0]["splits"][0]["stat"]
    pa = float(stat.get("plateAppearances", 1))
    return {
        "runs_rolling10_team": run_avg,
        "OBP_team": float(stat.get("obp", np.nan)),
        "SLG_team": float(stat.get("slg", np.nan)),
        "K_rate_team": float(stat.get("strikeOuts", 0)) / pa,
        "BB_rate_team": float(stat.get("baseOnBalls", 0)) / pa,
    }


def fetch_games(date_str: str) -> pd.DataFrame:
    """Fetch today's games with both probable pitchers."""
    schedule = statsapi.schedule(date=date_str)
    rows = []
    for g in schedule:
        home_pitcher = g.get("home_probable_pitcher")
        away_pitcher = g.get("away_probable_pitcher")
        if not home_pitcher or not away_pitcher:
            continue
        game_id = g.get("game_id")
        home = g["home_name"]
        away = g["away_name"]
        rows.append(
            {
                "game_id": game_id,
                "team": away,
                "pitcher_name": home_pitcher,
                "inning_topbot": "Top",
                "home_name": home,
                "away_name": away,
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "team": home,
                "pitcher_name": away_pitcher,
                "inning_topbot": "Bot",
                "home_name": home,
                "away_name": away,
            }
        )
    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame, date_str: str):
    with open("pitcher_encoding.json") as f:
        enc = json.load(f)
    mapping = {int(k): v for k, v in enc["mapping"].items()}
    global_mean = enc["global_mean"]
    team_map = {
        'Arizona Diamondbacks': 'AZ', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
        'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
        'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
        'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
        'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
        'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
        'New York Yankees': 'NYY', 'Oakland Athletics': 'ATH', 'Philadelphia Phillies': 'PHI',
        'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
        'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
        'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH'
    }
    df['team_abbr'] = df['team'].map(team_map)
    df['half_inning'] = df['inning_topbot'] + '_1st'
    df['pitcher'] = df['pitcher_name'].apply(get_pitcher_id)
    df = df.dropna(subset=['pitcher']).copy()
    df['pitcher_te'] = df['pitcher'].map(mapping).fillna(global_mean)
    df['season'] = pd.to_datetime(date_str).year
    df['inning'] = 1
    df['is_home_team'] = (df['inning_topbot'] == 'Bot').astype(int)

    feature_cols = [
        'inning','season','hits_allowed','walks','strikeouts',
        'batters_faced','runs_allowed','ERA_season','WHIP_season','FIP_season',
        'K/9_season','BB/9_season','xFIP_season','CSW%_season','xERA_season',
        'runs_rolling10_team','OBP_team','SLG_team','K_rate_team','BB_rate_team',
        'is_home_team','pitcher_te'
    ]

    X = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)

    for idx, row in df.iterrows():
        p_form = pitcher_form_features(row['pitcher'], row['season'])
        p_season = pitcher_season_features(row['pitcher'], row['season'])
        t_off = team_offense_features(row['team_abbr'], row['season'], date_str)
        feats = {**p_form, **p_season, **t_off}

        X.loc[idx, 'inning'] = row['inning']
        X.loc[idx, 'pitcher_te'] = row['pitcher_te']
        X.loc[idx, 'season'] = row['season']
        X.loc[idx, 'is_home_team'] = row['is_home_team']
        for col, val in feats.items():
            X.loc[idx, col] = val

    return df, X


def main():
    today = datetime.today().strftime('%Y-%m-%d')
    games = fetch_games(today)
    if games.empty:
        print('No games with both pitchers found.')
        return
    games, X = build_features(games, today)
    model = xgb.Booster()
    model.load_model('xgboost_nrfi_model.json')
    dmat = xgb.DMatrix(X)
    games['P_YRFI'] = model.predict(dmat)
    games['P_NRFI'] = 1 - games['P_YRFI']

    # Prepare features for the full inning model (top followed by bottom)
    top_feats = X.iloc[::2].reset_index(drop=True).add_prefix('top_')
    bot_feats = X.iloc[1::2].reset_index(drop=True).add_prefix('bot_')
    full_X = pd.concat([top_feats, bot_feats], axis=1)

    full_model = xgb.Booster()
    full_model.load_model('xgboost_yrfi_full_inning.json')
    full_probs = full_model.predict(xgb.DMatrix(full_X))

    # Pivot to get separate predictions for each half inning
    pivot = games.pivot_table(
        index=["game_id", "away_name", "home_name"],
        columns="inning_topbot",
        values=["P_YRFI", "P_NRFI"],
    ).reset_index()

    # Flatten the MultiIndex columns produced by the pivot
    pivot.columns = ["_".join(col).strip("_") for col in pivot.columns.to_flat_index()]

    # Totals predicted by the full inning model
    pivot["P_YRFI_total"] = full_probs
    pivot["P_NRFI_total"] = 1 - full_probs

    pivot["matchup"] = pivot["away_name"] + " @ " + pivot["home_name"]

    result = pivot[
        [
            "matchup",
            "P_YRFI_Top",
            "P_NRFI_Top",
            "P_YRFI_Bot",
            "P_NRFI_Bot",
            "P_YRFI_total",
            "P_NRFI_total",
        ]
    ].sort_values("P_YRFI_total", ascending=False)

    # Write results with comma separated values and fixed precision
    result.to_csv(
        "predictions.txt",
        index=False,
        sep=",",
        float_format="%.3f",
    )

    # Display with the same formatting so console output matches the file
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(result.to_string(index=False))

if __name__ == '__main__':
    main()
