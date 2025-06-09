import pandas as pd
import statsapi
from pybaseball import playerid_lookup
import xgboost as xgb
from datetime import datetime


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
    df['season'] = pd.to_datetime(date_str).year
    df['inning'] = 1
    df['is_home_team'] = (df['inning_topbot'] == 'Bot').astype(int)

    feature_cols = ['inning','pitcher','season','hits_allowed','walks','strikeouts',
                    'batters_faced','runs_allowed','ERA_season','WHIP_season','FIP_season',
                    'K/9_season','BB/9_season','xFIP_season','CSW%_season','xERA_season',
                    'runs_rolling10_team','OBP_team','SLG_team','K_rate_team','BB_rate_team',
                    'is_home_team']

    X = pd.DataFrame(0, index=df.index, columns=feature_cols)
    X['inning'] = df['inning']
    X['pitcher'] = df['pitcher']
    X['season'] = df['season']
    X['is_home_team'] = df['is_home_team']
    return df, X


def main():
    today = datetime.today().strftime('%Y-%m-%d')
    games = fetch_games(today)
    if games.empty:
        print('No games with both pitchers found.')
        return
    games, X = build_features(games, today)
    model = xgb.Booster()
    model.load_model('xgboost_yrfi_final.json')
    dmat = xgb.DMatrix(X)
    games['P_YRFI'] = model.predict(dmat)
    games['P_NRFI'] = 1 - games['P_YRFI']

    # Pivot to get separate predictions for each half inning
    pivot = games.pivot_table(
        index=["game_id", "away_name", "home_name"],
        columns="inning_topbot",
        values=["P_YRFI", "P_NRFI"],
    ).reset_index()

    # Flatten the MultiIndex columns produced by the pivot
    pivot.columns = ["_".join(col).strip("_") for col in pivot.columns.to_flat_index()]

    # Calculate inning totals from the top and bottom probabilities
    pivot["P_YRFI_total"] = 1 - (1 - pivot.get("P_YRFI_Top", 0)) * (1 - pivot.get("P_YRFI_Bot", 0))
    pivot["P_NRFI_total"] = 1 - pivot["P_YRFI_total"]

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

    result.to_csv('predictions.txt', index=False, sep='\t', float_format='%.6f')
    print(result)

if __name__ == '__main__':
    main()
