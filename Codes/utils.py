import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import random
import re

SURFACES = ["CLAY", "GRASS", "HARD", "CARPET"]

def get_latest_features_by_player(parquet_path: Path, cutoff: str) -> pd.DataFrame:
    """
    Return the most recent match row per player (any surface), before the cutoff date.
    Result is indexed by PLAYER1_ID.
    """
    df = pd.read_parquet(parquet_path)
    df["DATE"] = pd.to_datetime(df["TOURNEY_DATE"].astype(str))
    df = df[df["DATE"] < cutoff]
    idx = df.groupby("PLAYER1_ID")["DATE"].idxmax()
    latest_df = df.loc[idx].set_index("PLAYER1_ID")
    return latest_df


def get_latest_features_by_surface(parquet_path: Path, cutoff: str):
    """
    Return two dictionaries:
    - `global_df` : latest match row per player (any surface)
    - `surface_dfs` : latest match row per player on each surface
    """
    df = pd.read_parquet(parquet_path)
    df["DATE"] = pd.to_datetime(df["TOURNEY_DATE"].astype(str))
    df = df[df["DATE"] < cutoff]

    # Global features
    idx = df.groupby("PLAYER1_ID")["DATE"].idxmax()
    global_df = df.loc[idx].set_index("PLAYER1_ID")

    # Per-surface features
    surface_dfs = {}
    for surf in ["CLAY", "GRASS", "HARD", "CARPET"]:
        sub = df[df[f"SURFACE_{surf}"] == 1]
        if not sub.empty:
            idx = sub.groupby("PLAYER1_ID")["DATE"].idxmax()
            surface_dfs[surf] = sub.loc[idx].set_index("PLAYER1_ID")

    return global_df, surface_dfs


def player_name_to_id(first: str, last: str, csv_path: Path) -> int:
    """
    Convert a player's first and last name to their unique ATP ID,
    based on the atp_players.csv master file.
    """
    players = pd.read_csv(csv_path, names=[
        "id", "first", "last", "hand", "dob", "ioc", "height", "wikidata"
    ])
    mask = (
        players["first"].str.lower().str.strip() == first.lower().strip()
    ) & (
        players["last"].str.lower().str.strip() == last.lower().strip()
    )
    matches = players[mask]
    if len(matches) == 0:
        raise ValueError(f"{first} {last} not found in the dataset.")
    if len(matches) > 1:
        print(matches[["id", "first", "last"]])
        raise ValueError(f"{first} {last} is ambiguous in the dataset.")
    return int(matches.iloc[0]["id"])


def get_player_stats(player_id: int,
                              prefix: str,
                              surface: str,
                              global_df: pd.DataFrame,
                              surface_dfs: dict) -> pd.Series:
    """
    Return the latest feature row for a player, prefixed accordingly (PLAYER1_ or PLAYER2_).
    Tries surface-specific stats first, then global.
    """
    surf_df = surface_dfs.get(surface.upper())
    if surf_df is not None and player_id in surf_df.index:
        row = surf_df.loc[player_id]
    elif player_id in global_df.index:
        row = global_df.loc[player_id]
    else:
        raise KeyError(f"Player {player_id} not found for surface {surface}")
    return row.filter(regex="^PLAYER1_").rename(lambda c: c.replace("PLAYER1_", prefix))


def build_match_row(p1: int, p2: int, surface: str,
                            global_df: pd.DataFrame, surface_dfs: dict) -> pd.DataFrame:
    """
    Construct the full feature row for a match between two players.
    Includes both players' stats and engineered difference metrics.
    """
    r1 = get_player_stats(p1, "PLAYER1_", surface, global_df, surface_dfs)
    r2 = get_player_stats(p2, "PLAYER2_", surface, global_df, surface_dfs)
    feat = pd.concat([r1, r2])

    # Engineered features
    feat["ATP_POINT_DIFF"] = feat["PLAYER1_RANK_POINTS"] - feat["PLAYER2_RANK_POINTS"]
    feat["ATP_RANK_DIFF"] = feat["PLAYER2_RANK"] - feat["PLAYER1_RANK"]
    feat["AGE_DIFF"] = feat["PLAYER1_AGE"] - feat["PLAYER2_AGE"]
    feat["HEIGHT_DIFF"] = feat["PLAYER1_HT"] - feat["PLAYER2_HT"]
    feat["RANK_RATIO"] = feat["PLAYER1_RANK"] / feat["PLAYER2_RANK"]
    feat["ELO_SURFACE_DIFF"] = feat["PLAYER1_ELO_SURFACE_BEFORE"] - feat["PLAYER2_ELO_SURFACE_BEFORE"]

    for s in ["CLAY", "GRASS", "HARD", "CARPET"]:
        feat[f"SURFACE_{s}"] = int(surface.upper() == s)

    return feat.to_frame().T


def load_trained_model(path: Path):
    """
    Load a trained XGBoost model from the specified path.
    """
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def predict_match(p1: int, p2: int, surface: str,
                          model, global_df, surface_dfs) -> float:
    """
    Predict the probability that Player 1 beats Player 2 on the given surface.
    """
    X = build_match_row(p1, p2, surface, global_df, surface_dfs)
    X = X.reindex(columns=model.get_booster().feature_names, fill_value=0.0)
    X = X.astype(np.float32)
    return float(model.predict_proba(X)[0, 1])


def run_monte_carlo(p1: int, p2: int, model,
                    global_df, surface_dfs,
                    n: int = 1000, surface: str = "CLAY", seed: int = 0) -> float:
    """
    Simulate `n` matches between Player 1 and Player 2 and return win percentage.
    """
    rng = random.Random(seed)
    wins = sum(
        rng.random() < predict_match(p1, p2, surface, model, global_df, surface_dfs)
        for _ in range(n)
    )
    return wins / n
