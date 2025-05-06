# src/utils.py
from __future__ import annotations
import joblib, re, random
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. I/O
# ------------------------------------------------------------------
def load_model(path: str | Path):
    """Load a fitted XGBoost model (joblib dump)."""
    return joblib.load(path)

def load_feature_bank(parquet_path: str | Path, cutoff: str = "2025-05-20") -> pd.DataFrame:
    """
    Keep the most recent row BEFORE `cutoff` for each player.
    Returns a DataFrame indexed by PLAYER_ID.
    """
    df = pd.read_parquet(parquet_path)
    df["DATE"] = pd.to_datetime(df["TOURNEY_DATE"].astype(str))
    df = df[df["DATE"] < cutoff]
    idx = df.groupby("PLAYER1_ID")["DATE"].idxmax()
    feature_bank = df.loc[idx].set_index("PLAYER1_ID")
    return feature_bank

# ------------------------------------------------------------------
# 2. Feature engineering for inference
# ------------------------------------------------------------------
_DIFF_COLS = [
    "ATP_POINT_DIFF", "ATP_RANK_DIFF", "AGE_DIFF", "HEIGHT_DIFF",
    "H2H_TOTAL_DIFF", "H2H_SURFACE_DIFF",
    "ELO_DIFF", "ELO_SURFACE_DIFF"
]

def _rename_cols(series: pd.Series, player: str) -> pd.Series:
    """Rename PLAYER1_/PLAYER2_ prefix according to `player`."""
    other = "PLAYER2" if player == "PLAYER1" else "PLAYER1"
    s = series.rename(lambda c: re.sub(rf"^{other}_", f"{player}_", c))
    return s

def build_feature_row(p1: int, p2: int, bank: pd.DataFrame, surface: str = "Clay") -> pd.DataFrame:
    """Return one DataFrame row with correct schema for the model."""
    row_p1 = _rename_cols(bank.loc[p1], "PLAYER1")
    row_p2 = _rename_cols(bank.loc[p2], "PLAYER2")

    # concatenate
    row = pd.concat([row_p1, row_p2])
    # ensure diff / ratio
    row["ATP_POINT_DIFF"]  = row["PLAYER1_RANK_POINTS"] - row["PLAYER2_RANK_POINTS"]
    row["ATP_RANK_DIFF"]   = row["PLAYER2_RANK"] - row["PLAYER1_RANK"]
    row["AGE_DIFF"]        = row["PLAYER1_AGE"] - row["PLAYER2_AGE"]
    row["HEIGHT_DIFF"]     = row["PLAYER1_HT"]  - row["PLAYER2_HT"]
    row["RANK_RATIO"]      = row["PLAYER1_RANK"] / row["PLAYER2_RANK"]
    # same for Elo diff etc. (already present but overwrite to be safe)

    # surfaces one‑hot
    for s in ["CLAY", "GRASS", "HARD", "CARPET"]:
        row[f"SURFACE_{s}"] = 1 if surface.upper() == s else 0

    return row.to_frame().T  # shape (1, n_features)

def predict_match(model, p1, p2, bank, surface="Clay", rng=random) -> float:
    X = build_feature_row(p1, p2, bank, surface)
    p = float(model.predict_proba(X)[0, 1])  # prob that PLAYER1 wins
    winner = p1 if rng.random() < p else p2
    return p, winner

# ------------------------------------------------------------------
# 3. Tournament simulation
# ------------------------------------------------------------------
def simulate_round(matches: List[Tuple[int, int]], model, bank, surface, rng) -> List[int]:
    winners = []
    for p1, p2 in matches:
        _, w = predict_match(model, p1, p2, bank, surface, rng)
        winners.append(w)
    return winners

def simulate_tournament(draw: List[Tuple[int, int]], model, bank,
                        surface="Clay", n_runs: int = 1000, seed: int = 42
                        ) -> Dict[int, int]:
    rng = random.Random(seed)
    champions = {pid: 0 for pid in bank.index}
    for _ in range(n_runs):
        current = list(draw)
        while len(current) > 1:
            current = list(zip(*[iter(simulate_round(current, model, bank, surface, rng))]*2))
        champion = current[0][0]
        champions[champion] += 1
    return champions
