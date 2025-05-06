# utils.py  ────────────────────────────────────────────────────────────
from __future__ import annotations
import re, random
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import xgboost as xgb

# ------------------------------------------------------------------ #
# 0.  Helpers
# ------------------------------------------------------------------ #
def _dbg(msg: str, on: bool = True):
    if on:
        print(f"[utils] {msg}")

DEBUG = True                       # ← passe à False pour couper les prints

# ------------------------------------------------------------------ #
# 1.  Model I/O
# ------------------------------------------------------------------ #
def load_xgb_from_json(json_path: str | Path,
                       debug: bool = DEBUG) -> xgb.XGBClassifier:
    """
    Reload an XGBClassifier previously saved with `save_model(<path>.json)`.
    Re‑hydrates missing wrapper attributes so that predict_proba works.
    """
    _dbg(f"Loading XGB model from {json_path}", debug)
    model = xgb.XGBClassifier()
    model.load_model(json_path)

    # --- re‑hydrate scikit‑wrapper bits (classes, encoder) ------------
    from xgboost.compat import XGBoostLabelEncoder
    model._le = XGBoostLabelEncoder().fit(model.classes_)
    _dbg(f"Model loaded – {model.get_booster().best_ntree_limit or model.n_estimators} trees", debug)
    return model


# ------------------------------------------------------------------ #
# 2.  Feature bank
# ------------------------------------------------------------------ #
def load_feature_bank(parquet_path: str | Path,
                      cutoff: str = "2025-05-20",
                      debug: bool = DEBUG) -> pd.DataFrame:
    """
    Return latest row per player **before** `cutoff`.
    """
    _dbg(f"Reading Parquet {parquet_path}", debug)
    df = pd.read_parquet(parquet_path)
    df["DATE"] = pd.to_datetime(df["TOURNEY_DATE"].astype(str))
    df = df[df["DATE"] < cutoff]

    idx = df.groupby("PLAYER1_ID")["DATE"].idxmax()
    bank = df.loc[idx].set_index("PLAYER1_ID")
    _dbg(f"Feature bank built – {len(bank)} players, {bank.shape[1]} features", debug)
    return bank


# ------------------------------------------------------------------ #
# 3.  Name ↔︎ ID helper
# ------------------------------------------------------------------ #
def player_name_to_id(first: str, last: str,
                      players_csv: str | Path = "data/atp_players.csv",
                      debug: bool = DEBUG) -> int:
    players = pd.read_csv(players_csv, names=["id", "first", "last", "hand",
                                              "dob", "ioc", "height", "wikidataid"])
    hit = players[
        (players["first"].str.lower() == first.lower())
        & (players["last"].str.lower() == last.lower())
    ]
    if len(hit) != 1:
        raise ValueError(f"Unable to map {first} {last}")
    pid = int(hit.iloc[0]["id"])
    _dbg(f"Mapped {first} {last} → {pid}", debug)
    return pid


# ------------------------------------------------------------------ #
# 4.  Build one feature row
# ------------------------------------------------------------------ #
_SURFACES = ["CLAY", "GRASS", "HARD", "CARPET"]

def _rename_cols(row: pd.Series, tgt_prefix: str) -> pd.Series:
    other_prefix = "PLAYER1_" if tgt_prefix == "PLAYER2_" else "PLAYER2_"
    return row.rename(lambda c: re.sub(rf"^{other_prefix}", tgt_prefix, c))

def build_feature_row(p1: int, p2: int, bank: pd.DataFrame,
                      surface: str = "CLAY",
                      debug: bool = DEBUG) -> pd.DataFrame:
    row1 = _rename_cols(bank.loc[p1], "PLAYER1_")
    row2 = _rename_cols(bank.loc[p2], "PLAYER2_")
    feat = pd.concat([row1, row2])

    feat["ATP_POINT_DIFF"] = feat["PLAYER1_RANK_POINTS"] - feat["PLAYER2_RANK_POINTS"]
    feat["ATP_RANK_DIFF"]  = feat["PLAYER2_RANK"] - feat["PLAYER1_RANK"]
    feat["AGE_DIFF"]       = feat["PLAYER1_AGE"] - feat["PLAYER2_AGE"]
    feat["HEIGHT_DIFF"]    = feat["PLAYER1_HT"] - feat["PLAYER2_HT"]
    feat["RANK_RATIO"]     = feat["PLAYER1_RANK"] / feat["PLAYER2_RANK"]

    for s in _SURFACES:
        feat[f"SURFACE_{s}"] = 1 if s == surface.upper() else 0

    _dbg(f"Feature row built for ({p1} vs {p2}) – shape {feat.shape}", debug)
    return feat.to_frame().T


# ------------------------------------------------------------------ #
# 5.  Predict a single match
# ------------------------------------------------------------------ #
def predict_match(model: xgb.XGBClassifier, p1: int, p2: int,
                  bank: pd.DataFrame, surface: str = "CLAY",
                  debug: bool = DEBUG) -> Tuple[float, float]:
    X = build_feature_row(p1, p2, bank, surface, debug)
    p1_win = float(model.predict_proba(X)[0, 1])
    _dbg(f"P(Player1={p1} beats Player2={p2}) = {p1_win:.4f}", debug)
    return p1_win, 1.0 - p1_win
