# test_utils.py
from utils import (
    load_xgb_from_json,
    load_feature_bank,
    build_feature_row,
    predict_match
)
import pandas as pd

MODEL_PATH   = "./models/xgb_model.json"
PARQUET_PATH = "./Datasets/final_tennis_dataset_symmetric.parquet"

# ---------------------------------------------------
# Test 1 : Load model
# ---------------------------------------------------
print("\nTest 1 – Loading XGBoost model")
model = load_xgb_from_json(MODEL_PATH)

# ---------------------------------------------------
# Test 2 : Load feature bank (cutoff < Roland Garros 2025)
# ---------------------------------------------------
print("\nTest 2 – Loading feature bank")
bank = load_feature_bank(PARQUET_PATH, cutoff="2025-05-20")
print(bank.head(2))

# ---------------------------------------------------
# Test 3 : Get player IDs by name (from feature bank)
# ---------------------------------------------------
print("\nTest 3 – Mapping player names to IDs")
def get_id_by_name(bank: pd.DataFrame, name: str) -> int:
    hits = bank[bank['PLAYER1_NAME'].str.contains(name, case=False, na=False)]
    if len(hits) == 0:
        raise ValueError(f"Aucun joueur trouvé pour le nom: {name}")
    if len(hits['PLAYER1_ID'].unique()) > 1:
        print(hits[['PLAYER1_ID', 'PLAYER1_NAME']].drop_duplicates())
        raise ValueError(f"Plusieurs IDs trouvés pour {name}")
    return hits['PLAYER1_ID'].iloc[0]

p1 = get_id_by_name(bank, "Alcaraz")
p2 = get_id_by_name(bank, "Sinner")
print(f"Alcaraz ID: {p1}, Sinner ID: {p2}")

# ---------------------------------------------------
# Test 4 : Build feature row
# ---------------------------------------------------
print("\nTest 4 – Building feature row for (Alcaraz vs Sinner) on CLAY")
X = build_feature_row(p1, p2, bank, surface="CLAY")
print(X.T)

# ---------------------------------------------------
# Test 5 : Predict match
# ---------------------------------------------------
print("\nTest 5 – Predicting match outcome (Alcaraz vs Sinner)")
p1_prob, p2_prob = predict_match(model, p1, p2, bank, surface="CLAY")
print(f"Probability Alcaraz wins: {p1_prob:.2%}")
print(f"Probability Sinner wins : {p2_prob:.2%}")