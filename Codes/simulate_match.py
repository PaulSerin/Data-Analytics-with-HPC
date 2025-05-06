# simulate_match.py
from utils import load_xgb_from_json, load_feature_bank, player_name_to_id, predict_match

MODEL_PATH   = "./models/xgb_model.json"
PARQUET_PATH = "./Datasets/final_tennis_dataset_symmetric.parquet"

def main():
    model = load_xgb_from_json(MODEL_PATH)
    bank  = load_feature_bank(PARQUET_PATH, cutoff="2025-05-20")

    alcaraz_id = player_name_to_id("Carlos", "Alcaraz")
    sinner_id  = player_name_to_id("Jannik", "Sinner")

    p_alcaraz, p_sinner = predict_match(model, alcaraz_id, sinner_id, bank, surface="CLAY")

    print(f"Roland‑Garros (clay) – Alcaraz vs Sinner")
    print(f"----------------------------------------")
    print(f"Carlos Alcaraz win prob : {p_alcaraz:.1%}")
    print(f"Jannik Sinner win prob : {p_sinner:.1%}")

if __name__ == "__main__":
    main()
