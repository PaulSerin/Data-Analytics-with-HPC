from pathlib import Path
from utils import (
    get_latest_features_by_player,
    get_latest_features_by_surface,
    player_name_to_id,
    get_player_stats,
    build_match_row,
    load_trained_model,
    predict_match,
    run_monte_carlo
)

PARQUET_PATH = Path("./Datasets/final_tennis_dataset_symmetric.parquet")
PLAYERS_CSV = Path("./data/atp_players.csv")
MODEL_PATH = Path("./models/xgb_model.json")
CUTOFF_DATE = "2024-05-20"

# Load latest features
global_df, surface_dfs = get_latest_features_by_surface(PARQUET_PATH, CUTOFF_DATE)

# Load model
model = load_trained_model(MODEL_PATH)

# Convert names to IDs
alcaraz_id = player_name_to_id("Carlos", "Alcaraz", PLAYERS_CSV)
sinner_id = player_name_to_id("Jannik", "Sinner", PLAYERS_CSV)

# Evaluate on different surfaces
for surface in ["CLAY", "HARD"]:
    print(f"\n--- Match on {surface} ---")

    # Predict win probability
    prob = predict_match(alcaraz_id, sinner_id, surface, model, global_df, surface_dfs)
    print(f"Alcaraz win probability: {prob:.2%}")
    print(f"Sinner  win probability: {1 - prob:.2%}")

    # Run Monte Carlo simulations
    for n in [100, 200, 500]:
        winrate = run_monte_carlo(alcaraz_id, sinner_id, model, global_df, surface_dfs, n=n, surface=surface)
        print(f"Monte Carlo ({n} runs): Alcaraz wins {winrate:.2%}")