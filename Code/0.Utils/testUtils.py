from pathlib import Path
from utils import (
    get_latest_features_by_surface,
    player_name_to_id,
    get_player_stats,
    build_match_row,
    load_trained_model,
    predict_match,
    run_monte_carlo
)

# Constants
PARQUET_PATH = Path("././Datasets/final_tennis_dataset_symmetric.parquet")
PLAYERS_CSV = Path("././Data/players/atp_players.csv")
MODEL_PATH = Path("././Models/xgb_model.json")
CUTOFF_DATE = "2024-05-20"

# Load feature data
global_df, surface_dfs = get_latest_features_by_surface(PARQUET_PATH, CUTOFF_DATE)

# Load model
model = load_trained_model(MODEL_PATH)

# Player full names
player1_name = "Carlos Alcaraz"
player2_name = "Jannik Sinner"

# Split full names into first and last
player1_first, player1_last = player1_name.split(maxsplit=1)
player2_first, player2_last = player2_name.split(maxsplit=1)

# Get player IDs
player1_id = player_name_to_id(player1_first, player1_last, PLAYERS_CSV)
player2_id = player_name_to_id(player2_first, player2_last, PLAYERS_CSV)

# Evaluate on different surfaces
for surface in ["CLAY", "HARD"]:
    print(f"\n--- Match on {surface} ---")

    # Print player stats (key metrics)
    p1_stats = get_player_stats(player1_id, "PLAYER1_", surface, global_df, surface_dfs)
    p2_stats = get_player_stats(player2_id, "PLAYER2_", surface, global_df, surface_dfs)

    def print_summary(name, stats, prefix):
        print(f"\n{name} ({prefix}) stats on {surface}:")
        print(f"  ELO:                    {stats[f'{prefix}ELO_BEFORE']:.0f}")
        print(f"  ELO_SURFACE:           {stats[f'{prefix}ELO_SURFACE_BEFORE']:.0f}")
        print(f"  RANK:                  {int(stats[f'{prefix}RANK'])}")
        print(f"  RANK_POINTS:           {int(stats[f'{prefix}RANK_POINTS'])}")
        print(f"  AGE:                   {stats[f'{prefix}AGE']:.1f}")
        print(f"  HEIGHT (cm):           {int(stats[f'{prefix}HT'])}")
        print(f"  SURFACE_MATCHES:       {int(stats[f'{prefix}SURFACE_MATCHES'])}")
        print(f"  LAST_5_WINRATE:        {stats.get(f'{prefix}LAST_5_WINRATE', 0):.2%}")

    print_summary(player1_name, p1_stats, "PLAYER1_")
    print_summary(player2_name, p2_stats, "PLAYER2_")

    # Predict win probability
    prob = predict_match(player1_id, player2_id, surface, model, global_df, surface_dfs)
    print(f"\nPrediction: {player1_name} win probability: {prob:.2%}")
    print(f"             {player2_name} win probability: {1 - prob:.2%}")

    # Monte Carlo simulations
    for n in [100, 200, 500]:
        winrate = run_monte_carlo(player1_id, player2_id, model, global_df, surface_dfs, n=n, surface=surface)
        print(f"Monte Carlo ({n} runs): {player1_name} wins {winrate:.2%}")