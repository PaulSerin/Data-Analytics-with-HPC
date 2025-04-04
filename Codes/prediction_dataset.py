import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque

#############################
# 1. Chargement et Concatenation
#############################
full_data = pd.concat([
    pd.read_csv(f"./data/atp_matches_{year}.csv") for year in range(1968, 2025)
], ignore_index=True)

full_data['year'] = full_data['tourney_date'].astype(str).str[:4].astype(int)
filtered_data = full_data[full_data['year'] >= 2000].copy()

subset_cols = [
    'winner_id', 'loser_id', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age',
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    'winner_rank_points', 'loser_rank_points', 'winner_rank', 'loser_rank', 'surface', 'score', 'minutes'
]
df = filtered_data.dropna(subset=subset_cols).copy()

#############################
# 2. Feature Engineering Basique
#############################
df['ATP_POINT_DIFF'] = df['winner_rank_points'] - df['loser_rank_points']
df['ATP_RANK_DIFF'] = df['loser_rank'] - df['winner_rank']
df['AGE_DIFF'] = df['winner_age'] - df['loser_age']
df['HEIGHT_DIFF'] = df['winner_ht'] - df['loser_ht']
df['RANK_RATIO'] = df['winner_rank'] / df['loser_rank']
df['SERVE_DOMINANCE'] = df['w_ace'] - df['l_ace']

# Cette colonne n'existait que pour le winner, on va en créer aussi pour le loser
df['BP_EFFICIENCY_WINNER'] = df['w_bpSaved'] / df['w_bpFaced'].replace(0, 1)
df['BP_EFFICIENCY_LOSER'] = df['l_bpSaved'] / df['l_bpFaced'].replace(0, 1)

df['surface_raw'] = df['surface']
df = pd.get_dummies(df, columns=['surface'], prefix='SURFACE')
df.sort_values(by='tourney_date', inplace=True)
df.reset_index(drop=True, inplace=True)

#############################
# 3. H2H Features
#############################
h2h_total = {}
h2h_surface = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    winner, loser, surface = row['winner_id'], row['loser_id'], row['surface_raw']
    pair = tuple(sorted([winner, loser]))
    pair_surface = (pair, surface)

    if pair not in h2h_total:
        h2h_total[pair] = {'winner_wins': 0, 'loser_wins': 0}
    if pair_surface not in h2h_surface:
        h2h_surface[pair_surface] = {'winner_wins': 0, 'loser_wins': 0}

    total_score = h2h_total[pair]
    surface_score = h2h_surface[pair_surface]

    if winner < loser:
        df.at[i, 'H2H_TOTAL_DIFF'] = total_score['winner_wins'] - total_score['loser_wins']
        df.at[i, 'H2H_SURFACE_DIFF'] = surface_score['winner_wins'] - surface_score['loser_wins']
        h2h_total[pair]['winner_wins'] += 1
        h2h_surface[pair_surface]['winner_wins'] += 1
    else:
        df.at[i, 'H2H_TOTAL_DIFF'] = total_score['loser_wins'] - total_score['winner_wins']
        df.at[i, 'H2H_SURFACE_DIFF'] = surface_score['loser_wins'] - surface_score['winner_wins']
        h2h_total[pair]['loser_wins'] += 1
        h2h_surface[pair_surface]['loser_wins'] += 1

#############################
# 4. Nombre de matchs joués (total + surface)
#############################
df['WINNER_TOTAL_MATCHES'], df['LOSER_TOTAL_MATCHES'] = 0, 0
df['WINNER_SURFACE_MATCHES'], df['LOSER_SURFACE_MATCHES'] = 0, 0
total_matches, surface_matches = {}, {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    w, l, s = row['winner_id'], row['loser_id'], row['surface_raw']
    df.at[i, 'WINNER_TOTAL_MATCHES'] = total_matches.get(w, 0)
    df.at[i, 'LOSER_TOTAL_MATCHES'] = total_matches.get(l, 0)
    df.at[i, 'WINNER_SURFACE_MATCHES'] = surface_matches.get((w, s), 0)
    df.at[i, 'LOSER_SURFACE_MATCHES'] = surface_matches.get((l, s), 0)
    total_matches[w] = total_matches.get(w, 0) + 1
    total_matches[l] = total_matches.get(l, 0) + 1
    surface_matches[(w, s)] = surface_matches.get((w, s), 0) + 1
    surface_matches[(l, s)] = surface_matches.get((l, s), 0) + 1

#############################
# 5. Winrate Sliding Windows
#############################
windows = [3, 5, 10, 25, 50, 100]
for w in windows:
    df[f'WINNER_LAST_{w}_WINRATE'], df[f'LOSER_LAST_{w}_WINRATE'] = 0.0, 0.0

player_results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    w_id, l_id = row['winner_id'], row['loser_id']
    player_results.setdefault(w_id, deque(maxlen=100))
    player_results.setdefault(l_id, deque(maxlen=100))
    for window_size in windows:
        last_wins_w = list(player_results[w_id])[-window_size:]
        last_wins_l = list(player_results[l_id])[-window_size:]
        df.at[i, f'WINNER_LAST_{window_size}_WINRATE'] = sum(last_wins_w) / (len(last_wins_w) or 1)
        df.at[i, f'LOSER_LAST_{window_size}_WINRATE'] = sum(last_wins_l) / (len(last_wins_l) or 1)
    player_results[w_id].append(1)
    player_results[l_id].append(0)

#############################
# 6. Stats cumulées (Ace, DF, etc.)
#############################
def mean(arr):
    return sum(arr) / len(arr) if arr else 0.5

for k in [3, 5, 10, 20, 50, 100, 200, 300, 2000]:
    last_k = defaultdict(lambda: defaultdict(lambda: deque(maxlen=k)))
    data_dict = {
        f"P_ACE_WINNER_LAST_{k}": [], f"P_ACE_LOSER_LAST_{k}": [],
        f"P_DF_WINNER_LAST_{k}": [], f"P_DF_LOSER_LAST_{k}": [],
        f"P_1STIN_WINNER_LAST_{k}": [], f"P_1STIN_LOSER_LAST_{k}": [],
        f"P_1STWON_WINNER_LAST_{k}": [], f"P_1STWON_LOSER_LAST_{k}": [],
        f"P_2NDWON_WINNER_LAST_{k}": [], f"P_2NDWON_LOSER_LAST_{k}": [],
        f"P_BPSAVED_WINNER_LAST_{k}": [], f"P_BPSAVED_LOSER_LAST_{k}": []
    }
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        w_id, l_id = row.winner_id, row.loser_id
        for stat in ["p_ace", "p_df", "p_1stIn", "p_1stWon", "p_2ndWon", "p_bpSaved"]:
            data_dict[f"{stat.upper()}_WINNER_LAST_{k}"].append(mean(last_k[w_id][stat]))
            data_dict[f"{stat.upper()}_LOSER_LAST_{k}"].append(mean(last_k[l_id][stat]))

        # Mise à jour de l'historique après ce match
        if row.w_svpt:
            last_k[w_id]["p_ace"].append(100 * row.w_ace / row.w_svpt)
            last_k[w_id]["p_df"].append(100 * row.w_df / row.w_svpt)
            last_k[w_id]["p_1stIn"].append(100 * row.w_1stIn / row.w_svpt)
            if row.w_svpt != row.w_1stIn:
                last_k[w_id]["p_2ndWon"].append(100 * row.w_2ndWon / (row.w_svpt - row.w_1stIn))
        if row.l_svpt:
            last_k[l_id]["p_ace"].append(100 * row.l_ace / row.l_svpt)
            last_k[l_id]["p_df"].append(100 * row.l_df / row.l_svpt)
            last_k[l_id]["p_1stIn"].append(100 * row.l_1stIn / row.l_svpt)
            if row.l_svpt != row.l_1stIn:
                last_k[l_id]["p_2ndWon"].append(100 * row.l_2ndWon / (row.l_svpt - row.l_1stIn))
        if row.w_1stIn:
            last_k[w_id]["p_1stWon"].append(100 * row.w_1stWon / row.w_1stIn)
        if row.l_1stIn:
            last_k[l_id]["p_1stWon"].append(100 * row.l_1stWon / row.l_1stIn)
        if row.w_bpFaced:
            last_k[w_id]["p_bpSaved"].append(100 * row.w_bpSaved / row.w_bpFaced)
        if row.l_bpFaced:
            last_k[l_id]["p_bpSaved"].append(100 * row.l_bpSaved / row.l_bpFaced)

    df = pd.concat([df, pd.DataFrame(data_dict)], axis=1)

#############################
# 7. Tennis ELO + Surface ELO
#############################
elo_global = defaultdict(lambda: 1500)
elo_surface = defaultdict(lambda: 1500)
df['WINNER_ELO_BEFORE'], df['LOSER_ELO_BEFORE'], df['ELO_DIFF'] = 0.0, 0.0, 0.0
df['WINNER_ELO_SURFACE_BEFORE'], df['LOSER_ELO_SURFACE_BEFORE'], df['ELO_SURFACE_DIFF'] = 0.0, 0.0, 0.0
for i, row in tqdm(df.iterrows(), total=len(df)):
    w_id, l_id, surf = row['winner_id'], row['loser_id'], row['surface_raw']
    E_w = 1 / (1 + 10 ** ((elo_global[l_id] - elo_global[w_id]) / 400))
    E_w_surf = 1 / (1 + 10 ** ((elo_surface[(l_id, surf)] - elo_surface[(w_id, surf)]) / 400))

    df.at[i, 'WINNER_ELO_BEFORE'] = elo_global[w_id]
    df.at[i, 'LOSER_ELO_BEFORE'] = elo_global[l_id]
    df.at[i, 'ELO_DIFF'] = elo_global[w_id] - elo_global[l_id]

    df.at[i, 'WINNER_ELO_SURFACE_BEFORE'] = elo_surface[(w_id, surf)]
    df.at[i, 'LOSER_ELO_SURFACE_BEFORE'] = elo_surface[(l_id, surf)]
    df.at[i, 'ELO_SURFACE_DIFF'] = elo_surface[(w_id, surf)] - elo_surface[(l_id, surf)]

    # Mise à jour ELO
    K = 32
    elo_global[w_id] += K * (1 - E_w)
    elo_global[l_id] += K * (0 - (1 - E_w))
    elo_surface[(w_id, surf)] += K * (1 - E_w_surf)
    elo_surface[(l_id, surf)] += K * (0 - (1 - E_w_surf))

#############################
# 8. Export (avant duplication)
#############################
# À ce stade, df contient les colonnes "winner_...", "loser_...", + les '..._DIFF', etc.

# On va d'abord sauvegarder une version "classique"
df.to_csv('../Datasets/final_tennis_dataset.csv', index=False)
print("Dataset final (winner/loser) sauvegardé dans 'final_tennis_dataset.csv'")

#############################
# 9. Création du dataset symétrique (player1/player2)
#############################

# 9.1 - Définir les colonnes "diff" à inverser dans la version backward
diff_cols_to_invert = [
    'ATP_POINT_DIFF', 'ATP_RANK_DIFF', 'AGE_DIFF', 'HEIGHT_DIFF',
    'SERVE_DOMINANCE', 'H2H_TOTAL_DIFF', 'H2H_SURFACE_DIFF',
    'ELO_DIFF', 'ELO_SURFACE_DIFF'
    # RANK_RATIO peut être inversé en 1 / ratio si tu veux
]

# 9.2 - Créer deux copies : forward (target=1) et backward (target=0)

# Forward: winner -> player1, loser -> player2
df_forward = df.copy()
df_forward['target'] = 1
df_forward.columns = [
    col.replace('winner', 'PLAYER1')
       .replace('loser', 'PLAYER2')
       .replace('WINNER', 'PLAYER1')
       .replace('LOSER', 'PLAYER2')
       .replace('w_', 'PLAYER1_')
       .replace('l_', 'PLAYER2_')
       .upper()
    for col in df_forward.columns
]


# Backward: winner -> player2, loser -> player1
df_backward = df.copy()
df_backward['target'] = 0
df_backward.columns = [
    col.replace('winner', 'PLAYER2')
       .replace('loser', 'PLAYER1')
       .replace('WINNER', 'PLAYER2')
       .replace('LOSER', 'PLAYER1')
       .replace('w_', 'PLAYER2_')
       .replace('l_', 'PLAYER1_')
       .upper()
    for col in df_backward.columns
]

# 9.3 - Inverser les colonnes "diff" dans df_backward
for c in diff_cols_to_invert:
    # Après le rename, la colonne ELO_DIFF est toujours ELO_DIFF
    # Mais le sens est inversé pour winner->player2
    df_backward[c] = - df_backward[c]

# EXEMPLE pour RANK_RATIO :
#   forward = winner_rank/loser_rank
#   backward devrait être loser_rank/winner_rank = 1 / ratio
if 'RANK_RATIO' in df_backward.columns:
    df_backward['RANK_RATIO'] = 1 / df_backward['RANK_RATIO']

# 9.4 - Concaténer
df_final = pd.concat([df_forward, df_backward], ignore_index=True)

# 9.5 - Sauvegarder
df_final.to_csv('../Datasets/final_tennis_dataset_symmetric.csv', index=False)
print("Dataset symétrique (player1/player2) sauvegardé dans 'final_tennis_dataset_symmetric.csv'")