#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import pandas as pd

def load_utils(utils_path: Path):
    """Dynamically load your utils.py from the given path."""
    return SourceFileLoader("utils", str(utils_path)).load_module()

def simulate_tournament(bracket_init, surface, model, global_df, surface_dfs, utils):
    """
    Run one tournament, returning (champion_id, [finalist1_id, finalist2_id], final_win_prob).
    Fallback: if a player's features are missing, the other advances automatically.
    """
    pairs = list(bracket_init)
    rounds = [
        '1st Round','2nd Round','3rd Round','4th Round',
        'Quarterfinals','Semifinals','The Final'
    ]
    # play up to semis
    for rnd in rounds[:-1]:
        winners = []
        for p1, p2 in pairs:
            if p1 is None:
                winners.append(p2); continue
            if p2 is None:
                winners.append(p1); continue
            try:
                prob_p1 = utils.predict_match(p1, p2, surface, model, global_df, surface_dfs)
                winner = p1 if random.random() < prob_p1 else p2
            except KeyError as e:
                msg = str(e)
                # missing p1 → p2, missing p2 → p1
                if f"Player {p1}" in msg:
                    winner = p2
                elif f"Player {p2}" in msg:
                    winner = p1
                else:
                    winner = p2
            winners.append(winner)
        # build next round pairs
        pairs = [(winners[i], winners[i+1] if i+1 < len(winners) else None)
                 for i in range(0, len(winners), 2)]

    # final: one pair only
    p1, p2 = pairs[0]
    finalists = (p1, p2)
    # determine final winner & win probability
    if p1 is None:
        return p2, finalists, 1.0
    if p2 is None:
        return p1, finalists, 1.0

    prob_p1 = utils.predict_match(p1, p2, surface, model, global_df, surface_dfs)
    if random.random() < prob_p1:
        return p1, finalists, prob_p1
    else:
        return p2, finalists, (1 - prob_p1)

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Monte Carlo AO 2025 simulation"
    )
    parser.add_argument("--utils-path", type=Path, required=True,
                        help="Path to utils.py")
    parser.add_argument("--json-draw",    type=Path, required=True,
                        help="JSON draw file")
    parser.add_argument("--parquet",      type=Path, required=True,
                        help="Parquet with historical features")
    parser.add_argument("--model",        type=Path, required=True,
                        help="XGBoost model JSON")
    parser.add_argument("--cutoff",       type=str, default="2025-01-01",
                        help="Cutoff date for features")
    parser.add_argument("--runs-per-job", type=int, required=True,
                        help="Number of MC simulations this job should run")
    parser.add_argument("--job-index",    type=int, required=True,
                        help="SLURM array job index (0-based)")
    parser.add_argument("--output-dir",   type=Path, required=True,
                        help="Directory to write partial results")
    args = parser.parse_args()

    # load utils
    utils = load_utils(args.utils_path)

    # load draw
    with open(args.json_draw, 'r', encoding='utf-8') as f:
        tournament = json.load(f)
    surface = tournament["surface"]

    # build ID→name map
    id_to_name = {}
    for m in tournament["matches"]:
        for side in ("player1","player2"):
            pid = m[side]["id"]
            name = m[side]["name"]
            if pid is not None:
                id_to_name[pid] = name

    # load features + model once
    global_df, surface_dfs = utils.get_latest_features_by_surface(
        args.parquet, args.cutoff
    )
    model = utils.load_trained_model(args.model)

    # initial bracket
    first_round = sorted(
        [m for m in tournament["matches"] if m["round"]=="1st Round"],
        key=lambda m: m["match_id"]
    )
    bracket_init = [(m["player1"]["id"], m["player2"]["id"])
                    for m in first_round]

    # run your assigned slice of simulations
    start = args.job_index * args.runs_per_job
    end   = start + args.runs_per_job
    champion_counts = {}
    final_probs = {}

    for i in range(start, end):
        random.seed(i)
        champ, (f1, f2), prob = simulate_tournament(
            bracket_init, surface, model, global_df, surface_dfs, utils
        )
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
        final_probs.setdefault(champ, []).append(prob)

    # build a name‐aware stats list
    champion_stats = [
        {
            "id":  pid,
            "name": id_to_name.get(pid, "Unknown"),
            "wins": count
        }
        for pid, count in champion_counts.items()
    ]

    # write partial results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "job_index":       args.job_index,
        "runs":            args.runs_per_job,
        "champion_stats":  champion_stats,
        "final_probs":     final_probs,
    }
    out_path = args.output_dir / f"partial_{args.job_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote partial results to {out_path}")

if __name__ == "__main__":
    main()