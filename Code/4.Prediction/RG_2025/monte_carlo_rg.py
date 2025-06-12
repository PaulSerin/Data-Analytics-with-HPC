import json
import random
import argparse
import time
from pathlib import Path
from importlib.machinery import SourceFileLoader
import os

def load_utils(utils_path: Path):
    """Dynamically load your utils.py from the given path."""
    return SourceFileLoader("utils", str(utils_path)).load_module()

def simulate_tournament(bracket_init, surface, model, global_df, surface_dfs, utils):
    """
    Run one tournament, returning:
      - champion_id,
      - finalists tuple (p1_id, p2_id),
      - qf_participants (list of 8 ids),
      - sf_participants (list of 4 ids).
    Fallback: if a player's features are missing, the other advances automatically.
    """
    pairs = list(bracket_init)
    rounds = [
        '1st Round','2nd Round','3rd Round','4th Round',
        'Quarterfinals','Semifinals','The Final'
    ]
    qf_participants = []
    sf_participants = []

    # Iterate through rounds up to (and including) Semifinals
    for rnd in rounds[:-1]:
        winners = []
        for p1, p2 in pairs:
            if p1 is None:
                winners.append(p2)
                continue
            if p2 is None:
                winners.append(p1)
                continue
            try:
                prob_p1 = utils.predict_match(p1, p2, surface, model, global_df, surface_dfs)
                winner = p1 if random.random() < prob_p1 else p2
            except KeyError as e:
                msg = str(e)
                if f"Player {p1}" in msg:
                    winner = p2
                elif f"Player {p2}" in msg:
                    winner = p1
                else:
                    winner = p2
            winners.append(winner)

        # After 4th Round, winners advance to QF
        if rnd == '4th Round':
            qf_participants = winners.copy()
        # After Quarterfinals, winners advance to SF
        if rnd == 'Quarterfinals':
            sf_participants = winners.copy()

        # Pair winners for next round
        pairs = [(winners[i], winners[i+1] if i+1 < len(winners) else None)
                 for i in range(0, len(winners), 2)]

    # Final: one pair only
    p1, p2 = pairs[0]
    finalists = (p1, p2)

    # Determine champion
    if p1 is None:
        champion = p2
    elif p2 is None:
        champion = p1
    else:
        prob_p1 = utils.predict_match(p1, p2, surface, model, global_df, surface_dfs)
        champion = p1 if random.random() < prob_p1 else p2

    return champion, finalists, qf_participants, sf_participants

def main():
    t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Distributed Monte Carlo Roland Garros 2025 simulation"
    )
    parser.add_argument("--utils-path", type=Path, required=True,
                        help="Path to utils.py")
    parser.add_argument("--json-draw", type=Path, required=True,
                        help="JSON draw file")
    parser.add_argument("--parquet", type=Path, required=True,
                        help="Parquet with historical features")
    parser.add_argument("--model", type=Path, required=True,
                        help="XGBoost model JSON")
    parser.add_argument("--cutoff", type=str, default="2025-01-01",
                        help="Cutoff date for features")
    parser.add_argument("--runs-per-job", type=int, required=True,
                        help="Number of RG simulations this job should run")
    parser.add_argument("--job-index", type=int, required=True,
                        help="SLURM array job index (0-based)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write partial results")
    args = parser.parse_args()

    start = time.perf_counter()

    # Load utils, draw, feature snapshots, and model
    utils = load_utils(args.utils_path)
    tournament = json.loads(args.json_draw.read_text(encoding='utf-8'))
    surface = tournament["surface"]

    id_to_name = {
        m[side]["id"]: m[side]["name"]
        for m in tournament["matches"]
        for side in ("player1", "player2")
        if m[side]["id"] is not None
    }

    global_df, surface_dfs = utils.get_latest_features_by_surface(
        args.parquet, args.cutoff
    )
    model = utils.load_trained_model(args.model)

    # Build initial bracket (1st Round pairs)
    first_round = sorted(
        [m for m in tournament["matches"] if m["round"] == "1st Round"],
        key=lambda m: m["match_id"]
    )
    bracket_init = [
        (m["player1"]["id"], m["player2"]["id"]) for m in first_round
    ]

    # Safety check
    if len(bracket_init) == 0:
        raise ValueError("Bracket is empty — check the input JSON draw file.")

    # Prepare counters
    start_idx = args.job_index * args.runs_per_job
    end_idx = start_idx + args.runs_per_job
    champion_counts = {}
    qf_counts = {}
    sf_counts = {}

    # Run simulations
    for i in range(start_idx, end_idx):
        random.seed(i)
        champ, (_, _), qfs, sfs = simulate_tournament(
            bracket_init, surface, model, global_df, surface_dfs, utils
        )

        champion_counts[champ] = champion_counts.get(champ, 0) + 1
        for pid in qfs:
            qf_counts[pid] = qf_counts.get(pid, 0) + 1
        for pid in sfs:
            sf_counts[pid] = sf_counts.get(pid, 0) + 1

        # Print progress every 50 simulations
        if (i - start_idx + 1) % 50 == 0:
            print(f"[Job {args.job_index}] Simulations completed: {i - start_idx + 1}/{args.runs_per_job}")

    # Build stats with names
    stats = []
    for pid, wins in champion_counts.items():
        stats.append({
            "id": pid,
            "name": id_to_name.get(pid, "Unknown"),
            "wins": wins,
            "semifinals": sf_counts.get(pid, 0),
            "quarterfinals": qf_counts.get(pid, 0)
        })

    # Write partial results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "job_index": args.job_index,
        "runs": args.runs_per_job,
        "stats": stats
    }
    out_path = args.output_dir / f"partial_{args.job_index}.json"
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    elapsed_total = time.perf_counter() - t0
    duration = time.perf_counter() - start

    print(f"Wrote partial results to {out_path}")
    print(f"Total elapsed time: {elapsed_total:.2f} seconds")
    print(f"Simulations only: {duration:.2f} seconds")

    # Log simulation duration for scaling analysis
    with open(args.output_dir / "times.csv", "a") as f:
        f.write(f"{args.job_index},{args.runs_per_job},{duration:.2f}\n")

if __name__ == "__main__":
    main()
