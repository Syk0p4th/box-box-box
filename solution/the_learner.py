"""
the_learner.py  –  Box Box Box parameter learner
=================================================

Reverse-engineers the lap-time formula from historical race data by
searching for the parameters that reproduce the most finishing orders.

Formula (per the regulations doc):
    tire_age increments by 1 BEFORE each lap calculation.
    First lap on fresh tires is driven at age = 1.

    lap_time = base_lap_time
             + compound_offset[tire]
             + (deg_base[tire] + temp_coeff[tire] * track_temp) * tire_age

    pit stop: pit_lane_time is added to total at the end of the pit lap.

Usage
-----
    # Quick test (1000 races, ~1-2 min):
    python the_learner.py

    # More races for better accuracy:
    python the_learner.py --files 5

    # Full dataset (slow but most accurate):
    python the_learner.py --files 30
"""

import json
import glob
import argparse
import numpy as np
from scipy.optimize import differential_evolution

import os as _os

def _find_repo_root():
    """Walk up from this file until we find the directory containing data/."""
    here = _os.path.abspath(_os.path.dirname(_os.path.abspath(__file__)))
    candidate = here
    for _ in range(6):
        if _os.path.isdir(_os.path.join(candidate, 'data', 'historical_races')):
            return candidate
        candidate = _os.path.dirname(candidate)
    return _os.getcwd()

BASE = _find_repo_root()

PARAM_NAMES = [
    'soft_offset',   # compound speed offset vs base_lap_time (s)
    'medium_offset',
    'hard_offset',
    'soft_deg',      # base degradation rate (s/lap)
    'medium_deg',
    'hard_deg',
    'soft_temp',     # temperature coefficient on degradation (s/lap/°C)
    'medium_temp',
    'hard_temp',
]

BOUNDS = [
    (-5.0,  0.5),   # soft_offset   (softs are faster → negative)
    (-2.0,  2.0),   # medium_offset
    ( 0.0,  5.0),   # hard_offset   (hards are slower → positive)
    ( 0.01, 0.60),  # soft_deg
    ( 0.005,0.30),  # medium_deg
    ( 0.001,0.15),  # hard_deg
    (-0.01, 0.03),  # soft_temp
    (-0.005,0.02),  # medium_temp
    (-0.005,0.02),  # hard_temp
]


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_race(race_config, strategies, params):
    """Return predicted finishing order (list of driver_id strings)."""
    total_laps    = race_config['total_laps']
    base_lap_time = race_config['base_lap_time']
    pit_lane_time = race_config['pit_lane_time']
    track_temp    = race_config['track_temp']

    s_off, m_off, h_off = params[0], params[1], params[2]
    s_deg, m_deg, h_deg = params[3], params[4], params[5]
    s_tmp, m_tmp, h_tmp = params[6], params[7], params[8]

    offsets  = {'SOFT': s_off, 'MEDIUM': m_off, 'HARD': h_off}
    deg_base = {'SOFT': s_deg, 'MEDIUM': m_deg, 'HARD': h_deg}
    temp_co  = {'SOFT': s_tmp, 'MEDIUM': m_tmp, 'HARD': h_tmp}

    results = []
    for strat in strategies.values():
        driver_id  = strat['driver_id']
        pit_stops  = {p['lap']: p['to_tire'] for p in strat['pit_stops']}
        curr_tire  = strat['starting_tire']
        tire_age   = 0
        total_time = 0.0

        for lap in range(1, total_laps + 1):
            tire_age  += 1   # age BEFORE lap time (per regulations)
            eff_deg    = deg_base[curr_tire] + temp_co[curr_tire] * track_temp
            total_time += base_lap_time + offsets[curr_tire] + eff_deg * tire_age

            # Pit stop at END of this lap (not on final lap)
            if lap in pit_stops and lap < total_laps:
                total_time += pit_lane_time
                curr_tire   = pit_stops[lap]
                tire_age    = 0

        results.append((driver_id, total_time))

    results.sort(key=lambda x: x[1])
    return [d[0] for d in results]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_races(max_files=3):
    files = sorted(glob.glob(f'{BASE}/data/historical_races/*.json'))[:max_files]
    races = []
    for fp in files:
        with open(fp) as f:
            races.extend(json.load(f))
    print(f'Loaded {len(races):,} races from {len(files)} file(s)')
    return races


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def optimise(races, maxiter=300, popsize=20):
    best_score = [0]
    call_count = [0]

    def objective(params):
        call_count[0] += 1
        correct = sum(
            1 for race in races
            if simulate_race(race['race_config'], race['strategies'], params)
               == race['finishing_positions']
        )
        if correct > best_score[0]:
            best_score[0] = correct
            pct = correct / len(races) * 100
            fmt = ', '.join(f'{v:.5f}' for v in params)
            print(f'  iter {call_count[0]:5d}  {correct}/{len(races)} ({pct:.1f}%)  [{fmt}]',
                  flush=True)
        return -correct

    print(f'\nOptimising {len(PARAM_NAMES)} parameters over {len(races):,} races …\n')
    result = differential_evolution(
        objective,
        BOUNDS,
        maxiter=maxiter,
        popsize=popsize,
        seed=42,
        tol=1e-9,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_test_cases(params):
    inp_files = sorted(glob.glob(f'{BASE}/data/test_cases/inputs/test_*.json'))
    out_files = sorted(glob.glob(f'{BASE}/data/test_cases/expected_outputs/test_*.json'))

    if not inp_files:
        print('No test case inputs found — skipping validation.')
        return 0

    correct = 0
    skipped = 0
    failures = []
    for inp_path, out_path in zip(inp_files, out_files):
        # Skip files that failed to extract from the archive (zero bytes)
        if _os.path.getsize(inp_path) == 0 or _os.path.getsize(out_path) == 0:
            skipped += 1
            continue
        with open(inp_path) as f: tc_in  = json.load(f)
        with open(out_path) as f: tc_out = json.load(f)

        pred     = simulate_race(tc_in['race_config'], tc_in['strategies'], params)
        expected = tc_out['finishing_positions']

        if pred == expected:
            correct += 1
        else:
            failures.append((tc_in['race_id'], pred[:3], expected[:3]))

    total = len(inp_files) - skipped
    if skipped:
        print(f'(Skipped {skipped} corrupt/empty files from incomplete RAR extraction)')
    print(f'Test cases: {correct}/{total} = {correct / total * 100:.1f}%')
    if failures:
        print(f'First few failures (predicted top-3 vs expected top-3):')
        for race_id, p, e in failures[:5]:
            print(f'  {race_id}  pred={p}  exp={e}')
    return correct


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',   type=int, default=3,
                        help='Number of historical race files to use (default 3 = 3000 races)')
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--popsize', type=int, default=20)
    args = parser.parse_args()

    races  = load_races(max_files=args.files)
    result = optimise(races, maxiter=args.maxiter, popsize=args.popsize)

    print('\n=== BEST PARAMETERS ===')
    best_params = result.x
    param_dict  = {}
    for name, val in zip(PARAM_NAMES, best_params):
        print(f'  {name:<18} = {val:.6f}')
        param_dict[name] = float(val)

    train_score = int(-result.fun)
    print(f'\nTraining score: {train_score}/{len(races)} ({train_score / len(races) * 100:.1f}%)')

    print('\n=== TEST CASE VALIDATION ===')
    validate_test_cases(best_params)

    # Save for the simulator to load
    import os
    out_path = f'{BASE}/solution/learned_params.json'
    os.makedirs(f'{BASE}/solution', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(param_dict, f, indent=2)
    print(f'\nParams saved → {out_path}')

    # Also save next to this script for convenience
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, 'learned_params.json')
    with open(local_path, 'w') as f:
        json.dump(param_dict, f, indent=2)
    print(f'Params saved → {local_path}')