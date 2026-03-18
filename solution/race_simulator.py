#!/usr/bin/env python3
import json
import sys
import os

# ── Load learned parameters ───────────────────────────────────────────────────
# Run the_learner.py first to generate this file, then paste the values below.
# Or point the path to wherever learned_params.json was saved.
_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'solution', 'learned_params.json')
if os.path.exists(_params_path):
    with open(_params_path) as f:
        _p = json.load(f)
else:
    # Fallback: hardcode values from the_learner.py output here
    _p = {
        'soft_offset':   -1.5,
        'medium_offset':  0.0,
        'hard_offset':    1.5,
        'soft_deg':       0.125,
        'medium_deg':     0.072,
        'hard_deg':       0.038,
        'soft_temp':      0.003,
        'medium_temp':    0.001,
        'hard_temp':      0.001,
    }

OFFSETS  = {'SOFT': _p['soft_offset'],  'MEDIUM': _p['medium_offset'],  'HARD': _p['hard_offset']}
DEG_BASE = {'SOFT': _p['soft_deg'],     'MEDIUM': _p['medium_deg'],     'HARD': _p['hard_deg']}
TEMP_CO  = {'SOFT': _p['soft_temp'],    'MEDIUM': _p['medium_temp'],    'HARD': _p['hard_temp']}
# ─────────────────────────────────────────────────────────────────────────────


class RaceEngine:
    def __init__(self, config):
        self.total_laps    = config['total_laps']
        self.base_time     = config['base_lap_time']
        self.pit_penalty   = config['pit_lane_time']   # BUG FIX 1: was 'pit_stop_time_loss' (doesn't exist)
        self.track_temp    = config['track_temp']      # BUG FIX 2: was config.get(..., 30) — key always present

    def simulate_driver(self, strategy):
        total_race_time = 0.0

        # BUG FIX 3: data has no 'tire_plan' — build lap->new_tire from actual keys:
        #   strategy['starting_tire']  and  strategy['pit_stops'] = [{'lap': N, 'to_tire': X}]
        pit_stops = {p['lap']: p['to_tire'] for p in strategy['pit_stops']}
        curr_tire = strategy['starting_tire']
        tire_age  = 0

        for lap in range(1, self.total_laps + 1):
            tire_age += 1   # BUG FIX 4: age starts at 1, not 0 (per regulations)
                            # old code did range(laps_in_stint) → first lap was age=0

            eff_deg   = DEG_BASE[curr_tire] + TEMP_CO[curr_tire] * self.track_temp
            lap_time  = self.base_time + OFFSETS[curr_tire] + eff_deg * tire_age
            total_race_time += lap_time

            # Pit at END of this lap (never on the final lap)
            if lap in pit_stops and lap < self.total_laps:
                total_race_time += self.pit_penalty
                curr_tire = pit_stops[lap]
                tire_age  = 0

        return total_race_time


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    engine  = RaceEngine(input_data['race_config'])
    results = []

    # BUG FIX 5: strategies is a dict {pos1: {...}, pos2: {...}}
    # old code iterated the dict directly → gave string keys 'pos1', 'pos2'...
    # then crashed on driver_strat['driver_id'] because 'pos1'['driver_id'] is invalid
    for driver_strat in input_data['strategies'].values():
        race_time = engine.simulate_driver(driver_strat)
        results.append({
            'driver_id': driver_strat['driver_id'],
            'total_time': race_time
        })

    results.sort(key=lambda x: x['total_time'])

    output = {
        'race_id': input_data['race_id'],
        'finishing_positions': [d['driver_id'] for d in results]
    }
    print(json.dumps(output))


if __name__ == '__main__':
    main()