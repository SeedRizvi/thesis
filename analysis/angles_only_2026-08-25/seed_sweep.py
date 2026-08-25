#!/usr/bin/env python3
"""
Reproduce the ANGLES_ONLY_FGO.md headline table in the CURRENT code:
10 seeds x {10", 2"}, angles-only, FGO-G, with prior. Identify which seeds
actually fail now, and log solver-health metrics for each.
"""
import os, sys, time, json
import numpy as np
import numpy.linalg as la

sys.path.insert(0, '/home/z5363026/thesis')
os.chdir('/home/z5363026/thesis')

from fgo_pipeline import load_config_parameters, simulate_measurements
from Orbit_FGO import eci_to_ric
from Orbit_EKF import build_P0
import mc_fgo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier0_diag import InstrumentedFGO, build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"
TAG = "deltaRIC0.5"
OUT = os.path.dirname(os.path.abspath(__file__))
ARCSEC = 1.0 / 3600.0
SEEDS = list(range(1, 11))


def main():
    truth_states, times, dt, delta_v_ric, delta_v_eci, manoeuvre_state, t_star_true = \
        mc_fgo.propagate_truth(CONFIG, TAG)
    cp, ground_stations = load_config_parameters(CONFIG)

    rows = []
    for noise_arcsec in (10.0, 2.0):
        params = {
            'q_pos_ric': np.array(cp['process_noise_pos'], float),
            'q_vel_ric': np.array(cp['process_noise_vel'], float),
            'use_range': False,
            'measurement_noise_deg': noise_arcsec * ARCSEC,
            'range_noise_m': cp['range_noise_m'],
            'initial_pos_error': cp['initial_pos_error'],
            'initial_vel_error': cp['initial_vel_error'],
            'dv_initial_error': cp['dv_initial_error'],
            't_star_initial_error': cp['t_star_initial_error'],
            'epsilon': cp['epsilon'],
            'max_iterations': cp['max_iterations'],
        }
        for seed in SEEDS:
            fgo, ts_err0, pos_err0 = build_seed(
                seed, truth_states, times, dt, ground_stations, params,
                delta_v_eci, manoeuvre_state, t_star_true, InstrumentedFGO)
            t0 = time.perf_counter()
            log, diag, term = fgo.opt_logged(max_iters=params['max_iterations'])
            rt = time.perf_counter() - t0

            errors = fgo.states - truth_states
            pos_rms = float(np.sqrt(np.mean(la.norm(errors[:, :3], axis=1) ** 2)))
            dv_est_ric = eci_to_ric(fgo.man_params[0:3], manoeuvre_state)

            last = log[-1]
            row = {
                'noise_arcsec': noise_arcsec, 'seed': seed,
                'pos_rms': pos_rms,
                'dv_err': float(la.norm(dv_est_ric - delta_v_ric)),
                'tstar_err': float(fgo.man_params[3] - t_star_true),
                'tstar_err0': float(ts_err0), 'pos_err0': float(pos_err0),
                'termination': term, 'n_iters': len(log), 'runtime': rt,
                'cost_dyn': last['cost_dyn'], 'cost_meas': last['cost_meas'],
                'cost_prior': last['cost_prior'],
                'lambda_max': max(r['lambda_after'] for r in log),
                'tstar_diag': diag['tstar'][1],
                'pos_diag_med': diag['pos'][1], 'vel_diag_med': diag['vel'][1],
                'dv_diag_med': diag['dv'][1],
                'median_scale': float(np.median([r['best_scale'] for r in log])),
                'n_scale_lt_quarter': int(sum(1 for r in log
                                              if 0 < r['best_scale'] < 0.25)),
                'n_rejected': int(sum(1 for r in log if r['best_scale'] == 0)),
            }
            rows.append(row)
            print(f"{noise_arcsec:>4.0f}\" seed {seed:>2}: pos_rms={pos_rms:9.2f} m  "
                  f"t*err={row['tstar_err']:+8.2f}s  dyn={row['cost_dyn']:.2e}  "
                  f"meas={row['cost_meas']:.2e}  {term:<12} "
                  f"iters={len(log):>3}  medscale={row['median_scale']:.4f}  "
                  f"({rt:.0f}s)", flush=True)

    with open(f"{OUT}/seed_sweep.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    # paired comparison
    by = {(r['noise_arcsec'], r['seed']): r for r in rows}
    print("\n" + "=" * 92)
    print("PAIRED 10\" -> 2\"  (angles-only, FGO-G, with prior)")
    print("=" * 92)
    print(f"{'seed':>4} {'10\" rms':>10} {'2\" rms':>10} {'change':>9} "
          f"{'2\" term':>13} {'2\" it':>6} {'2\" dyn':>10} {'2\" t*err':>9}")
    changes = []
    for s in SEEDS:
        a, b = by[(10.0, s)], by[(2.0, s)]
        ch = 100.0 * (b['pos_rms'] - a['pos_rms']) / a['pos_rms']
        changes.append(ch)
        flag = "  <-- FAIL" if ch > 50 else ""
        print(f"{s:>4} {a['pos_rms']:>10.2f} {b['pos_rms']:>10.2f} {ch:>+8.1f}% "
              f"{b['termination']:>13} {b['n_iters']:>6} {b['cost_dyn']:>10.2e} "
              f"{b['tstar_err']:>+9.2f}{flag}")
    r10 = [by[(10.0, s)]['pos_rms'] for s in SEEDS]
    r2 = [by[(2.0, s)]['pos_rms'] for s in SEEDS]
    print(f"\n  10\": median {np.median(r10):8.1f} m   mean {np.mean(r10):8.1f} m")
    print(f"   2\": median {np.median(r2):8.1f} m   mean {np.mean(r2):8.1f} m")
    print(f"  mean paired change {np.mean(changes):+.1f}%   "
          f"median {np.median(changes):+.1f}%   "
          f"failures>50%: {sum(1 for c in changes if c > 50)}/10")


if __name__ == '__main__':
    main()
