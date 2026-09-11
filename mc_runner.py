#!/usr/bin/env python3
"""Parallel runner for the FGO / BLS / EKF Monte Carlos.

Runs the same per-seed functions as mc_fgo.py / mc_bls.py / mc_ekf.py, but pools
seeds across cores and writes the identical CSVs. Truth is propagated once per
config in the parent and inherited by the workers via fork.

    python3 mc_runner.py --seeds 200 --workers 16
    python3 mc_runner.py --seeds 50 --estimators fgo bls --configs deltaRIC0.5
"""

import argparse
import multiprocessing as mp
import os
import time

import matplotlib
matplotlib.use('Agg')          # workers have no display; must precede pyplot import

import numpy as np
import pandas as pd

import mc_fgo, mc_bls, mc_ekf
from mc_fgo import CONFIG_DEFS, propagate_truth, build_summary, print_summary
from fgo_pipeline import load_config_parameters

EST = {
    'fgo': (mc_fgo, mc_fgo.run_fgo_seed, ['FGO-B', 'FGO-G'],
            'report_mc_fgo.csv', 'report_mc_fgo_summary.csv',
            'report_mc_fgo_initial_guesses.csv'),
    'bls': (mc_bls, mc_bls.run_bls_seed, ['BLS-B', 'BLS-G'],
            'report_mc_bls.csv', 'report_mc_bls_summary.csv',
            'report_mc_bls_initial_guesses.csv'),
    'ekf': (mc_ekf, mc_ekf.run_ekf_seed, ['EKF-B', 'EKF-G'],
            'report_mc_ekf.csv', 'report_mc_ekf_summary.csv',
            'report_mc_ekf_initial_guesses.csv'),
}

TRUTH, PARAMS, STATIONS = {}, {}, {}


def run_one(job):
    """One (estimator, config, mode, seed) solve. Returns the row and its guess row."""
    est, tag, mode, seed = job
    mod, fn = EST[est][0], EST[est][1]
    mod.INITIAL_GUESSES.clear()
    S, T, dt, dv_ric, dv_eci, man_state, t_star = TRUTH[tag]
    t0 = time.perf_counter()
    r = fn(seed, S, T, dt, STATIONS[tag], PARAMS[tag], dv_ric, dv_eci,
           man_state, t_star, mode, tag)
    r.update({'config': tag, 'mode': mode})
    guess = mod.INITIAL_GUESSES[-1] if mod.INITIAL_GUESSES else None
    return est, r, guess, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--configs', nargs='*', default=None, choices=list(CONFIG_DEFS))
    ap.add_argument('--estimators', nargs='*', default=['fgo', 'bls', 'ekf'],
                    choices=list(EST))
    ap.add_argument('--modes', nargs='*', default=None,
                    help='mode suffixes to run, e.g. B G (default: both)')
    ap.add_argument('--no-plot', action='store_true',
                    help='disable per-seed plots (much faster, far less disk)')
    a = ap.parse_args()

    tags = a.configs or list(CONFIG_DEFS)
    if a.no_plot:
        mc_fgo.ENABLE_PLOTTING = mc_bls.ENABLE_PLOTTING = mc_ekf.ENABLE_PLOTTING = False

    print('=' * 78)
    print(f'Monte Carlo runner | seeds 1..{a.seeds} | workers {a.workers}')
    print(f'Estimators: {", ".join(a.estimators)}')
    print(f'Configs:    {", ".join(tags)}')
    print(f'Plots:      {"off" if a.no_plot else "on"}')
    print('=' * 78)

    for tag in tags:                       # truth once per config, in the parent
        path = CONFIG_DEFS[tag]
        TRUTH[tag] = propagate_truth(path, tag)
        cp, gs = load_config_parameters(path)
        STATIONS[tag] = gs
        PARAMS[tag] = {
            'q_pos_ric': np.array(cp['process_noise_pos'], dtype=float),
            'q_vel_ric': np.array(cp['process_noise_vel'], dtype=float),
            'use_range': cp['use_range'],
            'measurement_noise_deg': cp['measurement_noise_deg'],
            'range_noise_m': cp['range_noise_m'],
            'initial_pos_error': cp['initial_pos_error'],
            'initial_vel_error': cp['initial_vel_error'],
            'dv_initial_error': cp['dv_initial_error'],
            't_star_initial_error': cp['t_star_initial_error'],
            'epsilon': cp['epsilon'],
            'max_iterations': cp['max_iterations'],
            'gmst0': cp['gmst0'],
        }
        print(f'  truth {tag}: N = {len(TRUTH[tag][0])}, '
              f'max_iterations = {cp["max_iterations"]}', flush=True)

    jobs = []
    for est in a.estimators:
        for mode in EST[est][2]:
            if a.modes and mode.split('-')[1] not in a.modes:
                continue
            for tag in tags:
                jobs += [(est, tag, mode, s) for s in range(1, a.seeds + 1)]
    print(f'\n{len(jobs)} runs\n', flush=True)

    mp.set_start_method('fork', force=True)
    rows = {e: [] for e in a.estimators}
    guesses = {e: [] for e in a.estimators}
    t0 = time.perf_counter()
    with mp.Pool(a.workers) as pool:
        for i, (est, r, g, wall) in enumerate(
                pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows[est].append(r)
            if g is not None:
                guesses[est].append(g)
            el = time.perf_counter() - t0
            print(f'[{i:5d}/{len(jobs)}] {est} {r["config"]:>17s} {r["mode"]:6s} '
                  f's{r["seed"]:<4d} pos={r["pos_rms"]:10.2f}  {wall:6.1f}s  '
                  f'elapsed {el/60:6.1f}m  ETA {el/i*(len(jobs)-i)/60:6.1f}m', flush=True)

    for est in a.estimators:
        _, _, _, out, summ, gpath = EST[est]
        df = pd.DataFrame(rows[est]).sort_values(['config', 'mode', 'seed'])
        df.to_csv(out, index=False)
        sdf = build_summary(df)
        sdf.to_csv(summ, index=False)
        if guesses[est]:
            pd.DataFrame(guesses[est]).to_csv(gpath, index=False)
        print(f'\n=== {est.upper()} === {out} ({len(df)} rows), {summ}')
        print_summary(sdf)
    print(f'\ntotal wall {(time.perf_counter()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
