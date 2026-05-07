#!/usr/bin/env python3
"""
Monte Carlo EKF evaluation across manoeuvre configurations.

Runs each config with two modes:
  1. EKF-B: base EKF without Gaussian manoeuvre estimation
  2. EKF-G: Gaussian-augmented EKF estimating delta-v and t*

Per-seed results are saved to report_mc_ekf.csv.
A summary is printed and saved to report_mc_ekf_summary.csv.

Usage:
    python3 mc_ekf.py
    python3 mc_ekf.py --seeds 10
    python3 mc_ekf.py --configs deltaRIC1 deltaI0.2
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from fgo_pipeline import (load_config_parameters,
                          simulate_measurements, plot_fgo_results)
from Orbit_FGO import eci_to_ric, ric_to_eci
from Orbit_EKF import SatelliteOrbitEKF
from mc_fgo import (propagate_truth, build_summary, print_summary,
                    CONFIG_DEFS, DEFAULT_N_SEEDS)


# Defaults
INITIAL_GUESSES = []
ENABLE_PLOTTING = True
DEFAULT_OUTPUT = "report_mc_ekf.csv"
DEFAULT_SUMMARY = "report_mc_ekf_summary.csv"

MODES = ["EKF-B", "EKF-G"]


# ---------------------------------------------------------------------------
# Single EKF solve
# ---------------------------------------------------------------------------

def run_ekf_seed(seed, truth_states, times, dt, ground_stations, params,
                 delta_v_ric, delta_v_eci, manoeuvre_state, t_star_true,
                 mode, config_tag):
    """
    Run one EKF solve and return a comprehensive metrics dict.

    mode: 'EKF-B' (no manoeuvre estimation) or 'EKF-G' (Gaussian estimation)
    """
    np.random.seed(seed)

    measurements, R = simulate_measurements(
        truth_states, times, ground_stations,
        params['measurement_noise_deg'],
        use_range=True,
        range_noise_m=params['range_noise_m'],
    )

    x0 = truth_states[0].copy()
    x0_err = np.zeros(6)
    x0_err[:3] = np.random.normal(0, params['initial_pos_error'], 3)
    x0_err[3:] = np.random.normal(0, params['initial_vel_error'], 3)
    x0 += x0_err

    manoeuvres = None
    dv_guess_err_ric = None
    t_star_guess_err = None
    if mode == 'EKF-G' and delta_v_eci is not None:
        dv_noise = np.random.normal(0, params['dv_initial_error'], 3)
        dv_guess = delta_v_eci + dv_noise
        t_star_guess_err = np.random.normal(0, params['t_star_initial_error'])
        t_star_guess = t_star_true + t_star_guess_err
        manoeuvres = [{'delta_v': dv_guess, 't_star': t_star_guess}]
        dv_guess_err_ric = eci_to_ric(dv_noise, manoeuvre_state)

    ekf = SatelliteOrbitEKF(
        measurements, R,
        params['q_pos_ric'], params['q_vel_ric'],
        ground_stations, dt, x0=x0,
        use_range=True, manoeuvres=manoeuvres,
        epsilon=params['epsilon'],
    )

    # Log initial guesses
    guess_row = {
        'config': config_tag,
        'mode': mode,
        'seed': seed,
        'x0_pos_err_x': float(x0_err[0]),
        'x0_pos_err_y': float(x0_err[1]),
        'x0_pos_err_z': float(x0_err[2]),
        'x0_pos_err_norm': float(np.linalg.norm(x0_err[:3])),
        'x0_vel_err_x': float(x0_err[3]),
        'x0_vel_err_y': float(x0_err[4]),
        'x0_vel_err_z': float(x0_err[5]),
        'x0_vel_err_norm': float(np.linalg.norm(x0_err[3:])),
    }
    if dv_guess_err_ric is not None:
        guess_row['dv_guess_err_R'] = float(dv_guess_err_ric[0])
        guess_row['dv_guess_err_I'] = float(dv_guess_err_ric[1])
        guess_row['dv_guess_err_C'] = float(dv_guess_err_ric[2])
        guess_row['dv_guess_err_norm'] = float(np.linalg.norm(dv_guess_err_ric))
        guess_row['t_star_guess_err'] = float(t_star_guess_err)
    else:
        guess_row['dv_guess_err_R'] = None
        guess_row['dv_guess_err_I'] = None
        guess_row['dv_guess_err_C'] = None
        guess_row['dv_guess_err_norm'] = None
        guess_row['t_star_guess_err'] = None
    INITIAL_GUESSES.append(guess_row)

    t0 = time.perf_counter()
    ekf.run(verbose=False)
    runtime = time.perf_counter() - t0

    # Position / velocity errors
    errors = ekf.states - truth_states
    pos_errors = np.linalg.norm(errors[:, :3], axis=1)
    vel_errors = np.linalg.norm(errors[:, 3:], axis=1)

    result = {
        'seed':      seed,
        'pos_rms':   float(np.sqrt(np.mean(pos_errors ** 2))),
        'pos_mean':  float(np.mean(pos_errors)),
        'pos_std':   float(np.std(pos_errors)),
        'pos_max':   float(np.max(pos_errors)),
        'vel_rms':   float(np.sqrt(np.mean(vel_errors ** 2))),
        'vel_mean':  float(np.mean(vel_errors)),
        'vel_std':   float(np.std(vel_errors)),
        'vel_max':   float(np.max(vel_errors)),
        'runtime_s': float(runtime),
    }

    # Delta-v and t* errors (only for EKF-G with manoeuvres)
    if mode == 'EKF-G' and manoeuvres is not None and delta_v_ric is not None:
        dv_est_eci = ekf.man_params[0:3]
        dv_est_ric = eci_to_ric(dv_est_eci, manoeuvre_state)
        dv_err_ric = dv_est_ric - delta_v_ric

        result['dv_err_R'] = float(dv_err_ric[0])
        result['dv_err_I'] = float(dv_err_ric[1])
        result['dv_err_C'] = float(dv_err_ric[2])
        result['dv_err_norm'] = float(np.linalg.norm(dv_err_ric))
        result['t_star_error'] = float(ekf.man_params[3] - t_star_true)
    else:
        result['dv_err_R'] = None
        result['dv_err_I'] = None
        result['dv_err_C'] = None
        result['dv_err_norm'] = None
        result['t_star_error'] = None

    # Plotting
    plot_results = {
        'fgo': ekf,
        'truth': truth_states,
        'estimated': ekf.states,
        'measurements': measurements,
        'delta_v_ric': delta_v_ric,
        'errors': errors,
        'pos_errors': pos_errors,
        'vel_errors': vel_errors,
        'times': times,
        'dt': dt,
        'ground_stations': ground_stations,
        'use_range': True,
        't_star_true': t_star_true,
    }
    if mode == 'EKF-G' and manoeuvres is not None and delta_v_ric is not None:
        plot_results['dv_true'] = delta_v_ric
        plot_results['dv_estimated'] = dv_est_ric
        plot_results['dv_error'] = dv_est_ric - delta_v_ric
        plot_results['manoeuvre_state'] = manoeuvre_state
        if t_star_true is not None:
            plot_results['t_star_true'] = t_star_true
            plot_results['t_star_estimated'] = float(ekf.man_params[3])
            plot_results['t_star_error'] = float(ekf.man_params[3] - t_star_true)

    if ENABLE_PLOTTING:
        cfg_short = config_tag.replace('delta', '')
        save_path = f"plots/report_mc_{mode}_{cfg_short}_N{seed}.png"
        os.makedirs("plots", exist_ok=True)
        plot_fgo_results(plot_results, save_path=save_path)

    return result


# ---------------------------------------------------------------------------
# Run all seeds for one (config, mode) combination
# ---------------------------------------------------------------------------

def run_config_mode(config_tag, mode, truth_data, ground_stations, params,
                    seeds):
    """Run all seeds for a single (config, mode) pair."""
    (truth_states, times, dt, dv_ric, dv_eci,
     man_state, t_star) = truth_data
    rows = []

    for seed in seeds:
        try:
            r = run_ekf_seed(
                seed=seed,
                truth_states=truth_states, times=times, dt=dt,
                ground_stations=ground_stations,
                params=params,
                delta_v_ric=dv_ric, delta_v_eci=dv_eci,
                manoeuvre_state=man_state, t_star_true=t_star,
                mode=mode, config_tag=config_tag,
            )
            r.update({'config': config_tag, 'mode': mode})
            rows.append(r)
        except Exception as e:
            print(f"    FAILED seed={seed}: {e}")

    # Live progress
    if rows:
        pr = [r['pos_rms'] for r in rows]
        vr = [r['vel_rms'] for r in rows]
        rt = [r['runtime_s'] for r in rows]
        dv = [r['dv_err_norm'] for r in rows if r['dv_err_norm'] is not None]
        ts = [r['t_star_error'] for r in rows if r['t_star_error'] is not None]
        ts_arr = np.array(ts) if ts else np.array([])

        line = (f"  {config_tag:>12s} | {mode:5s} | "
                f"pos_rms {np.mean(pr):.1f}±{np.std(pr):.1f} m  "
                f"vel_rms {np.mean(vr):.4f}±{np.std(vr):.4f} m/s  "
                f"time {np.mean(rt):.1f}s")
        if dv:
            line += f"  |dv_err| {np.mean(dv):.4f}±{np.std(dv):.4f} m/s"
        if ts:
            line += (f"  t*_err mean={np.mean(ts_arr):.2f} "
                     f"std={np.std(ts_arr):.2f} "
                     f"rms={np.sqrt(np.mean(ts_arr**2)):.2f} s")
        print(line)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo EKF evaluation across manoeuvre configs')
    parser.add_argument('--seeds', type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Per-seed results CSV')
    parser.add_argument('--summary', default=DEFAULT_SUMMARY,
                        help='Aggregate summary CSV')
    parser.add_argument('--configs', nargs='*', default=None,
                        choices=list(CONFIG_DEFS.keys()),
                        help='Run only these configs (default: all)')
    parser.add_argument('--modes', nargs='*', default=None,
                        choices=MODES,
                        help='Run only these modes (default: both)')
    args = parser.parse_args()

    config_tags = args.configs or list(CONFIG_DEFS.keys())
    modes = args.modes or MODES

    print("=" * 80)
    print("Monte Carlo EKF Evaluation")
    print(f"Seeds:   1..{args.seeds}")
    print(f"Configs: {', '.join(config_tags)}")
    print(f"Modes:   {', '.join(modes)}")
    print(f"Output:  {args.output}")
    print(f"Summary: {args.summary}")
    print("=" * 80)

    seeds = list(range(1, args.seeds + 1))
    all_rows = []

    for tag in config_tags:
        config_path = CONFIG_DEFS[tag]
        print(f"\n{'='*60}")
        print(f"Config: {tag}  ({config_path})")
        print(f"{'='*60}")

        # Propagate truth once per config
        print("  Propagating truth trajectory...")
        truth_data = propagate_truth(config_path, tag)

        config_params, ground_stations = load_config_parameters(config_path)
        params = {
            'q_pos_ric':             np.array(config_params['process_noise_pos'],
                                              dtype=float),
            'q_vel_ric':             np.array(config_params['process_noise_vel'],
                                              dtype=float),
            'measurement_noise_deg': config_params['measurement_noise_deg'],
            'range_noise_m':         config_params['range_noise_m'],
            'initial_pos_error':     config_params['initial_pos_error'],
            'initial_vel_error':     config_params['initial_vel_error'],
            'dv_initial_error':      config_params.get('dv_initial_error', 0.1),
            't_star_initial_error':  config_params.get('t_star_initial_error',
                                                       120.0),
            'epsilon':               config_params.get('epsilon', 0.5),
            'max_iterations':        config_params['max_iterations'],
        }

        for mode in modes:
            rows = run_config_mode(
                config_tag=tag,
                mode=mode,
                truth_data=truth_data,
                ground_stations=ground_stations,
                params=params,
                seeds=seeds,
            )
            all_rows.extend(rows)

    # Save per-seed CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nPer-seed results saved to: {args.output}  ({len(df)} rows)")

    # Save initial guesses CSV
    if INITIAL_GUESSES:
        df_guesses = pd.DataFrame(INITIAL_GUESSES)
        guesses_path = "report_mc_ekf_initial_guesses.csv"
        df_guesses.to_csv(guesses_path, index=False)
        print(f"Initial guesses saved to: {guesses_path}  "
              f"({len(df_guesses)} rows)")

    # Build and save summary
    if not df.empty:
        summary = build_summary(df)
        summary.to_csv(args.summary, index=False)
        print(f"Summary saved to:         {args.summary}  "
              f"({len(summary)} rows)")
        print_summary(summary)


if __name__ == '__main__':
    main()
