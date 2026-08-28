#!/usr/bin/env python3
"""
Monte Carlo FGO evaluation across manoeuvre configurations.

Runs each config with two modes:
  1. FGO-B: base FGO without Gaussian manoeuvre estimation
  2. FGO-G: Gaussian-augmented FGO estimating delta-v and t*

Per-seed results are saved to report_mc_fgo.csv.
A summary is printed and saved to report_mc_fgo_summary.csv.

Usage:
    python3 mc_fgo.py
    python3 mc_fgo.py --seeds 10
    python3 mc_fgo.py --configs deltaRIC1 deltaI0.2
"""

import argparse
import os
import time
import re
import numpy as np
import pandas as pd
import yaml

from fgo_pipeline import (load_propagator_output, load_config_parameters,
                          simulate_measurements, plot_fgo_results)
from Orbit_FGO import SatelliteOrbitFGO, eci_to_ric, ric_to_eci
from Orbit_EKF import build_P0
from propagator import OrbitPropagator


# Defaults
INITIAL_GUESSES = []  # populated by run_fgo_seed
ENABLE_PLOTTING = True
DEFAULT_N_SEEDS = 5
DEFAULT_OUTPUT = "report_mc_fgo.csv"
DEFAULT_SUMMARY = "report_mc_fgo_summary.csv"

# Config definitions: tag -> path
CONFIG_DEFS = {
    "deltaRIC0":   "configs/config_geo_one_rev_deltaRIC0.yml",
    "deltaRIC1":   "configs/config_geo_one_rev_deltaRIC1.yml",
    "deltaRIC0.5": "configs/config_geo_one_rev_deltaRIC0.5.yml",
    "deltaRIC0.5_short":   "configs/config_geo_short_arc_deltaRIC0.5.yml",
    "deltaI0.2":   "configs/config_geo_one_rev_deltaI0.2.yml",
    "deltaC0.2":   "configs/config_geo_one_rev_deltaC0.2.yml",
}

MODES = ["FGO-B", "FGO-G"]


# ---------------------------------------------------------------------------
# Truth propagation (done once per config)
# ---------------------------------------------------------------------------

def propagate_truth(config_path, tag):
    """
    Propagate truth trajectory (pre + post manoeuvre).

    Returns
    -------
    truth_states   : ndarray (N, 6)
    times          : ndarray (N,)
    dt             : float
    delta_v_ric    : ndarray (3,) or None
    delta_v_eci    : ndarray (3,) or None
    manoeuvre_state : ndarray (6,) or None
    t_star_true    : float or None
    """
    config_params, _ = load_config_parameters(config_path)
    delta_v_ric = config_params['delta_v_ric']
    duration = config_params['pm_duration']
    prop = OrbitPropagator("orbDetHOUSE")

    if delta_v_ric is not None:
        delta_v_ric = np.array(delta_v_ric, dtype=float)

        pre_csv = prop.propagate(config_path,
                                 output_file=f"mc_fgo_truth_pre_{tag}.csv")
        df_pre = pd.read_csv(pre_csv)

        manoeuvre_state = df_pre[['x', 'y', 'z', 'vx', 'vy', 'vz']] \
                              .iloc[-1].values
        t_star_true = float(df_pre['tSec'].iloc[-1])
        delta_v_eci = ric_to_eci(delta_v_ric, manoeuvre_state)

        new_state = [
            float(df_pre.iloc[-1]['x']),
            float(df_pre.iloc[-1]['y']),
            float(df_pre.iloc[-1]['z']),
            float(df_pre.iloc[-1]['vx']) + delta_v_eci[0],
            float(df_pre.iloc[-1]['vy']) + delta_v_eci[1],
            float(df_pre.iloc[-1]['vz']) + delta_v_eci[2],
        ]

        with open(config_path, 'r') as f:
            raw = f.read()

        with open(config_path, 'r') as f:
            config_post = yaml.safe_load(f)
        # Start the post arc where the pre arc actually ended, which is the last
        # dt sample at or before MJD_end, not MJD_end itself.
        mjd_start_new = (config_post['scenario_parameters']['MJD_start']
                         + t_star_true / 86400.0)
        mjd_end_new = mjd_start_new + duration

        state_str = '[' + ', '.join(f'{v}' for v in new_state) + ']'
        raw = re.sub(r'(initial_state:\s*)\[.*?\]', rf'\1{state_str}',
                     raw, flags=re.DOTALL)
        raw = re.sub(r'(MJD_start:\s*)[\d.]+', rf'\g<1>{mjd_start_new}',
                     raw)
        raw = re.sub(r'(MJD_end:\s*)[\d.]+', rf'\g<1>{mjd_end_new}',
                     raw)

        tmp_cfg = f'/tmp/mc_fgo_post_{tag}.yml'
        with open(tmp_cfg, 'w') as f:
            f.write(raw)

        post_csv = prop.propagate(tmp_cfg,
                                  output_file=f"mc_fgo_truth_post_{tag}.csv")
        os.remove(tmp_cfg)

        df_post = pd.read_csv(post_csv)
        df_post['tSec'] = df_post['tSec'] + df_pre['tSec'].iloc[-1]
        df_combined = pd.concat([df_pre, df_post.iloc[1:]], ignore_index=True)
        csv_path = os.path.abspath(f"out/mc_fgo_truth_{tag}.csv")
        df_combined.to_csv(csv_path, index=False)
    else:
        csv_path = prop.propagate(config_path,
                                  output_file=f"mc_fgo_truth_{tag}.csv")
        delta_v_ric = None
        delta_v_eci = None
        manoeuvre_state = None
        t_star_true = None

    truth_states, times, dt = load_propagator_output(csv_path)
    return (truth_states, times, dt, delta_v_ric, delta_v_eci,
            manoeuvre_state, t_star_true)


# ---------------------------------------------------------------------------
# Single FGO solve
# ---------------------------------------------------------------------------

def run_fgo_seed(seed, truth_states, times, dt, ground_stations, params,
                 delta_v_ric, delta_v_eci, manoeuvre_state, t_star_true,
                 mode, config_tag):
    """
    Run one FGO solve and return a comprehensive metrics dict.

    mode: 'FGO-B' (no manoeuvre estimation) or 'FGO-G' (Gaussian estimation)
    """
    np.random.seed(seed)

    measurements, R = simulate_measurements(
        truth_states, times, ground_stations,
        params['measurement_noise_deg'],
        use_range=params['use_range'],
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
    if mode == 'FGO-G' and delta_v_eci is not None:
        dv_noise = np.random.normal(0, params['dv_initial_error'], 3)
        dv_guess = delta_v_eci + dv_noise
        t_star_guess_err = np.random.normal(0, params['t_star_initial_error'])
        t_star_guess = t_star_true + t_star_guess_err
        manoeuvres = [{'delta_v': dv_guess, 't_star': t_star_guess}]
        dv_guess_err_ric = eci_to_ric(dv_noise, manoeuvre_state)

    P0 = build_P0(
        0 if manoeuvres is None else len(manoeuvres),
        params['initial_pos_error'], params['initial_vel_error'],
        params['dv_initial_error'], params['t_star_initial_error'],
    )

    fgo = SatelliteOrbitFGO(
        measurements, R,
        params['q_pos_ric'], params['q_vel_ric'],
        ground_stations, dt, x0=x0, P0=P0,
        use_range=params['use_range'], manoeuvres=manoeuvres,
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
    fgo.opt(max_iters=params['max_iterations'], verbose=False)
    runtime = time.perf_counter() - t0

    # Position / velocity errors
    errors = fgo.states - truth_states
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

    # Delta-v and t* errors (only for FGO-G with manoeuvres)
    if mode == 'FGO-G' and manoeuvres is not None and delta_v_ric is not None:
        dv_est_eci = fgo.man_params[0:3]
        dv_est_ric = eci_to_ric(dv_est_eci, manoeuvre_state)
        dv_err_ric = dv_est_ric - delta_v_ric

        result['dv_err_R'] = float(dv_err_ric[0])
        result['dv_err_I'] = float(dv_err_ric[1])
        result['dv_err_C'] = float(dv_err_ric[2])
        result['dv_err_norm'] = float(np.linalg.norm(dv_err_ric))
        result['t_star_error'] = float(fgo.man_params[3] - t_star_true)
    else:
        result['dv_err_R'] = None
        result['dv_err_I'] = None
        result['dv_err_C'] = None
        result['dv_err_norm'] = None
        result['t_star_error'] = None

    # Plotting
    plot_results = {
        'fgo': fgo,
        'truth': truth_states,
        'estimated': fgo.states,
        'measurements': measurements,
        'delta_v_ric': delta_v_ric,
        'errors': errors,
        'pos_errors': pos_errors,
        'vel_errors': vel_errors,
        'times': times,
        'dt': dt,
        'ground_stations': ground_stations,
        'use_range': params['use_range'],
        't_star_true': t_star_true,
    }
    if mode == 'FGO-G' and manoeuvres is not None and delta_v_ric is not None:
        plot_results['dv_true'] = delta_v_ric
        plot_results['dv_estimated'] = dv_est_ric
        plot_results['dv_error'] = dv_est_ric - delta_v_ric
        plot_results['manoeuvre_state'] = manoeuvre_state
        if t_star_true is not None:
            plot_results['t_star_true'] = t_star_true
            plot_results['t_star_estimated'] = float(fgo.man_params[3])
            plot_results['t_star_error'] = float(fgo.man_params[3] - t_star_true)

    # Strip 'delta' prefix from config tag for shorter filenames
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
            r = run_fgo_seed(
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
# Summary
# ---------------------------------------------------------------------------

def build_summary(df):
    """Build per-(config, mode) aggregate summary DataFrame."""
    summary_rows = []

    for (cfg, mode), grp in df.groupby(['config', 'mode'], sort=False):
        row = {
            'config':          cfg,
            'mode':            mode,
            'n_seeds':         len(grp),
            'pos_rms_mean':    grp['pos_rms'].mean(),
            'pos_rms_std':     grp['pos_rms'].std(),
            'vel_rms_mean':    grp['vel_rms'].mean(),
            'vel_rms_std':     grp['vel_rms'].std(),
            'runtime_mean':    grp['runtime_s'].mean(),
            'runtime_std':     grp['runtime_s'].std(),
        }

        dv_vals = grp['dv_err_norm'].dropna()
        ts_vals = grp['t_star_error'].dropna()

        if len(dv_vals) > 0:
            row['dv_err_mean'] = dv_vals.mean()
            row['dv_err_std'] = dv_vals.std()
            row['dv_err_R_mean'] = grp['dv_err_R'].dropna().mean()
            row['dv_err_I_mean'] = grp['dv_err_I'].dropna().mean()
            row['dv_err_C_mean'] = grp['dv_err_C'].dropna().mean()
        else:
            row['dv_err_mean'] = None
            row['dv_err_std'] = None
            row['dv_err_R_mean'] = None
            row['dv_err_I_mean'] = None
            row['dv_err_C_mean'] = None

        if len(ts_vals) > 0:
            row['t_star_mean'] = ts_vals.mean()
            row['t_star_std'] = ts_vals.std()
            row['t_star_rms'] = float(np.sqrt(np.mean(ts_vals ** 2)))
        else:
            row['t_star_mean'] = None
            row['t_star_std'] = None
            row['t_star_rms'] = None

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_summary(summary_df):
    """Pretty-print aggregate summary table."""
    print("\n" + "=" * 140)
    print("AGGREGATE SUMMARY (mean ± std over seeds)")
    print("=" * 140)

    print(f"\n  {'config':>12} | {'mode':>5} | {'pos_rms':>18} | "
          f"{'vel_rms':>18} | {'|dv_err|':>18} | {'t*_mean':>10} | "
          f"{'t*_std':>10} | {'t*_rms':>10} | {'time':>10}")
    print("  " + "-" * 135)

    for _, row in summary_df.iterrows():
        pos_str = (f"{row['pos_rms_mean']:.1f}±"
                   f"{row['pos_rms_std']:.1f} m")
        vel_str = (f"{row['vel_rms_mean']:.4f}±"
                   f"{row['vel_rms_std']:.4f}")

        if pd.notna(row.get('dv_err_mean')):
            dv_str = (f"{row['dv_err_mean']:.4f}±"
                      f"{row['dv_err_std']:.4f}")
        else:
            dv_str = "N/A"

        if pd.notna(row.get('t_star_mean')):
            tsm_str = f"{row['t_star_mean']:.2f}"
            tss_str = f"{row['t_star_std']:.2f}"
            tsr_str = f"{row['t_star_rms']:.2f}"
        else:
            tsm_str = tss_str = tsr_str = "N/A"

        rt_str = f"{row['runtime_mean']:.1f}s"

        print(f"  {row['config']:>12} | {row['mode']:>5} | {pos_str:>18} | "
              f"{vel_str:>18} | {dv_str:>18} | {tsm_str:>10} | "
              f"{tss_str:>10} | {tsr_str:>10} | {rt_str:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo FGO evaluation across manoeuvre configs')
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
    print("Monte Carlo FGO Evaluation")
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
            'use_range':             config_params['use_range'],
            'measurement_noise_deg': config_params['measurement_noise_deg'],
            'range_noise_m':         config_params['range_noise_m'],
            'initial_pos_error':     config_params['initial_pos_error'],
            'initial_vel_error':     config_params['initial_vel_error'],
            'dv_initial_error':      config_params['dv_initial_error'],
            't_star_initial_error':  config_params['t_star_initial_error'],
            'epsilon':               config_params['epsilon'],
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
        guesses_path = "report_mc_fgo_initial_guesses.csv"
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