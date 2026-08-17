#!/usr/bin/env python3
"""
Integration script for running Extended Kalman Filter with orbDetHOUSE propagator.
Mirrors fgo_pipeline.py so that the same configs, measurements, and plotting code
can be used for direct FGO vs EKF comparison.
"""

import os
import numpy as np
import pandas as pd
import yaml
from Orbit_EKF import SatelliteOrbitEKF, build_P0
from Orbit_FGO import ric_to_eci, eci_to_ric
from fgo_pipeline import (
    load_propagator_output,
    load_config_parameters,
    simulate_measurements,
    plot_fgo_results,
)


def run_ekf_with_propagator(config_path,
                            use_range=None,
                            verbose=True,
                            use_gaussian_estimation=True):
    """
    Complete pipeline: propagate orbit, simulate measurements, run EKF.

    Returns the same dict structure as run_fgo_with_propagator() so that
    plot_fgo_results() works without modification.
    """

    # Load parameters from config
    config_params, config_stations = load_config_parameters(config_path)

    if not config_stations:
        raise ValueError(
            f"Config '{config_path}' must define at least one ground station "
            f"under the 'ground_stations:' key."
        )
    ground_stations = config_stations

    # Override config with CLI arguments if provided
    use_range = use_range if use_range is not None else config_params['use_range']
    delta_v_ric = config_params['delta_v_ric']

    # Load all other parameters from config
    measurement_noise_deg = config_params['measurement_noise_deg']
    range_noise_m = config_params['range_noise_m']
    process_noise_pos = config_params['process_noise_pos']
    process_noise_vel = config_params['process_noise_vel']
    initial_pos_error = config_params['initial_pos_error']
    initial_vel_error = config_params['initial_vel_error']
    duration = config_params['pm_duration']

    from propagator import OrbitPropagator
    prop = OrbitPropagator("orbDetHOUSE")

    if verbose:
        print("=" * 70)
        print("Extended Kalman Filter Pipeline")
        print("=" * 70)
        print("\nConfiguration:")
        print(f"  Use range measurements: {use_range}")
        print(f"  Measurement type: {'Azimuth/Elevation/Range' if use_range else 'Azimuth/Elevation only'}")
        print(f"  Angular noise: {measurement_noise_deg} degrees")
        if use_range:
            print(f"  Range noise: {range_noise_m} metres")
        if delta_v_ric is not None:
            print(f"  Delta-v (RIC): [R:{delta_v_ric[0]:.4f}, I:{delta_v_ric[1]:.4f}, C:{delta_v_ric[2]:.4f}] m/s")
            print(f"  Post-manoeuvre duration: {duration} days")

    # Step 1: Run orbit propagator (identical to FGO pipeline)
    if verbose:
        print("\n1. Running orbit propagator...")

    delta_v_eci = None
    manoeuvre_state = None
    first_csv_path = None
    if delta_v_ric is not None:
        delta_v_ric = np.array(delta_v_ric, dtype=float)

        first_csv_path = prop.propagate(config_path, output_file="ekf_truth_pre_delta.csv")
        df_pre = pd.read_csv(first_csv_path)
        manoeuvre_state = df_pre[['x', 'y', 'z', 'vx', 'vy', 'vz']].iloc[-1].values
        delta_v_eci = ric_to_eci(delta_v_ric, manoeuvre_state)
        if verbose:
            print(f"   Delta-v (ECI): [{delta_v_eci[0]:.4f}, {delta_v_eci[1]:.4f}, {delta_v_eci[2]:.4f}] m/s")

        last = df_pre.iloc[-1]
        new_state = [
            float(last['x']), float(last['y']), float(last['z']),
            float(float(last['vx']) + delta_v_eci[0]),
            float(float(last['vy']) + delta_v_eci[1]),
            float(float(last['vz']) + delta_v_eci[2])
        ]

        with open(config_path, 'r') as f:
            config_post = yaml.safe_load(f)
        config_post['initial_orbtial_parameters']['initial_state'] = new_state
        config_post['scenario_parameters']['MJD_start'] = config_post['scenario_parameters']['MJD_end']
        config_post['scenario_parameters']['MJD_end'] = config_post['scenario_parameters']['MJD_start'] + duration

        temp_config = config_path.replace('.yml', '_temp_post_ekf.yml')
        with open(temp_config, 'w') as f:
            yaml.dump(config_post, f, default_flow_style=False, sort_keys=False)

        post_csv = prop.propagate(temp_config, output_file="ekf_truth_post_delta.csv")
        os.remove(temp_config)

        df_post = pd.read_csv(post_csv)
        df_post['tSec'] = df_post['tSec'] + df_pre['tSec'].iloc[-1]
        df_combined = pd.concat([df_pre, df_post.iloc[1:]], ignore_index=True)
        csv_path = os.path.abspath("out/ekf_truth_combined.csv")
        df_combined.to_csv(csv_path, index=False)
    else:
        csv_path = prop.propagate(config_path, output_file="ekf_truth.csv")

    if verbose:
        print(f"   Propagation complete: {csv_path}")

    # Step 2: Load propagation results
    truth_states, times, dt = load_propagator_output(csv_path)
    N = len(truth_states)

    if verbose:
        print(f"   Loaded {N} timesteps, dt = {dt} seconds")

    # Step 3: Simulate measurements
    if verbose:
        print("\n2. Simulating measurements...")
        print(f"   Ground stations: {len(ground_stations)}")
        print(f"   Angular noise: {measurement_noise_deg} degrees")
        if use_range:
            print(f"   Range noise: {range_noise_m} metres")

    measurements, R = simulate_measurements(truth_states, times, ground_stations,
                                            measurement_noise_deg, use_range, range_noise_m)

    # Step 4: Setup process noise
    q_pos_ric = np.array(process_noise_pos, dtype=float)
    q_vel_ric = np.array(process_noise_vel, dtype=float)

    # Step 5: Generate initial guess with errors
    if verbose:
        print("\n3. Generating initial state with errors...")

    x0 = truth_states[0].copy()
    x0[:3] += np.random.normal(0, initial_pos_error, 3)
    x0[3:] += np.random.normal(0, initial_vel_error, 3)

    initial_pos_error_actual = np.linalg.norm(x0[:3] - truth_states[0, :3])
    initial_vel_error_actual = np.linalg.norm(x0[3:] - truth_states[0, 3:])

    if verbose:
        print(f"   Position error: {initial_pos_error_actual:.1f} m")
        print(f"   Velocity error: {initial_vel_error_actual:.3f} m/s")
        print("\n4. Running Extended Kalman Filter...")
        print("=" * 70)

    # Step 6: Setup manoeuvres and initial covariance
    manoeuvres = None
    epsilon = config_params.get('epsilon', 0.5)
    dv_initial_error = config_params.get('dv_initial_error', 0.5)

    t_star_true = None
    if delta_v_eci is not None and first_csv_path is not None:
        df_pre_tstar = pd.read_csv(first_csv_path)
        t_star_true = float(df_pre_tstar['tSec'].iloc[-1])

    # Build initial covariance
    n_man_params = 0
    if delta_v_eci is not None and first_csv_path is not None and use_gaussian_estimation:
        dv_guess = delta_v_eci + np.random.normal(0, dv_initial_error, 3)

        t_star_initial_error = config_params.get('t_star_initial_error', 60.0)
        t_star_guess = t_star_true + np.random.normal(0, t_star_initial_error)

        manoeuvres = [{'delta_v': dv_guess, 't_star': t_star_guess}]
        n_man_params = 4

        if verbose:
            dv_guess_ric = eci_to_ric(dv_guess, manoeuvre_state)
            print(f"\n   Gaussian Impulse Approximation:")
            print(f"     epsilon = {epsilon} s")
            print(f"     True delta-v (RIC):  [{delta_v_ric[0]:.4f}, {delta_v_ric[1]:.4f}, {delta_v_ric[2]:.4f}]")
            print(f"     Initial guess (RIC): [{dv_guess_ric[0]:.4f}, {dv_guess_ric[1]:.4f}, {dv_guess_ric[2]:.4f}]")
            print(f"     t* true  = {t_star_true:.2f} s")
            print(f"     t* guess = {t_star_guess:.2f} s (error: {t_star_guess - t_star_true:.2f} s)")

    # P0: match the actual initial error magnitudes
    P0 = build_P0(n_man_params // 4, initial_pos_error, initial_vel_error,
                  dv_initial_error,
                  config_params.get('t_star_initial_error', 120.0))

    # Step 7: Run EKF
    ekf = SatelliteOrbitEKF(measurements, R, q_pos_ric, q_vel_ric,
                            ground_stations, dt, x0=x0, P0=P0,
                            use_range=use_range, manoeuvres=manoeuvres,
                            epsilon=epsilon)
    ekf.run(verbose=verbose)

    # Step 8: Compute final errors
    errors = ekf.states - truth_states
    pos_errors = np.linalg.norm(errors[:, :3], axis=1)
    vel_errors = np.linalg.norm(errors[:, 3:], axis=1)

    if verbose:
        print("\n" + "=" * 70)
        print("Final Results")
        print("=" * 70)
        print(f"Measurement Type: {'Angular + Range' if use_range else 'Angular Only'}")
        print(f"Position RMS: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
        print(f"Position Max: {np.max(pos_errors):.2f} m")
        print(f"Velocity RMS: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
        print(f"Velocity Max: {np.max(vel_errors):.4f} m/s")

        if manoeuvres is not None and delta_v_ric is not None:
            dv_estimated_eci = ekf.man_params[0:3]
            dv_estimated_ric = eci_to_ric(dv_estimated_eci, manoeuvre_state)
            dv_error_ric = dv_estimated_ric - delta_v_ric
            print(f"\nManoeuvre Estimation (RIC):")
            print(f"  True delta-v:      [{delta_v_ric[0]:.4f}, {delta_v_ric[1]:.4f}, {delta_v_ric[2]:.4f}] m/s")
            print(f"  Estimated delta-v: [{dv_estimated_ric[0]:.4f}, {dv_estimated_ric[1]:.4f}, {dv_estimated_ric[2]:.4f}] m/s")
            print(f"  Error:             [{dv_error_ric[0]:.4f}, {dv_error_ric[1]:.4f}, {dv_error_ric[2]:.4f}] m/s")
            print(f"  Error norm:        {np.linalg.norm(dv_error_ric):.4f} m/s")
            if t_star_true is not None:
                t_star_estimated = ekf.man_params[3]
                t_star_error = t_star_estimated - t_star_true
                print(f"  True t*:           {t_star_true:.2f} s")
                print(f"  Estimated t*:      {t_star_estimated:.2f} s")
                print(f"  t* error:          {t_star_error:.2f} s")

    results = {
        'fgo': ekf,  # keep key name to use in plot_fgo_results
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
        'use_range': use_range,
        't_star_true': t_star_true,
    }

    if manoeuvres is not None and delta_v_ric is not None:
        dv_estimated_eci = ekf.man_params[0:3]
        dv_estimated_ric = eci_to_ric(dv_estimated_eci, manoeuvre_state)
        results['dv_true'] = delta_v_ric
        results['dv_estimated'] = dv_estimated_ric
        results['dv_error'] = dv_estimated_ric - delta_v_ric
        results['manoeuvre_state'] = manoeuvre_state
        if t_star_true is not None:
            t_star_estimated = ekf.man_params[3]
            results['t_star_true'] = t_star_true
            results['t_star_estimated'] = t_star_estimated
            results['t_star_error'] = t_star_estimated - t_star_true

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Extended Kalman Filter')
    parser.add_argument('--config', type=str, default='configs/config_geo_one_rev_deltaRIC1.yml',
                        help='Path to orbit config file')
    parser.add_argument('--no-range', dest='use_range', action='store_false', default=True,
                        help='Disable range measurements')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-gaussian', dest='use_gaussian', action='store_false', default=True,
                        help='Disable Gaussian impulse estimation (legacy mode)')

    args = parser.parse_args()

    np.random.seed(7)

    results = run_ekf_with_propagator(
        config_path=args.config,
        use_range=args.use_range,
        verbose=not args.quiet,
        use_gaussian_estimation=args.use_gaussian
    )

    save_name = './plots/ekf_results.png'
    plot_fgo_results(results, save_path=save_name)

    save_data = './out/ekf_results.npz'
    np.savez(save_data,
             truth=results['truth'],
             estimated=results['estimated'],
             errors=results['errors'],
             times=results['times'],
             use_range=results['use_range'])

    print(f"\nResults saved to: {save_data}")
