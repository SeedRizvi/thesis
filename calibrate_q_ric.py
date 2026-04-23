#!/usr/bin/env python3
"""
Calibrate per-component velocity process noise Q in RIC frame.

Measures the per-step dynamics mismatch (truth propagator vs FGO two-body+J2),
rotates errors into RIC at each timestep, and reports per-component statistics
to determine Q_vel = [q_R, q_I, q_C].
"""

import numpy as np
import yaml

from fgo_pipeline import load_propagator_output, load_config_parameters
from Orbit_FGO import SatelliteOrbitFGO, eci_to_ric_rotation_matrix
from propagator import OrbitPropagator

CONFIG_PATH = "configs/config_geo_one_rev.yml"


def measure_ric_mismatch(config_path, dt_val=60.0):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['scenario_parameters']['time_step'] = dt_val
    config['scenario_parameters']['max_time_step'] = dt_val

    tmp_config = '/tmp/cal_ric.yml'
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f)

    _, ground_stations = load_config_parameters(config_path)

    prop = OrbitPropagator("orbDetHOUSE")
    csv_path = prop.propagate(tmp_config, output_file="cal_ric_truth.csv")
    truth_states, times, dt = load_propagator_output(csv_path)
    N = len(truth_states)

    # Create FGO just to use its dynamics
    dummy_meas = np.zeros(N * len(ground_stations) * 3)
    dummy_R = np.eye(3)
    q_pos_ric = np.array([1.0, 1.0, 1.0])
    q_vel_ric = np.array([1.0, 1.0, 1.0])

    fgo = SatelliteOrbitFGO(
        dummy_meas, dummy_R, q_pos_ric, q_vel_ric, ground_stations,
        dt=dt_val, x0=truth_states[0],
        use_range=True, manoeuvres=None,
    )

    # Per-step errors in RIC
    pos_err_ric = np.zeros((N - 1, 3))  # [R, I, C]
    vel_err_ric = np.zeros((N - 1, 3))  # [R, I, C]

    for i in range(N - 1):
        t_start = i * dt_val
        fgo_next = fgo.prop_one_timestep(truth_states[i], t_start)
        truth_next = truth_states[i + 1]

        # Error in ECI
        pos_err_eci = fgo_next[:3] - truth_next[:3]
        vel_err_eci = fgo_next[3:] - truth_next[3:]

        # Rotate into RIC using truth state at this timestep
        T = eci_to_ric_rotation_matrix(truth_states[i])
        pos_err_ric[i] = T @ pos_err_eci
        vel_err_ric[i] = T @ vel_err_eci

    return pos_err_ric, vel_err_ric, N


def print_report(pos_err, vel_err, N):
    labels = ['R', 'I', 'C']

    print("=" * 100)
    print(f"Per-component dynamics mismatch in RIC frame (FGO J2 vs truth), N={N} steps")
    print("=" * 100)

    print("\nPosition mismatch per step (m):")
    print(f"  {'axis':>4} | {'mean':>12} | {'|mean|':>12} | {'std':>12} | "
          f"{'RMS':>12} | {'max |err|':>12} | {'5x RMS':>12}")
    print("  " + "-" * 90)
    for j, ax in enumerate(labels):
        col = pos_err[:, j]
        mean = np.mean(col)
        abs_mean = np.mean(np.abs(col))
        std = np.std(col)
        rms = np.sqrt(np.mean(col**2))
        max_abs = np.max(np.abs(col))
        print(f"  {ax:>4} | {mean:>+12.4e} | {abs_mean:>12.4e} | {std:>12.4e} | "
              f"{rms:>12.4e} | {max_abs:>12.4e} | {5*rms:>12.4e}")

    print("\nVelocity mismatch per step (m/s):")
    print(f"  {'axis':>4} | {'mean':>12} | {'|mean|':>12} | {'std':>12} | "
          f"{'RMS':>12} | {'max |err|':>12} | {'5x RMS':>12}")
    print("  " + "-" * 90)
    for j, ax in enumerate(labels):
        col = vel_err[:, j]
        mean = np.mean(col)
        abs_mean = np.mean(np.abs(col))
        std = np.std(col)
        rms = np.sqrt(np.mean(col**2))
        max_abs = np.max(np.abs(col))
        print(f"  {ax:>4} | {mean:>+12.4e} | {abs_mean:>12.4e} | {std:>12.4e} | "
              f"{rms:>12.4e} | {max_abs:>12.4e} | {5*rms:>12.4e}")

    # Ratios
    print("\n" + "=" * 100)
    print("Axis ratios (relative to R):")
    print("=" * 100)
    for category, err in [("Position", pos_err), ("Velocity", vel_err)]:
        rms_vals = [np.sqrt(np.mean(err[:, j]**2)) for j in range(3)]
        ref = rms_vals[0] if rms_vals[0] > 0 else 1.0
        print(f"  {category} RMS ratio — R:I:C = "
              f"1.00 : {rms_vals[1]/ref:.2f} : {rms_vals[2]/ref:.2f}")

    # Suggested Q values
    print("\n" + "=" * 100)
    print("Suggested Q values (5x RMS):")
    print("=" * 100)
    q_pos_vals = []
    q_vel_vals = []
    for j, ax in enumerate(labels):
        q_vel = 5 * np.sqrt(np.mean(vel_err[:, j]**2))
        q_pos = 5 * np.sqrt(np.mean(pos_err[:, j]**2))
        q_pos_vals.append(q_pos)
        q_vel_vals.append(q_vel)
        print(f"  {ax}: Q_pos = {q_pos:.6e} m,  Q_vel = {q_vel:.6e} m/s")

    print(f"\n  Config lines:")
    print(f"  process_noise_position: [{q_pos_vals[0]:.6e}, {q_pos_vals[1]:.6e}, {q_pos_vals[2]:.6e}]")
    print(f"  process_noise_velocity: [{q_vel_vals[0]:.6e}, {q_vel_vals[1]:.6e}, {q_vel_vals[2]:.6e}]")


if __name__ == "__main__":
    pos_err, vel_err, N = measure_ric_mismatch(CONFIG_PATH)
    print_report(pos_err, vel_err, N)
