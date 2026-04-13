#!/usr/bin/env python3
"""
Integration script for running Factor Graph Optimisation with orbDetHOUSE propagator
Supports both angular-only and angular+range measurements (see README)
"""

import os
import numpy as np
import pandas as pd
import yaml
from math import pi, atan2, sin, cos, sqrt
import matplotlib.pyplot as plt
from Orbit_FGO import SatelliteOrbitFGO, ric_to_eci, eci_to_ric, eci_to_ric_rotation_matrix


def load_propagator_output(csv_path):
    """Load orbit propagation results from CSV file"""
    df = pd.read_csv(csv_path)
    # Expected columns: tSec, x, y, z, vx, vy, vz
    states = df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
    times = df['tSec'].values
    dt = times[1] - times[0] if len(times) > 1 else 60.0
    return states, times, dt


def simulate_measurements(states, times, ground_stations, 
                          measurement_noise_deg=0.01,
                          use_range=True,
                          range_noise_m=100.0):
    """
    Simulate azimuth/elevation and optionally range measurements from ground stations
    
    Args:
        states: Array of satellite states [x, y, z, vx, vy, vz]
        times: Array of time points
        ground_stations: List of (lat, lon, alt) tuples for ground stations
        measurement_noise_deg: Measurement noise in degrees for angles
        use_range: Whether to include range measurements
        range_noise_m: Range measurement noise in metres
        
    Returns:
        measurements: Flattened array of measurements
        R: Measurement noise covariance matrix
    """
    omega_earth = 7.2921159e-5 # Angular velocity of Earth (rad/s)
    R_earth = 6378137.0 # Radius of Earth (m)
    
    measurements = []
    angle_noise_rad = np.deg2rad(measurement_noise_deg)
    
    for i, (state, t) in enumerate(zip(states, times)):
        for lat, lon, alt in ground_stations:
            # Compute measurements
            az, el, rng = compute_measurements_full(state[:3], (lat, lon, alt), t, 
                                                   omega_earth, R_earth)
            
            # Add measurement noise
            az_meas = az + np.random.normal(0, angle_noise_rad)
            el_meas = el + np.random.normal(0, angle_noise_rad)
            
            if use_range:
                rng_meas = rng + np.random.normal(0, range_noise_m)
                measurements.extend([az_meas, el_meas, rng_meas])
            else:
                measurements.extend([az_meas, el_meas])
    
    measurements = np.array(measurements)
    
    # Measurement noise covariance matrix
    if use_range:
        R = np.eye(3)
        R[0, 0] = angle_noise_rad**2  # Azimuth
        R[1, 1] = angle_noise_rad**2  # Elevation
        R[2, 2] = range_noise_m**2    # Range
    else:
        R = np.eye(2) * angle_noise_rad**2
    
    return measurements, R


def compute_measurements_full(r_sat_eci, station_llh, t, omega_earth, R_earth):
    """Compute azimuth, elevation, and range from ground station to satellite"""
    lat, lon, alt = station_llh
    
    # Earth rotation angle
    theta = omega_earth * t
    
    # Rotation matrix from ECEF to ECI
    R_ecef_to_eci = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Ground station in ECEF
    r_station_ecef = np.array([
        (R_earth + alt) * cos(lat) * cos(lon),
        (R_earth + alt) * cos(lat) * sin(lon),
        (R_earth + alt) * sin(lat)
    ])
    
    # Convert to ECI
    r_station_eci = R_ecef_to_eci @ r_station_ecef
    
    # Relative position in ECI
    r_rel_eci = r_sat_eci - r_station_eci
    
    # Range
    range_val = np.linalg.norm(r_rel_eci)
    
    # Convert to ECEF
    R_eci_to_ecef = R_ecef_to_eci.T
    r_rel_ecef = R_eci_to_ecef @ r_rel_eci
    
    # Convert to ENU (East-North-Up)
    R_ecef_to_enu = np.array([
        [-sin(lon), cos(lon), 0],
        [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)],
        [cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]
    ])
    
    r_enu = R_ecef_to_enu @ r_rel_ecef
    e, n, u = r_enu
    
    # Compute azimuth and elevation
    range_horiz = sqrt(e**2 + n**2)
    azimuth = atan2(e, n)
    elevation = atan2(u, range_horiz)
    
    return azimuth, elevation, range_val


def load_config_parameters(config_path):
    """Load FGO parameters from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Default parameters
    params = {
        'use_range': True,
        'measurement_noise_deg': 0.00278,
        'range_noise_m': 100.0,
        'process_noise_pos': [0.01, 0.01, 0.01],
        'process_noise_vel': [0.001, 0.001, 0.001],
        'initial_pos_error': 1000.0,
        'initial_vel_error': 1.0,
        'max_iterations': 50,
        'delta_v_ric': None,
        'pm_duration': 0.85,
        'epsilon': 0.5,
        'dv_initial_error': 0.5,
        't_star_initial_error': 120.0
    }

    # Load from config if available (defaults come from the `params` dict above)
    if 'fgo_parameters' in config:
        fgo_params = config['fgo_parameters']
        params['use_range'] = fgo_params.get('use_range', params['use_range'])
        params['measurement_noise_deg'] = fgo_params.get('measurement_noise_deg', params['measurement_noise_deg'])
        params['range_noise_m'] = fgo_params.get('range_noise_m', params['range_noise_m'])
        params['process_noise_pos'] = fgo_params.get('process_noise_position', params['process_noise_pos'])
        params['process_noise_vel'] = fgo_params.get('process_noise_velocity', params['process_noise_vel'])
        params['initial_pos_error'] = fgo_params.get('initial_position_error', params['initial_pos_error'])
        params['initial_vel_error'] = fgo_params.get('initial_velocity_error', params['initial_vel_error'])
        params['max_iterations'] = fgo_params.get('max_iterations', params['max_iterations'])

    # Load manoeuvre parameters if available
    if 'manoeuvre_parameters' in config:
        manoeuvre_params = config['manoeuvre_parameters']
        params['delta_v_ric'] = manoeuvre_params.get('delta_v_ric', params['delta_v_ric'])
        params['pm_duration'] = manoeuvre_params.get('pm_duration', params['pm_duration'])
        params['epsilon'] = manoeuvre_params.get('epsilon', params['epsilon'])
        params['dv_initial_error'] = manoeuvre_params.get('dv_initial_error', params['dv_initial_error'])
        params['t_star_initial_error'] = manoeuvre_params.get('t_star_initial_error', params['t_star_initial_error'])

    # Load ground stations if available
    ground_stations = None
    if 'ground_stations' in config:
        ground_stations = []
        for station in config['ground_stations']:
            lat_deg = station.get('latitude', 0)
            lon_deg = station.get('longitude', 0)
            alt_m = station.get('altitude', 0)
            ground_stations.append((np.deg2rad(lat_deg), np.deg2rad(lon_deg), alt_m))
    
    return params, ground_stations


def run_fgo_with_propagator(config_path,
                           use_range=None,
                           max_iterations=None,
                           verbose=True,
                           use_gaussian_estimation=True):
    """
    Complete pipeline: propagate orbit, simulate measurements, run FGO

    Args:
        config_path: Path to orbit propagator config file
        use_range: Whether to use range measurements (None to load from config)
        max_iterations: Maximum optimisation iterations (None to load from config)
        verbose: Print progress information
        use_gaussian_estimation: If True, use Gaussian impulse approximation to
            estimate delta-v within the FGO. If False, use the legacy split-propagation
            approach (no manoeuvre modelling in FGO).

    Returns:
        Dictionary with results
    """

    # Load parameters from config
    config_params, config_stations = load_config_parameters(config_path)

    # Ground stations must be specified in the config -- no silent defaults.
    if not config_stations:
        raise ValueError(
            f"Config '{config_path}' must define at least one ground station "
            f"under the 'ground_stations:' key."
        )
    ground_stations = config_stations

    # Override config with CLI arguments if provided
    use_range = use_range if use_range is not None else config_params['use_range']
    max_iterations = max_iterations if max_iterations is not None else config_params['max_iterations']
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
        print("="*70)
        print("Factor Graph Optimisation Pipeline")
        print("="*70)
        print("\nConfiguration:")
        print(f"  Use range measurements: {use_range}")
        print(f"  Measurement type: {'Azimuth/Elevation/Range' if use_range else 'Azimuth/Elevation only'}")
        print(f"  Angular noise: {measurement_noise_deg} degrees")
        if use_range:
            print(f"  Range noise: {range_noise_m} metres")
        if delta_v_ric is not None:
            print(f"  Delta-v (RIC): [R:{delta_v_ric[0]:.4f}, I:{delta_v_ric[1]:.4f}, C:{delta_v_ric[2]:.4f}] m/s")
            print(f"  Post-manoeuvre duration: {duration} days")

    # Step 1: Run orbit propagator
    if verbose:
        print("\n1. Running orbit propagator...")

    delta_v_eci = None
    manoeuvre_state = None
    first_csv_path = None
    if delta_v_ric is not None:
        delta_v_ric = np.array(delta_v_ric, dtype=float)

        # Pre-manoeuvre propagation (single propagation, reused for RIC conversion
        # and as the first leg of the combined trajectory)
        first_csv_path = prop.propagate(config_path, output_file="fgo_truth_pre_delta.csv")
        df_pre = pd.read_csv(first_csv_path)
        manoeuvre_state = df_pre[['x', 'y', 'z', 'vx', 'vy', 'vz']].iloc[-1].values
        delta_v_eci = ric_to_eci(delta_v_ric, manoeuvre_state)
        print(f"   Delta-v (ECI): [{delta_v_eci[0]:.4f}, {delta_v_eci[1]:.4f}, {delta_v_eci[2]:.4f}] m/s")
        # Apply delta-v to get post-manoeuvre initial state
        last = df_pre.iloc[-1]
        new_state = [
            float(last['x']), float(last['y']), float(last['z']),
            float(float(last['vx']) + delta_v_eci[0]),
            float(float(last['vy']) + delta_v_eci[1]),
            float(float(last['vz']) + delta_v_eci[2])
        ]

        # Propagate post-manoeuvre arc
        with open(config_path, 'r') as f:
            config_post = yaml.safe_load(f)
        config_post['initial_orbtial_parameters']['initial_state'] = new_state
        config_post['scenario_parameters']['MJD_start'] = config_post['scenario_parameters']['MJD_end']
        config_post['scenario_parameters']['MJD_end'] = config_post['scenario_parameters']['MJD_start'] + duration

        temp_config = config_path.replace('.yml', '_temp_post.yml')
        with open(temp_config, 'w') as f:
            yaml.dump(config_post, f, default_flow_style=False, sort_keys=False)

        post_csv = prop.propagate(temp_config, output_file="fgo_truth_post_delta.csv")
        os.remove(temp_config)

        # Combine pre + post into single trajectory
        df_post = pd.read_csv(post_csv)
        df_post['tSec'] = df_post['tSec'] + df_pre['tSec'].iloc[-1]
        df_combined = pd.concat([df_pre, df_post.iloc[1:]], ignore_index=True)
        csv_path = os.path.abspath("out/fgo_truth_combined.csv")
        df_combined.to_csv(csv_path, index=False)
    else:
        csv_path = prop.propagate(config_path, output_file="fgo_truth.csv")

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
        print("\n4. Running Factor Graph Optimisation...")
        print("="*70)
    
    # Step 6: Run FGO
    manoeuvres = None
    epsilon = config_params.get('epsilon', 0.5)
    dv_initial_error = config_params.get('dv_initial_error', 0.5)

    t_star_true = None
    if delta_v_eci is not None and first_csv_path is not None:
        df_pre_tstar = pd.read_csv(first_csv_path)
        t_star_true = float(df_pre_tstar['tSec'].iloc[-1])

    if delta_v_eci is not None and first_csv_path is not None and use_gaussian_estimation:
        # Create initial guesses with noise (applied in ECI)
        dv_guess = delta_v_eci + np.random.normal(0, dv_initial_error, 3)

        t_star_initial_error = config_params.get('t_star_initial_error', 60.0)
        t_star_guess = t_star_true + np.random.normal(0, t_star_initial_error)

        manoeuvres = [{'delta_v': dv_guess, 't_star': t_star_guess}]

        if verbose:
            dv_guess_ric = eci_to_ric(dv_guess, manoeuvre_state)
            print(f"\n   Gaussian Impulse Approximation:")
            print(f"     epsilon = {epsilon} s")
            print(f"     True delta-v (RIC):  [{delta_v_ric[0]:.4f}, {delta_v_ric[1]:.4f}, {delta_v_ric[2]:.4f}]")
            print(f"     Initial guess (RIC): [{dv_guess_ric[0]:.4f}, {dv_guess_ric[1]:.4f}, {dv_guess_ric[2]:.4f}]")
            print(f"     t* true  = {t_star_true:.2f} s")
            print(f"     t* guess = {t_star_guess:.2f} s (error: {t_star_guess - t_star_true:.2f} s)")

    fgo = SatelliteOrbitFGO(measurements, R, q_pos_ric, q_vel_ric,
                            ground_stations, dt, x0=x0,
                            use_range=use_range, manoeuvres=manoeuvres, epsilon=epsilon)
    fgo.opt(max_iters=max_iterations, verbose=verbose)
    
    # Step 7: Compute final errors
    errors = fgo.states - truth_states
    pos_errors = np.linalg.norm(errors[:, :3], axis=1)
    vel_errors = np.linalg.norm(errors[:, 3:], axis=1)
    
    if verbose:
        print("\n" + "="*70)
        print("Final Results")
        print("="*70)
        print(f"Measurement Type: {'Angular + Range' if use_range else 'Angular Only'}")
        print(f"Position RMS: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
        print(f"Position Max: {np.max(pos_errors):.2f} m")
        print(f"Velocity RMS: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
        print(f"Velocity Max: {np.max(vel_errors):.4f} m/s")
        
        if not use_range:
            print("\nNote: High position errors are expected with angular-only measurements.")
            print("Enable range measurements for sub-kilometer accuracy.")

        # Report manoeuvre estimation errors in RIC
        if manoeuvres is not None and delta_v_ric is not None:
            dv_estimated_eci = fgo.man_params[0:3]
            dv_estimated_ric = eci_to_ric(dv_estimated_eci, manoeuvre_state)
            dv_error_ric = dv_estimated_ric - delta_v_ric
            print(f"\nManoeuvre Estimation (RIC):")
            print(f"  True delta-v:      [{delta_v_ric[0]:.4f}, {delta_v_ric[1]:.4f}, {delta_v_ric[2]:.4f}] m/s")
            print(f"  Estimated delta-v: [{dv_estimated_ric[0]:.4f}, {dv_estimated_ric[1]:.4f}, {dv_estimated_ric[2]:.4f}] m/s")
            print(f"  Error:             [{dv_error_ric[0]:.4f}, {dv_error_ric[1]:.4f}, {dv_error_ric[2]:.4f}] m/s")
            print(f"  Error norm:        {np.linalg.norm(dv_error_ric):.4f} m/s")
            if t_star_true is not None:
                t_star_estimated = fgo.man_params[3]
                t_star_error = t_star_estimated - t_star_true
                print(f"  True t*:           {t_star_true:.2f} s")
                print(f"  Estimated t*:      {t_star_estimated:.2f} s")
                print(f"  t* error:          {t_star_error:.2f} s")

    results = {
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
        'use_range': use_range,
        't_star_true': t_star_true,
    }

    # Add manoeuvre estimation info to results (all in RIC)
    if manoeuvres is not None and delta_v_ric is not None:
        dv_estimated_eci = fgo.man_params[0:3]
        dv_estimated_ric = eci_to_ric(dv_estimated_eci, manoeuvre_state)
        results['dv_true'] = delta_v_ric
        results['dv_estimated'] = dv_estimated_ric
        results['dv_error'] = dv_estimated_ric - delta_v_ric
        results['manoeuvre_state'] = manoeuvre_state
        if t_star_true is not None:
            t_star_estimated = fgo.man_params[3]
            results['t_star_true'] = t_star_true
            results['t_star_estimated'] = t_star_estimated
            results['t_star_error'] = t_star_estimated - t_star_true

    return results


def plot_fgo_results(results, save_path='fgo_results.png'):
    """Generate comprehensive plots of FGO results"""
    
    truth = results['truth']
    estimated = results['estimated']
    errors = results['errors']
    pos_errors = results['pos_errors']
    vel_errors = results['vel_errors']
    delta_v_ric = results.get('delta_v_ric')
    use_range = results.get('use_range', False)
    
    fig = plt.figure(figsize=(18, 12))

    meas_type = "Angular + Range" if use_range else "Angular Only"

    # 3D Trajectory
    ax1 = fig.add_subplot(2, 4, (1, 2), projection='3d')
    ax1.plot(truth[:, 0]/1e3, truth[:, 1]/1e3, truth[:, 2]/1e3,
             'r-', linewidth=2, label='Truth')
    ax1.plot(estimated[:, 0]/1e3, estimated[:, 1]/1e3, estimated[:, 2]/1e3,
             'b--', linewidth=1, alpha=0.7, label='Estimated')
    ax1.scatter(*truth[0, :3]/1e3, color='green', s=80, zorder=5, label='Start')
    ax1.scatter(*truth[-1, :3]/1e3, color='black', s=80, zorder=5, label='End')
    t_star_true_plot = results.get('t_star_true')
    if t_star_true_plot is not None:
        man_idx = min(round(t_star_true_plot / results['dt']), len(truth) - 1)
        ax1.scatter(*truth[man_idx, :3]/1e3, color='orange', s=120, marker='*', zorder=5, label='Manoeuvre')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title(f'3D Trajectory ({meas_type})')
    ax1.legend()

    # Set equal aspect ratio to show true orbital geometry
    truth_km = truth / 1e3  # Convert to km
    max_range = np.array([
        truth_km[:, 0].max() - truth_km[:, 0].min(),
        truth_km[:, 1].max() - truth_km[:, 1].min(),
        truth_km[:, 2].max() - truth_km[:, 2].min()
    ]).max() / 2.0

    mid_x = (truth_km[:, 0].max() + truth_km[:, 0].min()) * 0.5
    mid_y = (truth_km[:, 1].max() + truth_km[:, 1].min()) * 0.5
    mid_z = (truth_km[:, 2].max() + truth_km[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.set_box_aspect([1, 1, 1])

    # Summary statistics
    ax2 = fig.add_subplot(2, 4, (3, 4))
    ax2.axis('off')
    # Build manoeuvre estimation text
    man_text = ""
    if 'dv_estimated' in results:
        dv_est = results['dv_estimated']
        dv_err = results['dv_error']
        man_text = f"""
    Manoeuvre Estimation (RIC):
      Est: [{dv_est[0]:.4f}, {dv_est[1]:.4f}, {dv_est[2]:.4f}]
      Err: [{dv_err[0]:.4f}, {dv_err[1]:.4f}, {dv_err[2]:.4f}]
      |Err|: {np.linalg.norm(dv_err):.4f} m/s"""
        if 't_star_error' in results:
            man_text += f"\n      t* err: {results['t_star_error']:.2f} s"

    if delta_v_ric is not None:
        dv_label = f"""    Delta-V (RIC):
      R: {delta_v_ric[0]} m/s
      I: {delta_v_ric[1]} m/s
      C: {delta_v_ric[2]} m/s"""
    else:
        dv_label = "    No manoeuvre"

    stats_text = f"""
    Measurement Type: {meas_type}

{dv_label}
{man_text}
    Position Errors:
      RMS: {np.sqrt(np.mean(pos_errors**2)):.2f} m
      Max: {np.max(pos_errors):.2f} m
      Mean: {np.mean(pos_errors):.2f} m

    Velocity Errors:
      RMS: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s
      Max: {np.max(vel_errors):.4f} m/s
      Mean: {np.mean(vel_errors):.4f} m/s

    Ground Stations: {len(results['ground_stations'])}
    Timesteps: {len(truth)}
    """
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center',
            fontfamily='monospace')
    ax2.set_title('Summary Statistics')

    # Position errors
    ax3 = fig.add_subplot(2, 4, (5, 6))
    ax3.plot(errors[:, 0], label='X', alpha=0.7)
    ax3.plot(errors[:, 1], label='Y', alpha=0.7)
    ax3.plot(errors[:, 2], label='Z', alpha=0.7)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Component Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Velocity errors
    ax4 = fig.add_subplot(2, 4, (7, 8))
    ax4.plot(errors[:, 3]*1000, label='Vx', alpha=0.7)
    ax4.plot(errors[:, 4]*1000, label='Vy', alpha=0.7)
    ax4.plot(errors[:, 5]*1000, label='Vz', alpha=0.7)
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Velocity Error (mm/s)')
    ax4.set_title('Velocity Component Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'FGO Results - {meas_type}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")

    # Second figure for total errors + absolute velocities
    fig2 = plt.figure(figsize=(16, 10))

    # Total position error
    ax_pos = fig2.add_subplot(2, 2, 1)
    ax_pos.plot(pos_errors)
    ax_pos.axhline(y=np.mean(pos_errors), color='r', linestyle='--',
                   label=f'Mean: {np.mean(pos_errors):.1f}m')
    ax_pos.set_xlabel('Time (minutes)')
    ax_pos.set_ylabel('Position Error (m)')
    ax_pos.set_title('Total Position Error')
    if len(truth) < 50:
        ax_pos.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)

    # Total velocity error
    ax_vel = fig2.add_subplot(2, 2, 2)
    ax_vel.plot(vel_errors*1000)
    ax_vel.axhline(y=np.mean(vel_errors)*1000, color='r', linestyle='--',
                   label=f'Mean: {np.mean(vel_errors)*1000:.2f}mm/s')
    ax_vel.set_xlabel('Time (minutes)')
    ax_vel.set_ylabel('Velocity Error (mm/s)')
    ax_vel.set_title('Total Velocity Error')
    if len(truth) < 50:
        ax_vel.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)

    # Absolute velocities: truth vs estimated
    time_min = np.arange(len(truth))
    for idx, (comp, label) in enumerate([(3, 'Vx'), (4, 'Vy'), (5, 'Vz')]):
        ax_abs = fig2.add_subplot(2, 3, 4 + idx)
        ax_abs.plot(time_min, truth[:, comp], 'r-', linewidth=1.5, label='Truth')
        ax_abs.plot(time_min, estimated[:, comp], 'b--', linewidth=1, alpha=0.7, label='Estimated')
        ax_abs.set_xlabel('Time (minutes)')
        ax_abs.set_ylabel(f'{label} (m/s)')
        ax_abs.set_title(f'Absolute {label}')
        ax_abs.legend(fontsize=8)
        ax_abs.grid(True, alpha=0.3)

    plt.suptitle(f'Total Errors Using Measurements: {meas_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save second figure with _errors suffix
    error_save_path = save_path.replace('.png', '_errors.png')
    plt.savefig(error_save_path, dpi=150)
    print(f"Error plots saved to: {error_save_path}")

    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Factor Graph Optimisation')
    parser.add_argument('--config', type=str, default='configs/config_geo_realistic.yml',
                       help='Path to orbit propagator config file')
    parser.add_argument('--no-range', dest='use_range', action='store_false', default=True,
                       help='Disable range measurements (range enabled by default)')
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum optimisation iterations')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--no-gaussian', dest='use_gaussian', action='store_false', default=True,
                       help='Disable Gaussian impulse estimation (use legacy split-propagation)')

    args = parser.parse_args()

    np.random.seed(7)  # For reproducibility

    # Run FGO pipeline
    results = run_fgo_with_propagator(
        config_path=args.config,
        use_range=args.use_range,
        max_iterations=args.max_iters,
        verbose=not args.quiet,
        use_gaussian_estimation=args.use_gaussian
    )
    
    # Generate plots
    save_name = './plots/fgo_results.png' if results['use_range'] else './plots/fgo_results_angular.png'
    plot_fgo_results(results, save_path=save_name)
    
    # Save results
    save_data = './out/fgo_results.npz' if results['use_range'] else './out/fgo_results_angular.npz'
    np.savez(save_data,
             truth=results['truth'],
             estimated=results['estimated'],
             errors=results['errors'],
             times=results['times'],
             use_range=results['use_range'])
    
    print(f"\nResults saved to: {save_data}")
    
    # plt.show()
