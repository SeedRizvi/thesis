#!/usr/bin/env python3
"""
Integration script for running Factor Graph Optimization with orbDetHOUSE propagator
This script bridges the gap between the orbit propagator and FGO implementation
"""

import numpy as np
import pandas as pd
import yaml
import os
from math import pi, atan2, sin, cos, sqrt
import matplotlib.pyplot as plt
from Orbit_FGO_fixed import SatelliteOrbitFGO


def load_propagator_output(csv_path):
    """Load orbit propagation results from CSV file"""
    df = pd.read_csv(csv_path)
    # Expected columns: tSec, x, y, z, vx, vy, vz
    states = df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
    times = df['tSec'].values
    dt = times[1] - times[0] if len(times) > 1 else 60.0
    return states, times, dt


def simulate_measurements(states, times, ground_stations, measurement_noise_deg=0.01):
    """
    Simulate azimuth/elevation measurements from ground stations
    
    Args:
        states: Array of satellite states [x, y, z, vx, vy, vz]
        times: Array of time points
        ground_stations: List of (lat, lon, alt) tuples for ground stations
        measurement_noise_deg: Measurement noise in degrees
        
    Returns:
        measurements: Flattened array of az/el measurements
        R: Measurement noise covariance matrix
    """
    omega_earth = 7.2921159e-5
    R_earth = 6378137.0
    
    measurements = []
    noise_rad = np.deg2rad(measurement_noise_deg)
    
    for i, (state, t) in enumerate(zip(states, times)):
        for lat, lon, alt in ground_stations:
            # Compute azimuth and elevation
            az, el = compute_az_el(state[:3], (lat, lon, alt), t, 
                                  omega_earth, R_earth)
            
            # Add measurement noise
            az_meas = az + np.random.normal(0, noise_rad)
            el_meas = el + np.random.normal(0, noise_rad)
            
            measurements.extend([az_meas, el_meas])
    
    measurements = np.array(measurements)
    R = np.eye(2) * (noise_rad**2)
    
    return measurements, R


def compute_az_el(r_sat_eci, station_llh, t, omega_earth, R_earth):
    """Compute azimuth and elevation from ground station to satellite"""
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
    
    return azimuth, elevation


def run_fgo_with_propagator(config_path, ground_stations=None, 
                           measurement_noise_deg=0.01,
                           process_noise_pos=100.0,
                           process_noise_vel=0.01,
                           initial_pos_error=1000.0,
                           initial_vel_error=1.0,
                           verbose=True):
    """
    Complete pipeline: propagate orbit, simulate measurements, run FGO
    
    Args:
        config_path: Path to orbit propagator config file
        ground_stations: List of (lat, lon, alt) tuples, or None for defaults
        measurement_noise_deg: Measurement noise in degrees
        process_noise_pos: Process noise for position (m)
        process_noise_vel: Process noise for velocity (m/s)
        initial_pos_error: Initial position error std dev (m)
        initial_vel_error: Initial velocity error std dev (m/s)
        verbose: Print progress information
        
    Returns:
        Dictionary with results
    """
    
    # Default ground stations if not provided
    if ground_stations is None:
        ground_stations = [
            (np.deg2rad(40.7128), np.deg2rad(-74.0060), 0),   # New York
            (np.deg2rad(51.5074), np.deg2rad(-0.1278), 0),    # London  
            (np.deg2rad(35.6762), np.deg2rad(139.6503), 0),   # Tokyo
            (np.deg2rad(-33.8688), np.deg2rad(151.2093), 0),  # Sydney
            (np.deg2rad(1.3521), np.deg2rad(103.8198), 0),    # Singapore
            (np.deg2rad(-23.5505), np.deg2rad(-46.6333), 0)   # São Paulo
        ]
    
    if verbose:
        print("="*70)
        print("Factor Graph Optimization Pipeline")
        print("="*70)
    
    # Step 1: Run orbit propagator
    if verbose:
        print("\n1. Running orbit propagator...")
    
    try:
        from propagator import OrbitPropagator
        prop = OrbitPropagator("orbDetHOUSE")
        csv_path = prop.propagate(config_path, output_file="fgo_truth.csv")
        
        if verbose:
            print(f"   Propagation complete: {csv_path}")
    except ImportError:
        if verbose:
            print("   WARNING: Could not import propagator, using test data")
        # Generate test data as fallback
        return run_fgo_test_case(ground_stations, measurement_noise_deg,
                                process_noise_pos, process_noise_vel,
                                initial_pos_error, initial_vel_error,
                                verbose)
    
    # Step 2: Load propagation results
    truth_states, times, dt = load_propagator_output(csv_path)
    N = len(truth_states)
    
    if verbose:
        print(f"   Loaded {N} timesteps, dt = {dt} seconds")
    
    # Step 3: Simulate measurements
    if verbose:
        print("\n2. Simulating measurements...")
        print(f"   Ground stations: {len(ground_stations)}")
        print(f"   Measurement noise: {measurement_noise_deg} degrees")
    
    measurements, R = simulate_measurements(truth_states, times, ground_stations, 
                                           measurement_noise_deg)
    
    # Step 4: Setup process noise
    Q = np.eye(6)
    Q[:3, :3] *= process_noise_pos
    Q[3:, 3:] *= process_noise_vel
    
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
    
    # Step 6: Run FGO
    if verbose:
        print("\n4. Running Factor Graph Optimization...")
        print("="*70)
    
    fgo = SatelliteOrbitFGO(measurements, R, Q, ground_stations, dt, x0=x0)
    fgo.opt(max_iters=50, verbose=verbose)
    
    # Step 7: Compute final errors
    errors = fgo.states - truth_states
    pos_errors = np.linalg.norm(errors[:, :3], axis=1)
    vel_errors = np.linalg.norm(errors[:, 3:], axis=1)
    
    if verbose:
        print("\n" + "="*70)
        print("Final Results")
        print("="*70)
        print(f"Position RMS: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
        print(f"Position Max: {np.max(pos_errors):.2f} m")
        print(f"Velocity RMS: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
        print(f"Velocity Max: {np.max(vel_errors):.4f} m/s")
    
    return {
        'fgo': fgo,
        'truth': truth_states,
        'estimated': fgo.states,
        'measurements': measurements,
        'errors': errors,
        'pos_errors': pos_errors,
        'vel_errors': vel_errors,
        'times': times,
        'dt': dt,
        'ground_stations': ground_stations
    }


def run_fgo_test_case(ground_stations, measurement_noise_deg,
                     process_noise_pos, process_noise_vel,
                     initial_pos_error, initial_vel_error,
                     verbose=True):
    """Run FGO with generated test data (fallback when propagator unavailable)"""
    
    if verbose:
        print("   Using generated test case (GEO satellite)")
    
    # Generate GEO test case
    r0 = 42164000  # GEO radius
    inclination = np.deg2rad(5)
    
    x0_truth = np.array([
        r0 * cos(inclination),
        0,
        r0 * sin(inclination),
        0,
        sqrt(3.986004418e14 / r0) * cos(inclination),
        0
    ])
    
    dt = 60.0
    N = 100
    times = np.arange(N) * dt
    
    # Generate truth trajectory
    Q = np.eye(6)
    Q[:3, :3] *= process_noise_pos
    Q[3:, 3:] *= process_noise_vel
    
    R = np.eye(2) * (np.deg2rad(measurement_noise_deg))**2
    
    fgo_truth = SatelliteOrbitFGO(np.zeros(N * len(ground_stations) * 2),
                                  R, Q, ground_stations, dt, x0=x0_truth)
    truth_states = fgo_truth.states.copy()
    
    # Simulate measurements
    measurements, _ = simulate_measurements(truth_states, times, ground_stations,
                                           measurement_noise_deg)
    
    # Add initial errors
    x0 = x0_truth.copy()
    x0[:3] += np.random.normal(0, initial_pos_error, 3)
    x0[3:] += np.random.normal(0, initial_vel_error, 3)
    
    # Run FGO
    fgo = SatelliteOrbitFGO(measurements, R, Q, ground_stations, dt, x0=x0)
    fgo.opt(max_iters=50, verbose=verbose)
    
    errors = fgo.states - truth_states
    pos_errors = np.linalg.norm(errors[:, :3], axis=1)
    vel_errors = np.linalg.norm(errors[:, 3:], axis=1)
    
    return {
        'fgo': fgo,
        'truth': truth_states,
        'estimated': fgo.states,
        'measurements': measurements,
        'errors': errors,
        'pos_errors': pos_errors,
        'vel_errors': vel_errors,
        'times': times,
        'dt': dt,
        'ground_stations': ground_stations
    }


def plot_fgo_results(results, save_path='fgo_results.png'):
    """Generate comprehensive plots of FGO results"""
    
    truth = results['truth']
    estimated = results['estimated']
    errors = results['errors']
    pos_errors = results['pos_errors']
    vel_errors = results['vel_errors']
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D Trajectory
    ax1 = fig.add_subplot(241, projection='3d')
    ax1.plot(truth[:, 0]/1e6, truth[:, 1]/1e6, truth[:, 2]/1e6,
             'r-', linewidth=2, label='Truth')
    ax1.plot(estimated[:, 0]/1e6, estimated[:, 1]/1e6, estimated[:, 2]/1e6,
             'b--', linewidth=1, alpha=0.7, label='Estimated')
    ax1.set_xlabel('X (Mm)')
    ax1.set_ylabel('Y (Mm)')
    ax1.set_zlabel('Z (Mm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # XY plane
    ax2 = fig.add_subplot(242)
    ax2.plot(truth[:, 0]/1e6, truth[:, 1]/1e6, 'r-', label='Truth')
    ax2.plot(estimated[:, 0]/1e6, estimated[:, 1]/1e6, 'b--', alpha=0.7, label='Estimated')
    ax2.set_xlabel('X (Mm)')
    ax2.set_ylabel('Y (Mm)')
    ax2.set_title('XY Plane')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    
    # Position errors
    ax3 = fig.add_subplot(243)
    ax3.plot(errors[:, 0], label='X', alpha=0.7)
    ax3.plot(errors[:, 1], label='Y', alpha=0.7)
    ax3.plot(errors[:, 2], label='Z', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Component Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Velocity errors
    ax4 = fig.add_subplot(244)
    ax4.plot(errors[:, 3]*1000, label='Vx', alpha=0.7)
    ax4.plot(errors[:, 4]*1000, label='Vy', alpha=0.7)
    ax4.plot(errors[:, 5]*1000, label='Vz', alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Velocity Error (mm/s)')
    ax4.set_title('Velocity Component Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Total position error
    ax5 = fig.add_subplot(245)
    ax5.plot(pos_errors)
    ax5.axhline(y=np.mean(pos_errors), color='r', linestyle='--', 
                label=f'Mean: {np.mean(pos_errors):.1f}m')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Position Error (m)')
    ax5.set_title('Total Position Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Total velocity error
    ax6 = fig.add_subplot(246)
    ax6.plot(vel_errors*1000)
    ax6.axhline(y=np.mean(vel_errors)*1000, color='r', linestyle='--',
                label=f'Mean: {np.mean(vel_errors)*1000:.2f}mm/s')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Velocity Error (mm/s)')
    ax6.set_title('Total Velocity Error')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Error histogram
    ax7 = fig.add_subplot(247)
    ax7.hist(pos_errors, bins=30, alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Position Error (m)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Position Error Distribution')
    ax7.grid(True, alpha=0.3)
    
    # Convergence metric
    ax8 = fig.add_subplot(248)
    window = min(10, len(pos_errors)//10)
    if window > 1:
        rolling_rms = np.convolve(pos_errors**2, np.ones(window)/window, mode='valid')
        rolling_rms = np.sqrt(rolling_rms)
        ax8.plot(rolling_rms)
        ax8.set_xlabel('Time Step')
        ax8.set_ylabel('Rolling RMS Error (m)')
        ax8.set_title(f'Convergence (window={window})')
        ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Factor Graph Optimization Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Factor Graph Optimization')
    parser.add_argument('--config', type=str, default='configs/config_geo_realistic.yml',
                       help='Path to orbit propagator config file')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Measurement noise in degrees')
    parser.add_argument('--pos-error', type=float, default=1000.0,
                       help='Initial position error in meters')
    parser.add_argument('--vel-error', type=float, default=1.0,
                       help='Initial velocity error in m/s')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Run FGO pipeline
    results = run_fgo_with_propagator(
        config_path=args.config,
        measurement_noise_deg=args.noise,
        initial_pos_error=args.pos_error,
        initial_vel_error=args.vel_error,
        verbose=not args.quiet
    )
    
    # Generate plots
    plot_fgo_results(results, save_path='fgo_pipeline_results.png')
    
    # Save results
    np.savez('fgo_pipeline_results.npz',
             truth=results['truth'],
             estimated=results['estimated'],
             errors=results['errors'],
             times=results['times'])
    
    print("\nResults saved to: fgo_pipeline_results.npz")
    
    plt.show()
