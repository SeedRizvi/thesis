#!/usr/bin/env python3
"""
Integration script for running Factor Graph Optimisation with orbDetHOUSE propagator
Now supports both angular-only and angular+range measurements
"""

import numpy as np
import pandas as pd
import yaml
from math import pi, atan2, sin, cos, sqrt
import matplotlib.pyplot as plt
from Orbit_FGO_with_range import SatelliteOrbitFGO


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
        range_noise_m: Range measurement noise in meters
        
    Returns:
        measurements: Flattened array of measurements
        R: Measurement noise covariance matrix
    """
    omega_earth = 7.2921159e-5
    R_earth = 6378137.0
    
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
    
    # Create measurement noise covariance matrix
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
        'measurement_noise_deg': 0.01,
        'range_noise_m': 100.0,
        'process_noise_pos': 100.0,
        'process_noise_vel': 0.01,
        'initial_pos_error': 1000.0,
        'initial_vel_error': 1.0,
        'max_iterations': 50
    }
    
    # Load from config if available
    if 'fgo_parameters' in config:
        fgo_params = config['fgo_parameters']
        params['use_range'] = fgo_params.get('use_range', True)
        params['measurement_noise_deg'] = fgo_params.get('measurement_noise_deg', 0.01)
        params['range_noise_m'] = fgo_params.get('range_noise_m', 100.0)
        params['process_noise_pos'] = fgo_params.get('process_noise_position', 100.0)
        params['process_noise_vel'] = fgo_params.get('process_noise_velocity', 0.01)
        params['initial_pos_error'] = fgo_params.get('initial_position_error', 1000.0)
        params['initial_vel_error'] = fgo_params.get('initial_velocity_error', 1.0)
        params['max_iterations'] = fgo_params.get('max_iterations', 50)
    
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


def run_fgo_with_propagator(config_path, ground_stations=None,
                           use_range=None,
                           measurement_noise_deg=None,
                           range_noise_m=None,
                           process_noise_pos=None,
                           process_noise_vel=None,
                           initial_pos_error=None,
                           initial_vel_error=None,
                           max_iterations=None,
                           verbose=True):
    """
    Complete pipeline: propagate orbit, simulate measurements, run FGO
    
    Args:
        config_path: Path to orbit propagator config file
        ground_stations: List of (lat, lon, alt) tuples, or None to load from config/defaults
        use_range: Whether to use range measurements (None to load from config)
        measurement_noise_deg: Measurement noise in degrees
        range_noise_m: Range measurement noise in meters
        process_noise_pos: Process noise for position (m)
        process_noise_vel: Process noise for velocity (m/s)
        initial_pos_error: Initial position error std dev (m)
        initial_vel_error: Initial velocity error std dev (m/s)
        max_iterations: Maximum optimisation iterations
        verbose: Print progress information
        
    Returns:
        Dictionary with results
    """
    
    # Load parameters from config
    config_params, config_stations = load_config_parameters(config_path)
    
    # Use provided parameters or fall back to config
    if ground_stations is None:
        ground_stations = config_stations
    if ground_stations is None:
        # Default stations if not in config
        ground_stations = [
            (np.deg2rad(40.7128), np.deg2rad(-74.0060), 0),   # New York
            (np.deg2rad(51.5074), np.deg2rad(-0.1278), 0),    # London  
            (np.deg2rad(35.6762), np.deg2rad(139.6503), 0),   # Tokyo
            (np.deg2rad(-33.8688), np.deg2rad(151.2093), 0),  # Sydney
            (np.deg2rad(1.3521), np.deg2rad(103.8198), 0),    # Singapore
            (np.deg2rad(-23.5505), np.deg2rad(-46.6333), 0)   # São Paulo
        ]
    
    # Use command line args or config values
    use_range = use_range if use_range is not None else config_params['use_range']
    measurement_noise_deg = measurement_noise_deg or config_params['measurement_noise_deg']
    range_noise_m = range_noise_m or config_params['range_noise_m']
    process_noise_pos = process_noise_pos or config_params['process_noise_pos']
    process_noise_vel = process_noise_vel or config_params['process_noise_vel']
    initial_pos_error = initial_pos_error or config_params['initial_pos_error']
    initial_vel_error = initial_vel_error or config_params['initial_vel_error']
    max_iterations = max_iterations or config_params['max_iterations']
    
    if verbose:
        print("="*70)
        print("Factor Graph Optimisation Pipeline")
        print("="*70)
        print("\nConfiguration:")
        print(f"  Use range measurements: {use_range}")
        print(f"  Measurement type: {'Azimuth/Elevation/Range' if use_range else 'Azimuth/Elevation only'}")
        print(f"  Angular noise: {measurement_noise_deg} degrees")
        if use_range:
            print(f"  Range noise: {range_noise_m} meters")
    
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
        return run_fgo_test_case(ground_stations, use_range,
                                measurement_noise_deg, range_noise_m,
                                process_noise_pos, process_noise_vel,
                                initial_pos_error, initial_vel_error,
                                max_iterations, verbose)
    
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
            print(f"   Range noise: {range_noise_m} meters")
    
    measurements, R = simulate_measurements(truth_states, times, ground_stations, 
                                           measurement_noise_deg, use_range, range_noise_m)
    
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
        print("\n4. Running Factor Graph Optimisation...")
        print("="*70)
    
    fgo = SatelliteOrbitFGO(measurements, R, Q, ground_stations, dt, x0=x0, use_range=use_range)
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
        'ground_stations': ground_stations,
        'use_range': use_range
    }


def run_fgo_test_case(ground_stations, use_range,
                     measurement_noise_deg, range_noise_m,
                     process_noise_pos, process_noise_vel,
                     initial_pos_error, initial_vel_error,
                     max_iterations, verbose=True):
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
    
    if use_range:
        R = np.eye(3)
        R[0, 0] = (np.deg2rad(measurement_noise_deg))**2
        R[1, 1] = (np.deg2rad(measurement_noise_deg))**2
        R[2, 2] = range_noise_m**2
    else:
        R = np.eye(2) * (np.deg2rad(measurement_noise_deg))**2
    
    meas_per_station = 3 if use_range else 2
    fgo_truth = SatelliteOrbitFGO(np.zeros(N * len(ground_stations) * meas_per_station),
                                  R, Q, ground_stations, dt, x0=x0_truth, use_range=use_range)
    truth_states = fgo_truth.states.copy()
    
    # Simulate measurements
    measurements, _ = simulate_measurements(truth_states, times, ground_stations,
                                           measurement_noise_deg, use_range, range_noise_m)
    
    # Add initial errors
    x0 = x0_truth.copy()
    x0[:3] += np.random.normal(0, initial_pos_error, 3)
    x0[3:] += np.random.normal(0, initial_vel_error, 3)
    
    # Run FGO
    fgo = SatelliteOrbitFGO(measurements, R, Q, ground_stations, dt, x0=x0, use_range=use_range)
    fgo.opt(max_iters=max_iterations, verbose=verbose)
    
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
        'ground_stations': ground_stations,
        'use_range': use_range
    }


def plot_fgo_results(results, save_path='fgo_results.png'):
    """Generate comprehensive plots of FGO results"""
    
    truth = results['truth']
    estimated = results['estimated']
    errors = results['errors']
    pos_errors = results['pos_errors']
    vel_errors = results['vel_errors']
    use_range = results.get('use_range', False)
    
    fig = plt.figure(figsize=(18, 12))

    # Add title with measurement type
    meas_type = "Angular + Range" if use_range else "Angular Only"

    # 3D Trajectory - spanning first two columns
    ax1 = fig.add_subplot(2, 4, (1, 2), projection='3d')
    ax1.plot(truth[:, 0]/1e3, truth[:, 1]/1e3, truth[:, 2]/1e3,
             'r-', linewidth=2, label='Truth')
    ax1.plot(estimated[:, 0]/1e3, estimated[:, 1]/1e3, estimated[:, 2]/1e3,
             'b--', linewidth=1, alpha=0.7, label='Estimated')
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

    # Position errors
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(errors[:, 0], label='X', alpha=0.7)
    ax3.plot(errors[:, 1], label='Y', alpha=0.7)
    ax3.plot(errors[:, 2], label='Z', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Component Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Velocity errors
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(errors[:, 3]*1000, label='Vx', alpha=0.7)
    ax4.plot(errors[:, 4]*1000, label='Vy', alpha=0.7)
    ax4.plot(errors[:, 5]*1000, label='Vz', alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Velocity Error (mm/s)')
    ax4.set_title('Velocity Component Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Total position error
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(pos_errors)
    ax5.axhline(y=np.mean(pos_errors), color='r', linestyle='--',
                label=f'Mean: {np.mean(pos_errors):.1f}m')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Position Error (m)')
    ax5.set_title('Total Position Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Total velocity error
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.plot(vel_errors*1000)
    ax6.axhline(y=np.mean(vel_errors)*1000, color='r', linestyle='--',
                label=f'Mean: {np.mean(vel_errors)*1000:.2f}mm/s')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Velocity Error (mm/s)')
    ax6.set_title('Total Velocity Error')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Error histogram
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.hist(pos_errors, bins=30, alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Position Error (m)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Position Error Distribution')
    ax7.grid(True, alpha=0.3)

    # Summary statistics
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    stats_text = f"""
    Measurement Type: {meas_type}
    
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
    ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes, 
            fontsize=10, verticalalignment='center',
            fontfamily='monospace')
    ax8.set_title('Summary Statistics')
    
    plt.suptitle(f'Factor Graph Optimisation Results - {meas_type}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Factor Graph Optimisation')
    parser.add_argument('--config', type=str, default='configs/config_geo_realistic.yml',
                       help='Path to orbit propagator config file')
    parser.add_argument('--no-range', dest='use_range', action='store_false', default=True,
                       help='Disable range measurements (range enabled by default)')
    parser.add_argument('--noise', type=float, default=None,
                       help='Angular measurement noise in degrees')
    parser.add_argument('--range-noise', type=float, default=None,
                       help='Range measurement noise in meters')
    parser.add_argument('--pos-error', type=float, default=None,
                       help='Initial position error in meters')
    parser.add_argument('--vel-error', type=float, default=None,
                       help='Initial velocity error in m/s')
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum optimisation iterations')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Run FGO pipeline
    results = run_fgo_with_propagator(
        config_path=args.config,
        use_range=args.use_range,
        measurement_noise_deg=args.noise,
        range_noise_m=args.range_noise,
        initial_pos_error=args.pos_error,
        initial_vel_error=args.vel_error,
        max_iterations=args.max_iters,
        verbose=not args.quiet
    )
    
    # Generate plots
    save_name = './plots/fgo_results_full.png' if results['use_range'] else './plots/fgo_results_angular.png'
    plot_fgo_results(results, save_path=save_name)
    
    # Save results
    save_data = './out/fgo_results_full.npz' if results['use_range'] else './out/fgo_results_angular.npz'
    np.savez(save_data,
             truth=results['truth'],
             estimated=results['estimated'],
             errors=results['errors'],
             times=results['times'],
             use_range=results['use_range'])
    
    print(f"\nResults saved to: {save_data}")
    
    plt.show()
