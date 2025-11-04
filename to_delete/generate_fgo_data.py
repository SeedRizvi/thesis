import numpy as np
import pandas as pd
from math import pi, sqrt, atan2, sin, cos

def compute_azimuth_elevation(r_sat_eci, r_station_ecef, t):
    R_earth = 6378137.0
    lat, lon, alt = r_station_ecef
    
    r_station_ecef_xyz = np.array([
        (R_earth + alt) * cos(lat) * cos(lon),
        (R_earth + alt) * cos(lat) * sin(lon),
        (R_earth + alt) * sin(lat)
    ])
    
    theta = 7.2921159e-5 * t
    R_z = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    r_station_eci = R_z @ r_station_ecef_xyz
    
    r_rel = r_sat_eci - r_station_eci
    
    theta_ecef = 7.2921159e-5 * t
    R_z_ecef = np.array([
        [cos(theta_ecef), sin(theta_ecef), 0],
        [-sin(theta_ecef), cos(theta_ecef), 0],
        [0, 0, 1]
    ])
    r_rel_ecef = R_z_ecef @ r_rel
    
    R_SEZ = np.array([
        [sin(lat) * cos(lon), sin(lat) * sin(lon), -cos(lat)],
        [-sin(lon), cos(lon), 0],
        [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]
    ])
    
    r_sez = R_SEZ @ r_rel_ecef
    s, e, z = r_sez
    range_horiz = sqrt(s**2 + e**2)
    
    azimuth = atan2(e, s)
    elevation = atan2(z, range_horiz)
    
    return azimuth, elevation

def generate_measurements(truth_csv, output_npz):
    df = pd.read_csv(truth_csv)
    
    ground_stations = [
        np.array([np.radians(0.0), np.radians(0.0), 0.0]),
        np.array([np.radians(30.0), np.radians(90.0), 0.0]),
        np.array([np.radians(-30.0), np.radians(-90.0), 0.0])
    ]
    
    n_stations = len(ground_stations)
    n_timesteps = len(df)
    dt = df['tSec'].iloc[1] - df['tSec'].iloc[0] if len(df) > 1 else 60.0
    
    truth = np.zeros((n_timesteps, 6))
    truth[:, 0] = df['x'].values
    truth[:, 1] = df['y'].values
    truth[:, 2] = df['z'].values
    truth[:, 3] = df['vx'].values
    truth[:, 4] = df['vy'].values
    truth[:, 5] = df['vz'].values
    
    meas = np.zeros(n_timesteps * n_stations * 2)
    
    angle_noise_std = np.radians(0.1)
    R = np.eye(2) * (angle_noise_std**2)
    Q = np.eye(6)
    Q[:3, :3] *= 0.01**2
    Q[3:, 3:] *= 0.0001**2
    
    for i in range(n_timesteps):
        t = df['tSec'].iloc[i]
        r_sat = truth[i, :3]
        
        for s_idx, station in enumerate(ground_stations):
            az, el = compute_azimuth_elevation(r_sat, station, t)
            
            az_noisy = az + np.random.normal(0, angle_noise_std)
            el_noisy = el + np.random.normal(0, angle_noise_std)
            
            meas_idx = i * n_stations * 2 + s_idx * 2
            meas[meas_idx] = az_noisy
            meas[meas_idx + 1] = el_noisy
    
    np.savez(output_npz,
             meas=meas,
             truth=truth,
             dt=dt,
             R=R,
             Q=Q,
             ground_stations=np.array(ground_stations, dtype=object))
    
    print(f"Generated measurements:")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Time span: {df['tSec'].iloc[-1] - df['tSec'].iloc[0]:.1f} seconds")
    print(f"  Stations: {n_stations}")
    print(f"  Angle noise: {np.degrees(angle_noise_std):.3f} degrees")
    print(f"  Saved to: {output_npz}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_fgo_data.py <truth_csv> <output_npz>")
        sys.exit(1)
    
    generate_measurements(sys.argv[1], sys.argv[2])