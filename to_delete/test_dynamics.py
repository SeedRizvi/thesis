import numpy as np
import pandas as pd
import scipy.linalg as la

class Propagator2Body:
    def __init__(self, dt=60.0):
        self.GE = 3.986004418e14
        self.dt = dt
        
        if self.dt > 1:
            self.prop_dt = self.dt / np.ceil(self.dt)
            self.n_timesteps = int(np.ceil(self.dt))
        else:
            self.prop_dt = self.dt
            self.n_timesteps = 1
    
    def prop_one_timestep(self, state):
        going_out = state.copy()
        for _ in range(self.n_timesteps):
            r = going_out[:3]
            v = going_out[3:]
            r_norm = la.norm(r)
            
            a_total = -self.GE / (r_norm**3) * r
            
            v_new = v + a_total * self.prop_dt
            r_new = r + v * self.prop_dt + 0.5 * a_total * (self.prop_dt**2)
            
            going_out = np.concatenate([r_new, v_new])
        
        return going_out
    
    def propagate(self, x0, n_steps):
        states = np.zeros((n_steps, 6))
        states[0] = x0
        for i in range(1, n_steps):
            states[i] = self.prop_one_timestep(states[i-1])
        return states

class PropagatorJ2:
    def __init__(self, dt=60.0):
        self.GE = 3.986004418e14
        self.J2 = 1.082626683e-3
        self.R_earth = 6378137.0
        self.dt = dt
        
        if self.dt > 1:
            self.prop_dt = self.dt / np.ceil(self.dt)
            self.n_timesteps = int(np.ceil(self.dt))
        else:
            self.prop_dt = self.dt
            self.n_timesteps = 1
    
    def prop_one_timestep(self, state):
        going_out = state.copy()
        for _ in range(self.n_timesteps):
            r = going_out[:3]
            v = going_out[3:]
            r_norm = la.norm(r)
            
            a_2body = -self.GE / (r_norm**3) * r
            
            z2 = r[2]**2
            r2 = r_norm**2
            factor = 1.5 * self.J2 * self.GE * (self.R_earth**2) / (r_norm**5)
            a_J2 = factor * np.array([
                r[0] * (5 * z2 / r2 - 1),
                r[1] * (5 * z2 / r2 - 1),
                r[2] * (5 * z2 / r2 - 3)
            ])
            
            a_total = a_2body + a_J2
            
            v_new = v + a_total * self.prop_dt
            r_new = r + v * self.prop_dt + 0.5 * a_total * (self.prop_dt**2)
            
            going_out = np.concatenate([r_new, v_new])
        
        return going_out
    
    def propagate(self, x0, n_steps):
        states = np.zeros((n_steps, 6))
        states[0] = x0
        for i in range(1, n_steps):
            states[i] = self.prop_one_timestep(states[i-1])
        return states

def test_dynamics(truth_csv, test_type='j2'):
    print("="*70)
    print(f"DYNAMICS TEST: {test_type.upper()}")
    print("="*70)
    
    df = pd.read_csv(truth_csv)
    truth = np.zeros((len(df), 6))
    truth[:, 0] = df['x'].values
    truth[:, 1] = df['y'].values
    truth[:, 2] = df['z'].values
    truth[:, 3] = df['vx'].values
    truth[:, 4] = df['vy'].values
    truth[:, 5] = df['vz'].values
    
    dt = df['tSec'].iloc[1] - df['tSec'].iloc[0] if len(df) > 1 else 60.0
    
    print(f"\nTruth data:")
    print(f"  Timesteps: {len(truth)}")
    print(f"  dt: {dt} seconds")
    print(f"  Initial pos: {truth[0, :3]}")
    print(f"  Initial vel: {truth[0, 3:]}")
    print(f"  Initial radius: {np.linalg.norm(truth[0, :3])/1000:.1f} km")
    
    if test_type == '2body':
        prop = Propagator2Body(dt=dt)
    else:
        prop = PropagatorJ2(dt=dt)
    
    test_states = prop.propagate(truth[0], len(truth))
    
    pos_errors = np.linalg.norm(test_states[:, :3] - truth[:, :3], axis=1)
    vel_errors = np.linalg.norm(test_states[:, 3:] - truth[:, 3:], axis=1)
    
    print(f"\n{test_type.upper()} Propagator vs Truth:")
    print("="*70)
    print("Position Errors:")
    for idx, label in [(0, 't=0'), (10, 't=10min'), (60, 't=1hr'), (180, 't=3hr'), (360, 't=6hr'), (-1, 't=end')]:
        if idx == -1:
            idx = len(pos_errors) - 1
        if idx < len(pos_errors):
            print(f"  {label:8s}: {pos_errors[idx]:12.2e} m")
    print(f"  RMS:      {np.sqrt(np.mean(pos_errors**2)):12.2e} m")
    
    print("\nVelocity Errors:")
    for idx, label in [(0, 't=0'), (10, 't=10min'), (60, 't=1hr'), (180, 't=3hr'), (360, 't=6hr'), (-1, 't=end')]:
        if idx == -1:
            idx = len(vel_errors) - 1
        if idx < len(vel_errors):
            print(f"  {label:8s}: {vel_errors[idx]:12.2e} m/s")
    print(f"  RMS:      {np.sqrt(np.mean(vel_errors**2)):12.2e} m/s")
    
    rms_pos = np.sqrt(np.mean(pos_errors**2))
    rms_vel = np.sqrt(np.mean(vel_errors**2))
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    
    if rms_pos < 1.0:
        status = "✓ PERFECT MATCH"
        advice = "Dynamics models are identical"
    elif rms_pos < 100.0:
        status = "✓ EXCELLENT MATCH"
        advice = "Tiny differences, FGO will work well (<100m error)"
    elif rms_pos < 10000.0:
        status = "⚠ GOOD MATCH"
        advice = "Small differences, FGO should work (1-10km error)"
    elif rms_pos < 100000.0:
        status = "⚠ MODERATE MISMATCH"
        advice = "Noticeable differences, FGO may struggle (10-100km error)"
    else:
        status = "❌ CRITICAL MISMATCH"
        advice = "Large differences, FGO will fail"
    
    print(f"  Status: {status}")
    print(f"  {advice}")
    print(f"  Position RMS: {rms_pos/1000:.1f} km")
    print(f"  Velocity RMS: {rms_vel:.1f} m/s")
    
    if rms_pos > 1000:
        print("\n  Possible causes:")
        print("  1. orbDetHOUSE using different GM constant")
        print("  2. orbDetHOUSE using different J2 coefficient")
        print("  3. orbDetHOUSE using different Earth radius")
        print("  4. Numerical integration differences")
        print("  5. orbDetHOUSE still using higher-order terms despite config")
    
    print("="*70)
    return rms_pos, rms_vel

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_dynamics.py <truth_csv> [2body|j2]")
        print("Example: python test_dynamics.py out/out_propgeo_truth.csv j2")
        sys.exit(1)
    
    truth_file = sys.argv[1]
    test_type = sys.argv[2] if len(sys.argv) > 2 else 'j2'
    
    test_dynamics(truth_file, test_type)
