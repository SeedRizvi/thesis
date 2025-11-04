import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
from math import pi, sqrt, ceil, atan2, sin, cos
import matplotlib.pyplot as plt


def dense_2_sp_lists(M: np.array, tl_row: int, tl_col: int, row_vec=True):
    data_list = M.flatten()
    if len(M.shape) == 2:
        rows, cols = M.shape
    elif len(M.shape) == 1:
        if row_vec:
            rows = 1
            cols = len(M)
        else:
            cols = 1
            rows = len(M)
    else:
        assert False, 'M must be 1d or 2d!'
    row_list = np.zeros(len(data_list))
    col_list = np.zeros(len(data_list))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            row_list[idx] = i + tl_row
            col_list[idx] = j + tl_col
    return (data_list, row_list, col_list)


class SatelliteOrbitFGO:
    def __init__(self, meas: np.array, R: np.array, Q: np.array,
                 ground_stations: list,
                 dt: float = 60.0,
                 x0: np.array = None,
                 use_range: bool = True,  # NEW: Enable range measurements
                 meas_per_station: int = None):  # NEW: Auto-detect measurement type
        
        self.ground_stations = ground_stations
        self.n_stations = len(ground_stations)
        self.use_range = use_range
        
        # Auto-detect measurement type based on data size
        if meas_per_station is None:
            total_meas = len(meas)
            # Try to figure out if we have 2 (az/el) or 3 (az/el/range) measurements
            if total_meas % (self.n_stations * 3) == 0:
                self.meas_per_station = 3
                self.use_range = True
                print("Detected 3 measurements per station (azimuth, elevation, range)")
            elif total_meas % (self.n_stations * 2) == 0:
                self.meas_per_station = 2
                self.use_range = False
                print("Detected 2 measurements per station (azimuth, elevation only)")
            else:
                raise ValueError(f"Cannot determine measurement type from data size {total_meas}")
        else:
            self.meas_per_station = meas_per_station
            
        self.N = len(meas) // (self.n_stations * self.meas_per_station)
        self.dt = dt
        
        if self.dt > 1:
            self.prop_dt = self.dt / ceil(self.dt)
            self.n_timesteps = int(ceil(self.dt))
        else:
            self.prop_dt = self.dt
            self.n_timesteps = 1

        self.GE = 3.986004418e14
        self.J2 = 1.08262668e-3
        self.R_earth = 6378137.0
        self.omega_earth = 7.2921159e-5

        self.meas = meas
        self.S_Q_inv = la.inv(la.cholesky(Q))
        
        # Handle R matrix for different measurement types
        if self.use_range and R.shape[0] == 2:
            # Extend R matrix for range measurements
            R_extended = np.eye(3)
            R_extended[:2, :2] = R
            R_extended[2, 2] = 100.0**2  # 100m range noise (adjust as needed)
            self.S_R_inv = la.inv(la.cholesky(R_extended))
            print(f"Extended R matrix for range with 100m std dev")
        else:
            self.S_R_inv = la.inv(la.cholesky(R))

        self.states = np.zeros((self.N, 6))
        if x0 is not None:
            self.states[0] = x0
        
        self.create_init_state()

    def create_init_state(self):
        for i in range(1, self.N):
            self.states[i] = self.prop_one_timestep(self.states[i-1])

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

    def state_idx(self, i: int) -> int:
        return 6 * i

    def dyn_idx(self, i: int) -> int:
        return 6 * (i - 1)

    def meas_idx(self, i: int) -> int:
        return (self.N - 1) * 6 + i * self.n_stations * self.meas_per_station

    def compute_measurements(self, r_sat_eci, r_station_llh, t):
        """Compute azimuth, elevation, and optionally range"""
        lat, lon, alt = r_station_llh
        
        theta = self.omega_earth * t
        
        R_ecef_to_eci = np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        
        r_station_ecef = np.array([
            (self.R_earth + alt) * cos(lat) * cos(lon),
            (self.R_earth + alt) * cos(lat) * sin(lon),
            (self.R_earth + alt) * sin(lat)
        ])
        
        r_station_eci = R_ecef_to_eci @ r_station_ecef
        
        r_rel_eci = r_sat_eci - r_station_eci
        
        # Compute range
        range_val = la.norm(r_rel_eci)
        
        R_eci_to_ecef = R_ecef_to_eci.T
        r_rel_ecef = R_eci_to_ecef @ r_rel_eci
        
        R_ecef_to_enu = np.array([
            [-sin(lon), cos(lon), 0],
            [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)],
            [cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]
        ])
        
        r_enu = R_ecef_to_enu @ r_rel_ecef
        e, n, u = r_enu
        
        range_horiz = sqrt(e**2 + n**2)
        azimuth = atan2(e, n)
        elevation = atan2(u, range_horiz)
        
        if self.use_range:
            return azimuth, elevation, range_val
        else:
            return azimuth, elevation

    def H_mat(self, state, station_idx, t):
        """Compute measurement Jacobian with optional range"""
        eps = 1e-4
        n_meas = 3 if self.use_range else 2
        H = np.zeros((n_meas, 6))
        
        meas0 = self.compute_measurements(state[:3], 
                                         self.ground_stations[station_idx], t)
        
        for j in range(3):  # Only derivatives w.r.t position for measurements
            state_plus = state.copy()
            state_plus[j] += eps
            meas_plus = self.compute_measurements(state_plus[:3], 
                                                 self.ground_stations[station_idx], t)
            
            # Azimuth with wrapping
            az_diff = meas_plus[0] - meas0[0]
            if az_diff > pi:
                az_diff -= 2 * pi
            elif az_diff < -pi:
                az_diff += 2 * pi
            
            H[0, j] = az_diff / eps
            H[1, j] = (meas_plus[1] - meas0[1]) / eps
            
            if self.use_range:
                H[2, j] = (meas_plus[2] - meas0[2]) / eps
        
        return H

    def F_mat(self, state):
        eps = 1e-4
        F = np.zeros((6, 6))
        
        f0 = self.prop_one_timestep(state)
        
        for j in range(6):
            state_plus = state.copy()
            state_plus[j] += eps
            f_plus = self.prop_one_timestep(state_plus)
            
            F[:, j] = (f_plus - f0) / eps
        
        return F

    def create_L(self):
        n_meas = 3 if self.use_range else 2
        H_size = n_meas * 6
        F_size = 36
        nnz_entries = 2 * F_size * (self.N - 1) + H_size * self.N * self.n_stations
        data_l = np.zeros(nnz_entries)
        row_l = np.zeros(nnz_entries, dtype=int)
        col_l = np.zeros(nnz_entries, dtype=int)
        t_e = 0
        
        for i in range(1, self.N):
            mat1 = self.S_Q_inv @ self.F_mat(self.states[i-1])
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat1, self.dyn_idx(i), self.state_idx(i-1))
            t_e += F_size
            
            mat2 = -self.S_Q_inv
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat2, self.dyn_idx(i), self.state_idx(i))
            t_e += F_size
        
        for i in range(self.N):
            t = i * self.dt
            for s_idx in range(self.n_stations):
                mat = self.S_R_inv @ self.H_mat(self.states[i], s_idx, t)
                row_offset = self.meas_idx(i) + s_idx * self.meas_per_station
                data_l[t_e:t_e+H_size], row_l[t_e:t_e+H_size], col_l[t_e:t_e+H_size] = \
                    dense_2_sp_lists(mat, row_offset, self.state_idx(i))
                t_e += H_size
        
        return sp.csr_matrix((data_l, (row_l, col_l)))

    def create_y(self, state_vec=None):
        if state_vec is not None:
            state_data = self.vec_to_data(state_vec)
        else:
            state_data = self.states
        
        y = np.zeros(6 * (self.N - 1) + self.N * self.n_stations * self.meas_per_station)
        
        for i in range(1, self.N):
            pred_meas = self.prop_one_timestep(state_data[i-1]) - state_data[i]
            y[self.dyn_idx(i):self.dyn_idx(i)+6] = self.S_Q_inv @ (-pred_meas)
        
        for i in range(self.N):
            t = i * self.dt
            for s_idx in range(self.n_stations):
                meas_pred = self.compute_measurements(
                    state_data[i, :3], self.ground_stations[s_idx], t
                )
                
                meas_start = i * self.n_stations * self.meas_per_station + s_idx * self.meas_per_station
                
                if self.use_range:
                    az_meas = self.meas[meas_start]
                    el_meas = self.meas[meas_start + 1]
                    range_meas = self.meas[meas_start + 2]
                    
                    az_diff = az_meas - meas_pred[0]
                    if az_diff > pi:
                        az_diff -= 2 * pi
                    elif az_diff < -pi:
                        az_diff += 2 * pi
                    
                    residual = np.array([az_diff, el_meas - meas_pred[1], 
                                       range_meas - meas_pred[2]])
                else:
                    az_meas = self.meas[meas_start]
                    el_meas = self.meas[meas_start + 1]
                    
                    az_diff = az_meas - meas_pred[0]
                    if az_diff > pi:
                        az_diff -= 2 * pi
                    elif az_diff < -pi:
                        az_diff += 2 * pi
                    
                    residual = np.array([az_diff, el_meas - meas_pred[1]])
                
                y_start = self.meas_idx(i) + s_idx * self.meas_per_station
                y[y_start:y_start+len(residual)] = self.S_R_inv @ residual
        
        return y

    def vec_to_data(self, vec):
        going_out = np.zeros((self.N, 6))
        for i in range(self.N):
            going_out[i] = vec[i*6:(i+1)*6]
        return going_out

    def add_delta(self, delta_x: np.array = None) -> np.array:
        going_out = np.zeros(self.N * 6)
        if delta_x is None:
            delta_x = np.zeros(self.N * 6)
        for i in range(self.N):
            going_out[i*6:(i+1)*6] = self.states[i] + delta_x[i*6:(i+1)*6]
        return going_out

    def update_state(self, delta_x):
        for i in range(self.N):
            self.states[i] += delta_x[i*6:(i+1)*6]

    def opt(self, max_iters=50, verbose=True):
        finished = False
        num_iters = 0
        lambda_reg = 1e-6
        
        while not finished:
            L = self.create_L()
            y = self.create_y()
            current_cost = float(y.T @ y)
            
            if verbose:
                print(f'Iteration {num_iters}: cost = {current_cost:.2e}')
            
            M = L.T @ L
            M_reg = M + lambda_reg * sp.eye(M.shape[0])
            Lty = L.T @ y
            
            try:
                delta_x = spla.spsolve(M_reg, Lty)
            except:
                if verbose:
                    print(f'Solver failed, increasing regularization')
                lambda_reg *= 10
                continue
            
            scale = 1.0
            best_scale = 0
            best_cost = current_cost
            
            for _ in range(20):
                try:
                    test_state = self.add_delta(delta_x * scale)
                    next_y = self.create_y(test_state)
                    next_cost = float(next_y.T @ next_y)
                    
                    if next_cost < best_cost:
                        best_cost = next_cost
                        best_scale = scale
                    
                    pred_y = y - L @ (delta_x * scale)
                    pred_cost = float(pred_y.T @ pred_y)
                    
                    if pred_cost > 0:
                        ratio = (current_cost - next_cost) / (current_cost - pred_cost)
                    else:
                        ratio = 0
                    
                    if ratio > 0.25 and next_cost < current_cost:
                        best_scale = scale
                        break
                    
                except Exception as e:
                    pass
                
                scale *= 0.5
                if scale < 1e-10:
                    break
            
            if best_scale > 0:
                self.update_state(delta_x * best_scale)
                
                if best_cost < current_cost * 0.9:
                    lambda_reg = max(lambda_reg * 0.5, 1e-10)
                
                if verbose:
                    print(f'  delta norm: {la.norm(delta_x * best_scale):.2e}, scale: {best_scale:.3f}')
            else:
                lambda_reg *= 10
                if verbose:
                    print(f'  No improvement found, increasing regularization to {lambda_reg:.2e}')
            
            num_iters += 1
            
            if la.norm(delta_x * best_scale) < 1e-3 or num_iters >= max_iters:
                finished = True
            
            if best_cost >= current_cost * 0.999 and num_iters > 5:
                if verbose:
                    print(f'Converged: no significant improvement')
                break
        
        if verbose:
            print(f'\nOptimization finished after {num_iters} iterations')
            print(f'Final cost: {best_cost:.2e}')


def generate_test_data_with_range(use_range=True, range_noise_m=100.0):
    """Generate test data with optional range measurements"""
    np.random.seed(42)
    
    ground_stations = [
        (np.deg2rad(40.7128), np.deg2rad(-74.0060), 0),   # New York
        (np.deg2rad(51.5074), np.deg2rad(-0.1278), 0),    # London
        (np.deg2rad(35.6762), np.deg2rad(139.6503), 0),   # Tokyo
        (np.deg2rad(-33.8688), np.deg2rad(151.2093), 0)   # Sydney
    ]
    
    r0 = 42164000  # GEO radius in meters
    inclination = np.deg2rad(5)
    
    x0 = np.array([
        r0 * cos(inclination),
        0,
        r0 * sin(inclination),
        0,
        sqrt(3.986004418e14 / r0) * cos(inclination),
        0
    ])
    
    dt = 60.0
    N = 100
    
    Q = np.eye(6) * 1e-4
    Q[:3, :3] *= 100
    Q[3:, 3:] *= 0.01
    
    if use_range:
        # 3x3 R matrix for azimuth, elevation, range
        R = np.eye(3)
        R[0, 0] = (np.deg2rad(0.01))**2  # Azimuth noise
        R[1, 1] = (np.deg2rad(0.01))**2  # Elevation noise
        R[2, 2] = range_noise_m**2        # Range noise
    else:
        # 2x2 R matrix for azimuth, elevation only
        R = np.eye(2) * (np.deg2rad(0.01))**2
    
    # Generate truth trajectory
    fgo_truth = SatelliteOrbitFGO(
        np.zeros(N * len(ground_stations) * (3 if use_range else 2)), 
        R, Q, ground_stations, dt, x0, use_range=use_range
    )
    truth = fgo_truth.states.copy()
    
    # Generate measurements
    meas = []
    for i in range(N):
        t = i * dt
        for station in ground_stations:
            measurements = fgo_truth.compute_measurements(truth[i, :3], station, t)
            
            if use_range:
                az, el, rng = measurements
                az += np.random.normal(0, np.deg2rad(0.01))
                el += np.random.normal(0, np.deg2rad(0.01))
                rng += np.random.normal(0, range_noise_m)
                meas.extend([az, el, rng])
            else:
                az, el = measurements
                az += np.random.normal(0, np.deg2rad(0.01))
                el += np.random.normal(0, np.deg2rad(0.01))
                meas.extend([az, el])
    
    meas = np.array(meas)
    
    return meas, truth, dt, R, Q, ground_stations


def compare_with_without_range():
    """Compare FGO performance with and without range measurements"""
    
    print("="*70)
    print("Comparing FGO: Angular-Only vs Angular+Range")
    print("="*70)
    
    results = {}
    
    for use_range in [False, True]:
        print(f"\n{'='*70}")
        print(f"Testing with range measurements: {use_range}")
        print(f"{'='*70}")
        
        # Generate test data
        meas, truth, dt, R, Q, ground_stations = generate_test_data_with_range(use_range)
        
        # Add noise to initial state
        x0 = truth[0].copy()
        x0[:3] += np.random.normal(0, 1000.0, 3)  # 1km position error
        x0[3:] += np.random.normal(0, 1.0, 3)      # 1m/s velocity error
        
        print(f"\nInitial errors:")
        print(f"  Position: {np.linalg.norm(x0[:3] - truth[0,:3]):.1f} m")
        print(f"  Velocity: {np.linalg.norm(x0[3:] - truth[0,3:]):.3f} m/s")
        
        # Run FGO
        fgo = SatelliteOrbitFGO(meas, R, Q, ground_stations, dt, x0=x0, use_range=use_range)
        
        print(f"\nRunning optimization...")
        fgo.opt(max_iters=30, verbose=False)
        
        # Compute errors
        errors = fgo.states - truth
        pos_errors = np.linalg.norm(errors[:, :3], axis=1)
        vel_errors = np.linalg.norm(errors[:, 3:], axis=1)
        
        results[use_range] = {
            'pos_rms': np.sqrt(np.mean(pos_errors**2)),
            'pos_max': np.max(pos_errors),
            'vel_rms': np.sqrt(np.mean(vel_errors**2)),
            'vel_max': np.max(vel_errors),
            'errors': errors,
            'pos_errors': pos_errors,
            'vel_errors': vel_errors,
            'states': fgo.states,
            'truth': truth
        }
        
        print(f"\nResults:")
        print(f"  Position RMS: {results[use_range]['pos_rms']:.2f} m")
        print(f"  Position Max: {results[use_range]['pos_max']:.2f} m")
        print(f"  Velocity RMS: {results[use_range]['vel_rms']:.4f} m/s")
        print(f"  Velocity Max: {results[use_range]['vel_max']:.4f} m/s")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<20} {'Angular-Only':<20} {'Angular+Range':<20} {'Improvement':<20}")
    print("-"*80)
    
    metrics = [
        ('Position RMS (m)', 'pos_rms', 1),
        ('Position Max (m)', 'pos_max', 1),
        ('Velocity RMS (m/s)', 'vel_rms', 1),
        ('Velocity Max (m/s)', 'vel_max', 1)
    ]
    
    for label, key, scale in metrics:
        without = results[False][key]
        with_range = results[True][key]
        improvement = (without - with_range) / without * 100
        print(f"{label:<20} {without*scale:<20.2f} {with_range*scale:<20.2f} {improvement:<19.1f}%")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Position errors over time
    ax = axes[0, 0]
    ax.plot(results[False]['pos_errors'], 'r-', label='Angular-Only', alpha=0.7)
    ax.plot(results[True]['pos_errors'], 'b-', label='Angular+Range', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity errors over time
    ax = axes[0, 1]
    ax.plot(results[False]['vel_errors'], 'r-', label='Angular-Only', alpha=0.7)
    ax.plot(results[True]['vel_errors'], 'b-', label='Angular+Range', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Velocity Error (m/s)')
    ax.set_title('Velocity Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # XY trajectory comparison - Angular Only
    ax = axes[0, 2]
    truth = results[False]['truth']
    states = results[False]['states']
    ax.plot(truth[:, 0]/1e6, truth[:, 1]/1e6, 'g-', label='Truth', linewidth=2)
    ax.plot(states[:, 0]/1e6, states[:, 1]/1e6, 'r--', label='Angular-Only', alpha=0.7)
    ax.set_xlabel('X (Mm)')
    ax.set_ylabel('Y (Mm)')
    ax.set_title('Angular-Only Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # XY trajectory comparison - With Range
    ax = axes[1, 0]
    truth = results[True]['truth']
    states = results[True]['states']
    ax.plot(truth[:, 0]/1e6, truth[:, 1]/1e6, 'g-', label='Truth', linewidth=2)
    ax.plot(states[:, 0]/1e6, states[:, 1]/1e6, 'b--', label='Angular+Range', alpha=0.7)
    ax.set_xlabel('X (Mm)')
    ax.set_ylabel('Y (Mm)')
    ax.set_title('Angular+Range Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Error histogram comparison
    ax = axes[1, 1]
    bins = np.linspace(0, max(results[False]['pos_max'], results[True]['pos_max']), 30)
    ax.hist(results[False]['pos_errors'], bins=bins, alpha=0.5, label='Angular-Only', color='red')
    ax.hist(results[True]['pos_errors'], bins=bins, alpha=0.5, label='Angular+Range', color='blue')
    ax.set_xlabel('Position Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Improvement bar chart
    ax = axes[1, 2]
    metrics_short = ['Pos RMS', 'Pos Max', 'Vel RMS', 'Vel Max']
    improvements = [
        (results[False]['pos_rms'] - results[True]['pos_rms']) / results[False]['pos_rms'] * 100,
        (results[False]['pos_max'] - results[True]['pos_max']) / results[False]['pos_max'] * 100,
        (results[False]['vel_rms'] - results[True]['vel_rms']) / results[False]['vel_rms'] * 100,
        (results[False]['vel_max'] - results[True]['vel_max']) / results[False]['vel_max'] * 100
    ]
    bars = ax.bar(metrics_short, improvements)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement with Range')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color bars based on improvement
    for bar, imp in zip(bars, improvements):
        if imp > 50:
            bar.set_color('green')
        elif imp > 25:
            bar.set_color('yellow')
        else:
            bar.set_color('orange')
    
    plt.suptitle('Factor Graph Optimization: Impact of Range Measurements', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fgo_range_comparison.png', dpi=150)
    print(f"\nComparison plot saved to: fgo_range_comparison.png")
    
    return results


if __name__ == '__main__':
    # Run the comparison
    results = compare_with_without_range()
    plt.show()
