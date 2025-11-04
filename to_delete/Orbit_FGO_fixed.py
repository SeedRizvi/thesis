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
                 x0: np.array = None):
        self.ground_stations = ground_stations
        self.n_stations = len(ground_stations)
        self.N = len(meas) // (self.n_stations * 2)
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
        return (self.N - 1) * 6 + i * self.n_stations * 2

    def compute_azimuth_elevation(self, r_sat_eci, r_station_llh, t):
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
        
        return azimuth, elevation

    def H_mat(self, state, station_idx, t):
        eps = 1e-4
        H = np.zeros((2, 6))
        
        az0, el0 = self.compute_azimuth_elevation(state[:3], 
                                                 self.ground_stations[station_idx], t)
        
        for j in range(3):
            state_plus = state.copy()
            state_plus[j] += eps
            az_plus, el_plus = self.compute_azimuth_elevation(state_plus[:3], 
                                                             self.ground_stations[station_idx], t)
            
            az_diff = az_plus - az0
            if az_diff > pi:
                az_diff -= 2 * pi
            elif az_diff < -pi:
                az_diff += 2 * pi
            
            H[0, j] = az_diff / eps
            H[1, j] = (el_plus - el0) / eps
        
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
        H_size = 2 * 6
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
                row_offset = self.meas_idx(i) + s_idx * 2
                data_l[t_e:t_e+H_size], row_l[t_e:t_e+H_size], col_l[t_e:t_e+H_size] = \
                    dense_2_sp_lists(mat, row_offset, self.state_idx(i))
                t_e += H_size
        
        return sp.csr_matrix((data_l, (row_l, col_l)))

    def create_y(self, state_vec=None):
        if state_vec is not None:
            state_data = self.vec_to_data(state_vec)
        else:
            state_data = self.states
        
        y = np.zeros(6 * (self.N - 1) + self.N * self.n_stations * 2)
        
        for i in range(1, self.N):
            pred_meas = self.prop_one_timestep(state_data[i-1]) - state_data[i]
            y[self.dyn_idx(i):self.dyn_idx(i)+6] = self.S_Q_inv @ (-pred_meas)
        
        for i in range(self.N):
            t = i * self.dt
            for s_idx in range(self.n_stations):
                az_pred, el_pred = self.compute_azimuth_elevation(
                    state_data[i, :3], self.ground_stations[s_idx], t
                )
                
                meas_start = i * self.n_stations * 2 + s_idx * 2
                az_meas = self.meas[meas_start]
                el_meas = self.meas[meas_start + 1]
                
                az_diff = az_meas - az_pred
                if az_diff > pi:
                    az_diff -= 2 * pi
                elif az_diff < -pi:
                    az_diff += 2 * pi
                
                residual = np.array([az_diff, el_meas - el_pred])
                y_start = self.meas_idx(i) + s_idx * 2
                y[y_start:y_start+2] = self.S_R_inv @ residual
        
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


def generate_test_data(config_file='config_geo_realistic.yml'):
    """Generate test data with ground stations and measurements"""
    np.random.seed(42)
    
    # Define ground stations (lat, lon, alt in radians and meters)
    ground_stations = [
        (np.deg2rad(40.7128), np.deg2rad(-74.0060), 0),   # New York
        (np.deg2rad(51.5074), np.deg2rad(-0.1278), 0),    # London
        (np.deg2rad(35.6762), np.deg2rad(139.6503), 0),   # Tokyo
        (np.deg2rad(-33.8688), np.deg2rad(151.2093), 0)   # Sydney
    ]
    
    # Generate a GEO satellite orbit
    r0 = 42164000  # GEO radius in meters
    inclination = np.deg2rad(5)  # Small inclination
    
    # Initial state (position and velocity)
    x0 = np.array([
        r0 * cos(inclination),
        0,
        r0 * sin(inclination),
        0,
        sqrt(3.986004418e14 / r0) * cos(inclination),
        0
    ])
    
    # Simulation parameters
    dt = 60.0  # seconds
    N = 100    # number of timesteps
    
    # Process noise (for dynamics)
    Q = np.eye(6) * 1e-4
    Q[:3, :3] *= 100  # Position uncertainty
    Q[3:, 3:] *= 0.01  # Velocity uncertainty
    
    # Measurement noise (azimuth and elevation)
    R = np.eye(2) * (np.deg2rad(0.01))**2  # 0.01 degree measurement noise
    
    # Generate truth trajectory
    fgo_truth = SatelliteOrbitFGO(np.zeros(N * len(ground_stations) * 2), 
                                   R, Q, ground_stations, dt, x0)
    truth = fgo_truth.states.copy()
    
    # Generate measurements
    meas = []
    for i in range(N):
        t = i * dt
        for station in ground_stations:
            az, el = fgo_truth.compute_azimuth_elevation(truth[i, :3], station, t)
            az += np.random.normal(0, np.deg2rad(0.01))
            el += np.random.normal(0, np.deg2rad(0.01))
            meas.extend([az, el])
    
    meas = np.array(meas)
    
    # Save test data
    np.savez('geo_example', 
             meas=meas, 
             truth=truth, 
             dt=dt, 
             R=R, 
             Q=Q,
             ground_stations=ground_stations)
    
    return meas, truth, dt, R, Q, ground_stations


if __name__ == '__main__':
    # Try to load existing test data, or generate if not found
    try:
        prefix = 'geo_example'
        data = np.load(f'{prefix}.npz', allow_pickle=True)
        meas = data['meas']
        truth = data['truth']
        dt = float(data['dt'])
        R = data['R']
        Q = data['Q']
        if 'ground_stations' in data:
            ground_stations = [tuple(gs) for gs in data['ground_stations']]
        else:
            # Default ground stations if not in file
            ground_stations = [
                (np.deg2rad(40.7128), np.deg2rad(-74.0060), 0),
                (np.deg2rad(51.5074), np.deg2rad(-0.1278), 0),
                (np.deg2rad(35.6762), np.deg2rad(139.6503), 0),
                (np.deg2rad(-33.8688), np.deg2rad(151.2093), 0)
            ]
        print(f"Loaded test data from {prefix}.npz")
    except:
        print("Generating test data...")
        meas, truth, dt, R, Q, ground_stations = generate_test_data()
        print("Test data generated and saved")
    
    print("="*70)
    print("3D Factor Graph Optimization with J2 Perturbations")
    print("="*70)
    print(f"Simulation: {len(truth)} timesteps, {len(ground_stations)} ground stations")
    print(f"Timestep: {dt} seconds")
    
    # Add noise to initial state
    x0 = truth[0].copy()
    pos_error = 1000.0  # 1 km position error
    vel_error = 1.0     # 1 m/s velocity error
    x0[:3] += np.random.normal(0, pos_error, 3)
    x0[3:] += np.random.normal(0, vel_error, 3)
    
    print(f"\nInitial state error:")
    print(f"  Position: {np.linalg.norm(x0[:3] - truth[0,:3]):.1f} m")
    print(f"  Velocity: {np.linalg.norm(x0[3:] - truth[0,3:]):.3f} m/s")
    
    # Create and run optimizer
    fgo = SatelliteOrbitFGO(meas, R, Q, ground_stations, dt, x0=x0)
    
    print(f"\nBefore optimization:")
    errors_before = fgo.states - truth
    pos_rms_before = np.sqrt(np.mean(np.linalg.norm(errors_before[:, :3], axis=1)**2))
    vel_rms_before = np.sqrt(np.mean(np.linalg.norm(errors_before[:, 3:], axis=1)**2))
    print(f"  Position RMS: {pos_rms_before:.2f} m")
    print(f"  Velocity RMS: {vel_rms_before:.4f} m/s")
    
    print("\n" + "="*70)
    print("Running optimization...")
    print("="*70)
    fgo.opt(max_iters=50, verbose=True)
    
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    errors = fgo.states - truth
    pos_errors = np.linalg.norm(errors[:, :3], axis=1)
    vel_errors = np.linalg.norm(errors[:, 3:], axis=1)
    
    print(f"\nAfter optimization:")
    print(f"  Position RMS: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
    print(f"  Position Max: {np.max(pos_errors):.2f} m")
    print(f"  Velocity RMS: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
    print(f"  Velocity Max: {np.max(vel_errors):.4f} m/s")
    
    # Plotting
    fig = plt.figure(figsize=(16, 10))
    
    # 3D Trajectory
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(truth[:, 0]/1e6, truth[:, 1]/1e6, truth[:, 2]/1e6, 
             'r-', linewidth=2, label='Truth')
    ax1.plot(fgo.states[:, 0]/1e6, fgo.states[:, 1]/1e6, fgo.states[:, 2]/1e6, 
             'b--', linewidth=1, alpha=0.7, label='Estimated')
    ax1.set_xlabel('X (Mm)')
    ax1.set_ylabel('Y (Mm)')
    ax1.set_zlabel('Z (Mm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Position errors
    ax2 = fig.add_subplot(232)
    ax2.plot(errors[:, 0], label='X error', alpha=0.7)
    ax2.plot(errors[:, 1], label='Y error', alpha=0.7)
    ax2.plot(errors[:, 2], label='Z error', alpha=0.7)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Component Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Velocity errors
    ax3 = fig.add_subplot(233)
    ax3.plot(errors[:, 3], label='Vx error', alpha=0.7)
    ax3.plot(errors[:, 4], label='Vy error', alpha=0.7)
    ax3.plot(errors[:, 5], label='Vz error', alpha=0.7)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Velocity Error (m/s)')
    ax3.set_title('Velocity Component Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Total position error
    ax4 = fig.add_subplot(234)
    ax4.plot(pos_errors)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Position Error Magnitude (m)')
    ax4.set_title('Total Position Error')
    ax4.grid(True, alpha=0.3)
    
    # Total velocity error
    ax5 = fig.add_subplot(235)
    ax5.plot(vel_errors)
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Velocity Error Magnitude (m/s)')
    ax5.set_title('Total Velocity Error')
    ax5.grid(True, alpha=0.3)
    
    # Error statistics over time
    ax6 = fig.add_subplot(236)
    ax6.semilogy(np.abs(errors[:, :3]).max(axis=1), label='Max Pos Error', alpha=0.7)
    ax6.semilogy(np.abs(errors[:, 3:]).max(axis=1), label='Max Vel Error', alpha=0.7)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Maximum Absolute Error (log scale)')
    ax6.set_title('Error Evolution (log scale)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{prefix}_fgo_results.png', dpi=150)
    print(f"\nPlot saved to {prefix}_fgo_results.png")
    
    # Save results
    np.savez('fg_' + prefix + '_res', fg_res=fgo.states, truth=truth)
    print(f"Results saved to fg_{prefix}_res.npz")
    
    plt.show()
