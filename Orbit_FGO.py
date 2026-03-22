import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
from math import pi, sqrt, ceil, atan2, sin, cos, exp
import matplotlib.pyplot as plt


def gaussian_impulse(t, t_star, epsilon):
    """Gaussian approximation to Dirac delta (Eq. 17 from Zhang et al.)"""
    return (1.0 / (epsilon * sqrt(2 * pi))) * exp(-(t - t_star)**2 / (2 * epsilon**2))


def gaussian_impulse_dt_star(t, t_star, epsilon):
    """Derivative of Gaussian impulse w.r.t. t_star: dg/dt* = g * (t - t*) / epsilon^2"""
    g = gaussian_impulse(t, t_star, epsilon)
    return g * (t - t_star) / (epsilon**2)


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
                 use_range: bool = True,
                 meas_per_station: int = None,
                 manoeuvres=None,
                 epsilon: float = 0.5):
        
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
        
        self.prop_dt = self.dt
        self.n_timesteps = 1

        self.GE = 3.986004418e14
        self.J2 = 1.08262668e-3
        self.R_earth = 6378137.0
        self.omega_earth = 7.2921159e-5

        self.meas = meas
        self.S_Q_inv = la.inv(la.cholesky(Q))
        
        # Handle R matrix for different measurement types
        # TODO: Confirm if needed, or just remove
        if self.use_range and R.shape[0] == 2:
            # Extend R matrix for range measurements
            R_extended = np.eye(3)
            R_extended[:2, :2] = R
            R_extended[2, 2] = 100.0**2  # TODO: 100m range noise (ADJUSTABLE)
            self.S_R_inv = la.inv(la.cholesky(R_extended))
            print(f"Extended R matrix for range with 100m std dev")
        else:
            self.S_R_inv = la.inv(la.cholesky(R))

        self.states = np.zeros((self.N, 6))
        if x0 is not None:
            self.states[0] = x0

        # Gaussian impulse manoeuvre parameters
        self.epsilon = epsilon
        if manoeuvres is not None and len(manoeuvres) > 0:
            self.manoeuvres = manoeuvres
            self.n_manoeuvres = len(manoeuvres)
        else:
            self.manoeuvres = []
            self.n_manoeuvres = 0
        self.n_man_params = 3 * self.n_manoeuvres  # 3 delta-v components per manoeuvre
        # Current delta-v estimates stored as flat array [dvx0, dvy0, dvz0, dvx1, ...]
        self.man_params = np.zeros(self.n_man_params)
        for j, man in enumerate(self.manoeuvres):
            self.man_params[3*j:3*j+3] = man['delta_v']

        self.create_init_state()

    def create_init_state(self):
        for i in range(1, self.N):
            t_start = (i - 1) * self.dt
            self.states[i] = self.prop_one_timestep(self.states[i-1], t_start)

    def orbital_dynamics(self, state, t=None):
        r = state[:3]
        v = state[3:]
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

        # Gaussian impulse acceleration (Eq. 21)
        if t is not None and self.n_manoeuvres > 0:
            for j in range(self.n_manoeuvres):
                dv_j = self.man_params[3*j:3*j+3]
                t_star_j = self.manoeuvres[j]['t_star']
                g = gaussian_impulse(t, t_star_j, self.epsilon)
                a_total += dv_j * g

        return np.concatenate([v, a_total])

    def _needs_substep(self, t_start, t_end):
        """Check if interval overlaps with any manoeuvre's 3-sigma Gaussian window."""
        for man in self.manoeuvres:
            t_star = man['t_star']
            window_lo = t_star - 3 * self.epsilon
            window_hi = t_star + 3 * self.epsilon
            if t_start < window_hi and t_end > window_lo:
                return True
        return False

    def _rk4_step(self, state, t, dt):
        """Single RK4 step with time-aware dynamics."""
        k1 = self.orbital_dynamics(state, t)
        k2 = self.orbital_dynamics(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.orbital_dynamics(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.orbital_dynamics(state + dt * k3, t + dt)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def prop_one_timestep(self, state, t_start=None):
        """Propagate state by one timestep using RK4 integration.

        When t_start is provided and manoeuvres exist, uses adaptive sub-stepping
        near manoeuvre epochs to resolve the narrow Gaussian pulse.
        """
        going_out = state.copy()
        dt = self.prop_dt

        for step_i in range(self.n_timesteps):
            if t_start is not None:
                t_current = t_start + step_i * dt
            else:
                t_current = None

            if t_current is not None and self.n_manoeuvres > 0 and \
               self._needs_substep(t_current, t_current + dt):
                # Adaptive sub-stepping: use small steps to resolve Gaussian
                sub_dt = self.epsilon / 5.0
                n_sub = int(ceil(dt / sub_dt))
                sub_dt = dt / n_sub  # Exact division
                t_sub = t_current
                for _ in range(n_sub):
                    going_out = self._rk4_step(going_out, t_sub, sub_dt)
                    t_sub += sub_dt
            elif t_current is not None:
                going_out = self._rk4_step(going_out, t_current, dt)
            else:
                # No time info: original behaviour (no Gaussian term)
                k1 = self.orbital_dynamics(going_out)
                k2 = self.orbital_dynamics(going_out + 0.5 * dt * k1)
                k3 = self.orbital_dynamics(going_out + 0.5 * dt * k2)
                k4 = self.orbital_dynamics(going_out + dt * k3)
                going_out = going_out + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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

    def man_param_col_start(self):
        """Column index where manoeuvre parameters begin in the Jacobian."""
        return 6 * self.N

    def man_dv_col(self, j):
        """Column index for j-th manoeuvre's delta-v block."""
        return self.man_param_col_start() + 3 * j

    def F_man_mat(self, state, t_start):
        """Jacobian of propagated state w.r.t. manoeuvre delta-v parameters.

        Returns 6 x (3*n_manoeuvres) matrix via finite differences.
        """
        if self.n_manoeuvres == 0:
            return np.zeros((6, 0))

        eps = 1e-4
        F_man = np.zeros((6, self.n_man_params))
        f0 = self.prop_one_timestep(state, t_start)

        saved_params = self.man_params.copy()
        for j in range(self.n_man_params):
            self.man_params[j] += eps
            f_plus = self.prop_one_timestep(state, t_start)
            F_man[:, j] = (f_plus - f0) / eps
            self.man_params[j] = saved_params[j]

        return F_man

    def F_mat(self, state, t_start=None):
        eps = 1e-4
        F = np.zeros((6, 6))

        f0 = self.prop_one_timestep(state, t_start)

        for j in range(6):
            state_plus = state.copy()
            state_plus[j] += eps
            f_plus = self.prop_one_timestep(state_plus, t_start)

            F[:, j] = (f_plus - f0) / eps

        return F

    def create_L(self):
        n_meas = 3 if self.use_range else 2
        H_size = n_meas * 6
        F_size = 36
        F_man_size = 6 * self.n_man_params  # entries per manoeuvre Jacobian block
        nnz_entries = (2 * F_size + F_man_size) * (self.N - 1) + H_size * self.N * self.n_stations
        data_l = np.zeros(nnz_entries)
        row_l = np.zeros(nnz_entries, dtype=int)
        col_l = np.zeros(nnz_entries, dtype=int)
        t_e = 0

        n_rows = 6 * (self.N - 1) + self.N * self.n_stations * self.meas_per_station
        n_cols = 6 * self.N + self.n_man_params

        for i in range(1, self.N):
            t_start = (i - 1) * self.dt
            mat1 = self.S_Q_inv @ self.F_mat(self.states[i-1], t_start)
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat1, self.dyn_idx(i), self.state_idx(i-1))
            t_e += F_size

            mat2 = -self.S_Q_inv
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat2, self.dyn_idx(i), self.state_idx(i))
            t_e += F_size

            # Manoeuvre Jacobian columns
            if self.n_manoeuvres > 0:
                mat_man = self.S_Q_inv @ self.F_man_mat(self.states[i-1], t_start)
                data_l[t_e:t_e+F_man_size], row_l[t_e:t_e+F_man_size], col_l[t_e:t_e+F_man_size] = \
                    dense_2_sp_lists(mat_man, self.dyn_idx(i), self.man_param_col_start())
                t_e += F_man_size

        for i in range(self.N):
            t = i * self.dt
            for s_idx in range(self.n_stations):
                mat = self.S_R_inv @ self.H_mat(self.states[i], s_idx, t)
                row_offset = self.meas_idx(i) + s_idx * self.meas_per_station
                data_l[t_e:t_e+H_size], row_l[t_e:t_e+H_size], col_l[t_e:t_e+H_size] = \
                    dense_2_sp_lists(mat, row_offset, self.state_idx(i))
                t_e += H_size

        return sp.csr_matrix((data_l[:t_e], (row_l[:t_e], col_l[:t_e])),
                             shape=(n_rows, n_cols))

    def create_y(self, state_vec=None):
        saved_man_params = None
        if state_vec is not None:
            state_data = self.vec_to_data(state_vec)
            # Temporarily set manoeuvre params from the augmented vector
            if self.n_man_params > 0:
                saved_man_params = self.man_params.copy()
                self.man_params = state_vec[self.N*6:self.N*6+self.n_man_params]
        else:
            state_data = self.states
        
        y = np.zeros(6 * (self.N - 1) + self.N * self.n_stations * self.meas_per_station)
        
        for i in range(1, self.N):
            t_start = (i - 1) * self.dt
            pred_meas = self.prop_one_timestep(state_data[i-1], t_start) - state_data[i]
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

        # Restore manoeuvre params if we temporarily changed them
        if saved_man_params is not None:
            self.man_params = saved_man_params

        return y

    def vec_to_data(self, vec):
        going_out = np.zeros((self.N, 6))
        for i in range(self.N):
            going_out[i] = vec[i*6:(i+1)*6]
        return going_out

    def add_delta(self, delta_x: np.array = None) -> np.array:
        total_len = self.N * 6 + self.n_man_params
        going_out = np.zeros(total_len)
        if delta_x is None:
            delta_x = np.zeros(total_len)
        for i in range(self.N):
            going_out[i*6:(i+1)*6] = self.states[i] + delta_x[i*6:(i+1)*6]
        # Append manoeuvre parameters + their deltas
        if self.n_man_params > 0:
            going_out[self.N*6:] = self.man_params + delta_x[self.N*6:]
        return going_out

    def update_state(self, delta_x):
        for i in range(self.N):
            self.states[i] += delta_x[i*6:(i+1)*6]
        # Update manoeuvre parameters
        if self.n_man_params > 0:
            self.man_params += delta_x[self.N*6:self.N*6+self.n_man_params]

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
            print(f'\nOptimisation finished after {num_iters} iterations')
            print(f'Final cost: {best_cost:.2e}')

            if self.n_manoeuvres > 0:
                print(f'\nEstimated Manoeuvre Parameters:')
                for j in range(self.n_manoeuvres):
                    dv = self.man_params[3*j:3*j+3]
                    t_star = self.manoeuvres[j]['t_star']
                    print(f'  Manoeuvre {j+1}:')
                    print(f'    Delta-v: [{dv[0]:.4f}, {dv[1]:.4f}, {dv[2]:.4f}] m/s')
                    print(f'    (t* fixed at {t_star:.2f} s)')

