import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as la
from math import pi, sqrt, ceil, atan2, sin, cos, exp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Finite-difference step sizes for numerical Jacobians
# -----------------------------------------------------------------------------
# FD_REL_STEP = sqrt(float64 machine epsilon). The theoretically optimal
# relative step size for forward differences (balances truncation vs roundoff).
FD_REL_STEP = 1e-8
# Floor for state-scaled steps, for components that are near-zero.
FD_STATE_FLOOR = 1e-6
# Perturbation for delta-v components in F_man_mat. dv is O(1) m/s, so 1e-4
# gives ~4 significant digits of FD accuracy.
FD_DV_STEP = 1e-4
# Perturbation for t* in F_man_mat. Sized to the Gaussian pulse width
# `epsilon` (config: manoeuvre_parameters/epsilon), not to t* itself.
# The FD step must shift the pulse by a measurable fraction of its shape.
# 0.01 s = 2% of epsilon=0.5s.
FD_TSTAR_STEP = 0.01


def fd_state_step(state_component: float) -> float:
    """Forward-difference step scaled to a state component's magnitude.

    Returns max(FD_REL_STEP * |x|, FD_STATE_FLOOR). For GEO states this gives
    ~0.4 m for position (O(1e7) m) and ~3e-5 m/s for velocity (O(1e3) m/s).
    """
    return max(FD_REL_STEP * abs(state_component), FD_STATE_FLOOR)


def eci_to_ric_rotation_matrix(state):
    """3x3 rotation matrix from ECI to RIC frame.
      R (radial)     = r / |r|
      C (cross-track) = (r x v) / |r x v|
      I (in-track)   = C x R
    """
    r, v = state[:3], state[3:]
    R_hat = r / la.norm(r)
    C_hat = np.cross(r, v)
    C_hat = C_hat / la.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    return np.array([R_hat, I_hat, C_hat])


def ric_to_eci(dv_ric, state):
    """Convert vector from RIC to ECI using the state's orbital frame."""
    T = eci_to_ric_rotation_matrix(state)
    return T.T @ dv_ric


def eci_to_ric(dv_eci, state):
    """Convert vector from ECI to RIC using the state's orbital frame."""
    T = eci_to_ric_rotation_matrix(state)
    return T @ dv_eci


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
    def __init__(self, meas: np.array, R: np.array,
                 q_pos_ric: np.array, q_vel_ric: np.array,
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
        self.q_pos_ric = np.array(q_pos_ric, dtype=float)
        self.q_vel_ric = np.array(q_vel_ric, dtype=float)
        
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
        self.n_man_params = 4 * self.n_manoeuvres  # 3 delta-v + 1 t* per manoeuvre
        # Manoeuvre params stored as flat array [dvx0, dvy0, dvz0, t*0, dvx1, ...]
        self.man_params = np.zeros(self.n_man_params)
        for j, man in enumerate(self.manoeuvres):
            self.man_params[4*j:4*j+3] = man['delta_v']
            self.man_params[4*j+3] = man['t_star']

        self.create_init_state()

    def compute_S_Q_inv(self, state):
        """Compute S_Q_inv = inv(chol(Q)) for a given state.

        Rotates the RIC position and velocity process noise into ECI
        using the orbital state, then creates the full 6x6 Q matrix.
        """
        T = eci_to_ric_rotation_matrix(state)
        Q_pos_eci = T.T @ np.diag(self.q_pos_ric) @ T
        Q_vel_eci = T.T @ np.diag(self.q_vel_ric) @ T
        Q = np.zeros((6, 6))
        Q[:3, :3] = Q_pos_eci
        Q[3:, 3:] = Q_vel_eci
        return la.inv(la.cholesky(Q))

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
                dv_j = self.man_params[4*j:4*j+3]
                t_star_j = self.man_params[4*j+3]
                g = gaussian_impulse(t, t_star_j, self.epsilon)
                a_total += dv_j * g

        return np.concatenate([v, a_total])

    def _needs_substep(self, t_start, t_end):
        """Check if interval overlaps with any manoeuvre's 3-sigma Gaussian window."""
        for j in range(self.n_manoeuvres):
            t_star = self.man_params[4*j+3]
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
        n_meas = 3 if self.use_range else 2
        H = np.zeros((n_meas, 6))

        meas0 = self.compute_measurements(state[:3],
                                         self.ground_stations[station_idx], t)

        for j in range(3):  # Only derivatives w.r.t position for measurements
            eps_j = fd_state_step(state[j])
            state_plus = state.copy()
            state_plus[j] += eps_j
            meas_plus = self.compute_measurements(state_plus[:3],
                                                 self.ground_stations[station_idx], t)

            # Azimuth with wrapping
            az_diff = meas_plus[0] - meas0[0]
            if az_diff > pi:
                az_diff -= 2 * pi
            elif az_diff < -pi:
                az_diff += 2 * pi

            H[0, j] = az_diff / eps_j
            H[1, j] = (meas_plus[1] - meas0[1]) / eps_j

            if self.use_range:
                H[2, j] = (meas_plus[2] - meas0[2]) / eps_j

        return H

    def man_param_col_start(self):
        """Column index where manoeuvre parameters begin in the Jacobian."""
        return 6 * self.N

    def man_dv_col(self, j):
        """Column index for j-th manoeuvre's delta-v block."""
        return self.man_param_col_start() + 4 * j

    def man_tstar_col(self, j):
        """Column index for j-th manoeuvre's t* parameter."""
        return self.man_param_col_start() + 4 * j + 3

    def F_man_mat(self, state, t_start):
        """Jacobian of propagated state w.r.t. manoeuvre parameters (dv, t*).

        Returns 6 x (4*n_manoeuvres) matrix via finite differences.
        Uses different step sizes for delta-v (1e-4 m/s) and t* (0.01 s).
        """
        if self.n_manoeuvres == 0:
            return np.zeros((6, 0))

        F_man = np.zeros((6, self.n_man_params))
        f0 = self.prop_one_timestep(state, t_start)

        saved_params = self.man_params.copy()
        for j in range(self.n_man_params):
            eps = FD_TSTAR_STEP if j % 4 == 3 else FD_DV_STEP
            self.man_params[j] += eps
            f_plus = self.prop_one_timestep(state, t_start)
            F_man[:, j] = (f_plus - f0) / eps
            self.man_params[j] = saved_params[j]

        return F_man

    def F_mat(self, state, t_start=None):
        F = np.zeros((6, 6))

        f0 = self.prop_one_timestep(state, t_start)

        for j in range(6):
            eps_j = fd_state_step(state[j])
            state_plus = state.copy()
            state_plus[j] += eps_j
            f_plus = self.prop_one_timestep(state_plus, t_start)

            F[:, j] = (f_plus - f0) / eps_j

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
            S_Q_inv_i = self.compute_S_Q_inv(self.states[i-1])
            mat1 = S_Q_inv_i @ self.F_mat(self.states[i-1], t_start)
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat1, self.dyn_idx(i), self.state_idx(i-1))
            t_e += F_size

            mat2 = -S_Q_inv_i
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat2, self.dyn_idx(i), self.state_idx(i))
            t_e += F_size

            # Manoeuvre Jacobian columns
            if self.n_manoeuvres > 0:
                mat_man = S_Q_inv_i @ self.F_man_mat(self.states[i-1], t_start)
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
            S_Q_inv_i = self.compute_S_Q_inv(state_data[i-1])
            y[self.dyn_idx(i):self.dyn_idx(i)+6] = S_Q_inv_i @ (-pred_meas)
        
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
        '''
        Build the Jacobian matrix (L) and residual vector (y) for the current
        state, solve the regularised normal equations for the best linear step
        that minimises y, and move in that direction. Repeat until convergence.
        (Levenberg-Marquardt damped Gauss-Newton with backtracking line search.)

        Backtracking line search: the dynamics are nonlinear, so trying
        scales 1, 1/2, 1/4, ... exploits the fact that smaller steps sit
        closer to the linearisation point, where the quadratic model is more
        accurate. The first scale whose actual cost reduction covers >= 25%
        of the predicted reduction is accepted.
        '''
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
                    dv = self.man_params[4*j:4*j+3]
                    t_star = self.man_params[4*j+3]
                    print(f'  Manoeuvre {j+1}:')
                    print(f'    Delta-v: [{dv[0]:.4f}, {dv[1]:.4f}, {dv[2]:.4f}] m/s')
                    print(f'    t* = {t_star:.2f} s (estimated)')

