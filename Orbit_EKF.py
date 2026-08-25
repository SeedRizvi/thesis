import numpy as np
import scipy.linalg as la
from math import pi
from Orbit_FGO import SatelliteOrbitFGO, eci_to_ric_rotation_matrix


def build_P0(n_manoeuvres, sigma_pos, sigma_vel,
             sigma_dv=None, sigma_tstar=None):
    """Create initial covariance matrix from standard deviations in config."""
    if n_manoeuvres > 0 and (sigma_dv is None or sigma_tstar is None):
        raise ValueError('sigma_dv and sigma_tstar are required when '
                         'n_manoeuvres > 0')

    n_aug = 6 + 4 * n_manoeuvres
    P0 = np.zeros((n_aug, n_aug))
    P0[:3, :3] = np.eye(3) * sigma_pos ** 2
    P0[3:6, 3:6] = np.eye(3) * sigma_vel ** 2
    for j in range(n_manoeuvres):
        b = 6 + 4 * j
        P0[b:b+3, b:b+3] = np.eye(3) * sigma_dv ** 2
        P0[b+3, b+3] = sigma_tstar ** 2
    return P0


class SatelliteOrbitEKF(SatelliteOrbitFGO):
    """Extended Kalman Filter orbit state estimator.

    Inherits dynamics, measurement model, and Jacobian computation from
    SatelliteOrbitFGO.  Replaces the batch factor-graph optimisation with
    a sequential predict/update loop
    """

    # Override so the parent __init__ does not pre-propagate all states
    def create_init_state(self):
        pass

    def __init__(self, meas, R, q_pos_ric, q_vel_ric, ground_stations,
                 dt=60.0, x0=None, P0=None, use_range=True,
                 meas_per_station=None, manoeuvres=None, epsilon=0.5):

        # Parent sets up physics, noise, manoeuvre params, constants, etc.
        super().__init__(meas, R, q_pos_ric, q_vel_ric, ground_stations,
                         dt, x0, P0, use_range, meas_per_station, manoeuvres,
                         epsilon)

        # Reconstruct R covariance from parent's S_R_inv = inv(chol(R))
        L_R = la.inv(self.S_R_inv)
        self.R_matrix = L_R @ L_R.T

        # Augmented state dimension
        self.n_aug = 6 + self.n_man_params

        # Initial covariance (validated by the parent)
        self.P0 = np.asarray(P0, dtype=float).copy()

        self.covariances = np.zeros((self.N, 6, 6))

    def compute_Q(self, state):
        """6x6 process noise covariance in ECI (rotated from RIC).
        q_pos_ric / q_vel_ric are per-axis standard deviations.
        """
        T = eci_to_ric_rotation_matrix(state)
        Q_pos_eci = T.T @ np.diag(self.q_pos_ric ** 2) @ T
        Q_vel_eci = T.T @ np.diag(self.q_vel_ric ** 2) @ T
        Q = np.zeros((6, 6))
        Q[:3, :3] = Q_pos_eci
        Q[3:, 3:] = Q_vel_eci
        return Q


    def run(self, verbose=True):
        """Execute the EKF over all timesteps."""
        n_aug = self.n_aug
        n_meas = self.meas_per_station

        # Initialise augmented state
        x = np.zeros(n_aug)
        x[:6] = self.states[0]
        if self.n_man_params > 0:
            x[6:] = self.man_params.copy()

        P = self.P0.copy()

        # Update for k=0
        x, P = self._update_at_timestep(x, P, 0)
        self.states[0] = x[:6]
        self.covariances[0] = P[:6, :6]

        for k in range(1, self.N):
            t_start = (k - 1) * self.dt
            t_k = k * self.dt

            # Predict
            # Sync manoeuvre params so inherited methods see current estimates
            if self.n_man_params > 0:
                self.man_params[:] = x[6:]

            # Propagate orbital state (manoeuvre params constant)
            x_pred = np.zeros(n_aug)
            x_pred[:6] = self.prop_one_timestep(x[:6], t_start)
            x_pred[6:] = x[6:]

            # Augmented state-transition Jacobian
            F_aug = np.eye(n_aug)
            F_aug[:6, :6] = self.F_mat(x[:6], t_start)
            if self.n_man_params > 0:
                F_aug[:6, 6:] = self.F_man_mat(x[:6], t_start)

            # Augmented process noise (zero for manoeuvre params)
            Q_aug = np.zeros((n_aug, n_aug))
            Q_aug[:6, :6] = self.compute_Q(x[:6])

            P_pred = F_aug @ P @ F_aug.T + Q_aug
            P_pred = 0.5 * (P_pred + P_pred.T)  # floating point stability to reliably compute inverse

            # Update
            x, P = self._update_at_timestep(x_pred, P_pred, k)

            self.states[k] = x[:6]
            self.covariances[k] = P[:6, :6]
            if self.n_man_params > 0:
                self.man_params[:] = x[6:]

            if verbose and (k % 50 == 0 or k == self.N - 1):
                pos_std = np.sqrt(np.trace(P[:3, :3]))
                print(f'  EKF step {k}/{self.N-1}: pos std = {pos_std:.1f} m')

        if verbose:
            print(f'\nEKF finished ({self.N} timesteps)')
            if self.n_manoeuvres > 0:
                print(f'Estimated Manoeuvre Parameters:')
                for j in range(self.n_manoeuvres):
                    dv = self.man_params[4*j:4*j+3]
                    t_star = self.man_params[4*j+3]
                    print(f'  Manoeuvre {j+1}:')
                    print(f'    Delta-v: [{dv[0]:.4f}, {dv[1]:.4f}, {dv[2]:.4f}] m/s')
                    print(f'    t* = {t_star:.2f} s (estimated)')

    def _update_at_timestep(self, x, P, k):
        """Sequential measurement update for all stations at timestep k."""
        n_aug = self.n_aug
        n_meas = self.meas_per_station
        t_k = k * self.dt

        x_upd = x.copy()
        P_upd = P.copy()

        for s_idx in range(self.n_stations):
            # Predicted measurement
            z_pred = np.array(self.compute_measurements(
                x_upd[:3], self.ground_stations[s_idx], t_k))

            # Actual measurement
            meas_start = k * self.n_stations * n_meas + s_idx * n_meas
            z_actual = self.meas[meas_start:meas_start + n_meas]

            # Innovation with azimuth wrapping
            innov = z_actual - z_pred
            if innov[0] > pi:
                innov[0] -= 2 * pi
            elif innov[0] < -pi:
                innov[0] += 2 * pi

            # Augmented measurement Jacobian (H_orb | 0)
            # Assumes measurements depend only on the satellite's orbital state, not on the manoeuvre parameters,
            # so their partial derivatives w.r.t. delta-v and t* are zero.
            H_orb = self.H_mat(x_upd[:6], s_idx, t_k)
            H_aug = np.zeros((n_meas, n_aug))
            H_aug[:, :6] = H_orb

            # Innovation covariance and Kalman gain
            S = H_aug @ P_upd @ H_aug.T + self.R_matrix
            K = P_upd @ H_aug.T @ la.inv(S)

            # State update
            x_upd = x_upd + K @ innov

            # Covariance update (Joseph form)
            I_KH = np.eye(n_aug) - K @ H_aug
            P_upd = I_KH @ P_upd @ I_KH.T + K @ self.R_matrix @ K.T
            P_upd = 0.5 * (P_upd + P_upd.T)  # floating point stability to reliably compute inverse

        return x_upd, P_upd
