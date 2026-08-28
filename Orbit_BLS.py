import numpy as np
import scipy.linalg as la
from Orbit_FGO import (SatelliteOrbitFGO, eci_to_ric_rotation_matrix,
                       fd_state_step, FD_DV_STEP, FD_TSTAR_STEP,
                       CONV_REL_PRED, STALL_WINDOW, STALL_REL_TOL)


class SatelliteOrbitBLS(SatelliteOrbitFGO):
    """Weighted Nonlinear Batch Least Squares orbit estimator.

    Inherits dynamics and measurement model from SatelliteOrbitFGO.
    Uses finite-difference to calculate Jacobians: perturb each of the n_solve parameters,
    re-propagate the full trajectory, and difference the residuals

    Solve vector: [x0(6), dv(3), t*(1)]  =  6 + 4M unknowns (M manoeuvres)
    """

    # Override so the parent __init__ does not pre-propagate all states
    def create_init_state(self):
        pass

    def __init__(self, meas, R, q_pos_ric, q_vel_ric, ground_stations,
                 dt=60.0, x0=None, P0=None, use_range=True, meas_per_station=None,
                 manoeuvres=None, epsilon=0.5):

        super().__init__(meas, R, q_pos_ric, q_vel_ric, ground_stations,
                         dt, x0, P0, use_range, meas_per_station, manoeuvres,
                         epsilon)

        # Reconstruct R covariance from parent's S_R_inv = inv(chol(R))
        L_R = la.inv(self.S_R_inv)
        self.R_matrix = L_R @ L_R.T
        self.R_inv = la.inv(self.R_matrix)

        # Solve-vector dimension: 6 (initial state) + 4 per manoeuvre
        self.n_solve = 6 + self.n_man_params


    # Propagate full trajectory from x0 + current man_params
    def _propagate_trajectory(self):
        """Propagate self.states[0] forward through all N time-steps."""
        for i in range(1, self.N):
            t_start = (i - 1) * self.dt
            self.states[i] = self.prop_one_timestep(self.states[i - 1], t_start)


    # Residual vector
    def _compute_residuals(self):
        """Compute weighted measurement residual vector.

        Returns the stacked residual  r = W^{1/2} (z - h(x))  for all
        time-steps and stations, where W = R^{-1}, followed by the prior
        residual  S_P0_inv (gamma - p)  on the solve vector.
        """
        from math import pi
        n_meas = self.meas_per_station
        total = self.N * self.n_stations * n_meas + self.n_solve
        residuals = np.zeros(total)

        for i in range(self.N):
            t = i * self.dt
            for s_idx in range(self.n_stations):
                z_pred = np.array(self.compute_measurements(
                    self.states[i, :3], self.ground_stations[s_idx], t))

                meas_start = i * self.n_stations * n_meas + s_idx * n_meas
                z_actual = self.meas[meas_start:meas_start + n_meas]

                innov = z_actual - z_pred
                # Azimuth wrapping
                if innov[0] > pi:
                    innov[0] -= 2 * pi
                elif innov[0] < -pi:
                    innov[0] += 2 * pi

                r_start = i * self.n_stations * n_meas + s_idx * n_meas
                residuals[r_start:r_start + n_meas] = self.S_R_inv @ innov

        p = np.concatenate([self.states[0], self.man_params])
        residuals[-self.n_solve:] = self.S_P0_inv @ (self.gamma - p)

        return residuals


    # Jacobian via finite differences
    def _compute_jacobian(self):
        """Jacobian of the weighted residual vector w.r.t. the solve vector.

        Uses forward finite differences: perturb each of the n_solve
        parameters, re-propagate the full trajectory, recompute residuals.
        """
        r0 = self._compute_residuals()
        n_res = len(r0)
        J = np.zeros((n_res, self.n_solve))

        # Save baseline
        x0_save = self.states[0].copy()
        man_save = self.man_params.copy() if self.n_man_params > 0 else None

        # Columns 0..5: initial state perturbations
        for j in range(6):
            eps_j = fd_state_step(x0_save[j])
            self.states[0] = x0_save.copy()
            self.states[0][j] += eps_j
            if self.n_man_params > 0:
                self.man_params[:] = man_save
            self._propagate_trajectory()
            r_pert = self._compute_residuals()
            J[:, j] = (r_pert - r0) / eps_j

        # Columns 6..: manoeuvre parameter perturbations
        for j in range(self.n_man_params):
            eps_j = FD_TSTAR_STEP if j % 4 == 3 else FD_DV_STEP
            self.states[0] = x0_save.copy()
            self.man_params[:] = man_save
            self.man_params[j] += eps_j
            self._propagate_trajectory()
            r_pert = self._compute_residuals()
            J[:, 6 + j] = (r_pert - r0) / eps_j

        # Restore baseline state
        self.states[0] = x0_save
        if self.n_man_params > 0:
            self.man_params[:] = man_save
        self._propagate_trajectory()

        return J, r0


    def run(self, max_iters=50, verbose=True):
        """Solve using Gauss-Newton with a backtracking line search.

        Matches Orbit_FGO.opt().
        """

        # Initial propagation from x0
        self._propagate_trajectory()

        lambda_reg = 1e-6
        finished = False
        num_iters = 0
        cost_history = []

        while not finished:
            J, r = self._compute_jacobian()
            current_cost = float(r @ r)

            if verbose:
                print(f'Iteration {num_iters}: cost = {current_cost:.2e}')

            # Column scaling (normalisation) to improve conditioning
            col_norms = np.sqrt(np.sum(J ** 2, axis=0))
            col_norms[col_norms < 1e-12] = 1.0
            D_inv = 1.0 / col_norms
            J_scaled = J * D_inv[np.newaxis, :]

            # Gauss-Newton normal equations: (Js^T Js) δps = Js^T r
            JtJ = J_scaled.T @ J_scaled
            Jtr = J_scaled.T @ r

            try:
                delta_p_scaled = la.solve(JtJ, Jtr, assume_a='pos')
            except la.LinAlgError:
                if verbose:
                    print(f'  Solver failed on iteration {num_iters}')
                # Undamped, so the retry is identical: bail rather than spin.
                finished = True
                continue

            # Un-scale
            delta_p = D_inv * delta_p_scaled

            # Backtracking line search
            x0_save = self.states[0].copy()
            man_save = self.man_params.copy() if self.n_man_params > 0 else None

            scale = 1.0
            best_scale = 0
            best_cost = current_cost

            for _ in range(20):
                try:
                    self.states[0] = x0_save - delta_p[:6] * scale
                    if self.n_man_params > 0:
                        self.man_params[:] = man_save - delta_p[6:] * scale
                    self._propagate_trajectory()
                    r_test = self._compute_residuals()
                    test_cost = float(r_test @ r_test)

                    if test_cost < best_cost:
                        best_cost = test_cost
                        best_scale = scale

                    # Accept if actual cost reduction covers >= 25% of predicted
                    pred_r = r - J @ (delta_p * scale)
                    pred_cost = float(pred_r @ pred_r)
                    if pred_cost > 0:
                        ratio = (current_cost - test_cost) / (current_cost - pred_cost)
                    else:
                        ratio = 0

                    if ratio > 0.25 and test_cost < current_cost:
                        break
                except Exception:
                    pass

                scale *= 0.5
                if scale < 1e-10:
                    break

            rel_pred_red = None
            if best_scale > 0:
                self.states[0] = x0_save - delta_p[:6] * best_scale
                if self.n_man_params > 0:
                    self.man_params[:] = man_save - delta_p[6:] * best_scale
                self._propagate_trajectory()

                lambda_reg = max(lambda_reg * 0.5, 1e-10)

                pred_r = r - J @ (delta_p * best_scale)
                rel_pred_red = ((current_cost - float(pred_r @ pred_r))
                                / current_cost)

                if verbose:
                    print(f'  delta norm: {la.norm(delta_p * best_scale):.2e}, scale: {best_scale:.3f}')
            else:
                # Restore and increase regularisation
                self.states[0] = x0_save
                if self.n_man_params > 0:
                    self.man_params[:] = man_save
                self._propagate_trajectory()
                lambda_reg *= 10
                if verbose:
                    print(f'  No improvement found, increasing regularisation to {lambda_reg:.2e}')

            num_iters += 1

            cost_history.append(best_cost)
            if len(cost_history) > STALL_WINDOW + 1:
                cost_history.pop(0)

            # The linear model has nothing left to promise.
            if rel_pred_red is not None and rel_pred_red < CONV_REL_PRED:
                finished = True

            # Or it keeps promising and never delivers.
            if len(cost_history) == STALL_WINDOW + 1 and cost_history[0] > 0:
                window_red = (cost_history[0] - best_cost) / cost_history[0]
                if window_red < STALL_REL_TOL:
                    finished = True

            if num_iters >= max_iters or lambda_reg > 1e10:
                finished = True

        if verbose:
            print(f'\nBLS finished after {num_iters} iterations')
            print(f'Final cost: {best_cost:.2e}')

            if self.n_manoeuvres > 0:
                print(f'\nEstimated Manoeuvre Parameters:')
                for j in range(self.n_manoeuvres):
                    dv = self.man_params[4*j:4*j+3]
                    t_star = self.man_params[4*j+3]
                    print(f'  Manoeuvre {j+1}:')
                    print(f'    Delta-v: [{dv[0]:.4f}, {dv[1]:.4f}, {dv[2]:.4f}] m/s')
                    print(f'    t* = {t_star:.2f} s (estimated)')
