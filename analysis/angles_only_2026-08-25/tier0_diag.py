#!/usr/bin/env python3
"""
Tier 0 diagnostics for the angles-only 2-arcsec catastrophic failure.

Test 1: is t_star_true on the dt grid?
Test 2: M.diagonal() magnitudes by variable block + lambda trajectory,
        failing seed vs good seed, at 10" and 2", angles-only, FGO-G.

Pure instrumentation. Nothing in the repo is modified.
"""
import os
import sys
import time
import json

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, '/home/z5363026/thesis')
os.chdir('/home/z5363026/thesis')

from fgo_pipeline import (load_propagator_output, load_config_parameters,
                          simulate_measurements)
from Orbit_FGO import SatelliteOrbitFGO, eci_to_ric, ric_to_eci
from Orbit_EKF import build_P0
import mc_fgo

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"
TAG = "deltaRIC0.5"
OUT = "/tmp/claude-1000/-home-z5363026-thesis/c89cd938-7927-417e-bc71-878a4c3ad278/scratchpad"

ARCSEC = 1.0 / 3600.0


# ---------------------------------------------------------------------------
# Instrumented solver: identical logic to Orbit_FGO.opt(), plus logging.
# ---------------------------------------------------------------------------
class InstrumentedFGO(SatelliteOrbitFGO):

    def block_diag_stats(self, M):
        """M.diagonal() summarised per variable block."""
        d = M.diagonal()
        N = self.N
        pos_idx = np.concatenate([np.arange(6 * i, 6 * i + 3) for i in range(N)])
        vel_idx = np.concatenate([np.arange(6 * i + 3, 6 * i + 6) for i in range(N)])
        out = {
            'pos': d[pos_idx],
            'vel': d[vel_idx],
        }
        if self.n_man_params > 0:
            c0 = self.man_param_col_start()
            out['dv'] = d[c0:c0 + 3]
            out['tstar'] = d[c0 + 3:c0 + 4]
        return out

    def opt_logged(self, max_iters=50):
        finished = False
        num_iters = 0
        lambda_reg = 1e-6
        stalled = 0
        log = []
        diag_snapshot = None

        while not finished:
            L = self.create_L()
            y = self.create_y()
            current_cost = float(y.T @ y)

            M = L.T @ L
            if diag_snapshot is None:
                diag_snapshot = {k: (float(v.min()), float(np.median(v)),
                                     float(v.max()))
                                 for k, v in self.block_diag_stats(M).items()}

            M_reg = M + lambda_reg * sp.eye(M.shape[0])
            Lty = L.T @ y

            try:
                delta_x = spla.spsolve(M_reg, Lty)
            except Exception:
                lambda_reg *= 10
                continue

            scale = 1.0
            best_scale = 0
            best_cost = current_cost
            accepted_ratio = None

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
                        accepted_ratio = float(ratio)
                        break

                except Exception:
                    pass

                scale *= 0.5
                if scale < 1e-10:
                    break

            lam_before = lambda_reg
            if best_scale > 0:
                self.update_state(delta_x * best_scale)
                lambda_reg = max(lambda_reg * 0.5, 1e-10)
            else:
                lambda_reg *= 10

            # split the cost for diagnosis
            y_now = self.create_y()
            n_dyn = 6 * (self.N - 1)
            n_meas_rows = self.N * self.n_stations * self.meas_per_station
            cost_dyn = float(y_now[:n_dyn] @ y_now[:n_dyn])
            cost_meas = float(y_now[n_dyn:n_dyn + n_meas_rows]
                              @ y_now[n_dyn:n_dyn + n_meas_rows])
            cost_prior = float(y_now[-self.n_prior:] @ y_now[-self.n_prior:])

            log.append({
                'iter': num_iters,
                'cost': current_cost,
                'lambda_before': lam_before,
                'lambda_after': lambda_reg,
                'best_scale': float(best_scale),
                'ratio': accepted_ratio,
                'delta_norm': float(la.norm(delta_x * best_scale)) if best_scale > 0 else 0.0,
                'tstar': float(self.man_params[3]) if self.n_man_params else None,
                'cost_dyn': cost_dyn,
                'cost_meas': cost_meas,
                'cost_prior': cost_prior,
            })

            num_iters += 1

            if (current_cost - best_cost) / current_cost < 1e-6:
                stalled += 1
            else:
                stalled = 0

            if best_scale > 0 and la.norm(delta_x * best_scale) < 1e-3:
                finished = True

            if num_iters >= max_iters or lambda_reg > 1e10 or stalled >= 10:
                finished = True

        term = ('max_iters' if num_iters >= max_iters
                else 'lambda_blowup' if lambda_reg > 1e10
                else 'stalled' if stalled >= 10 else 'step_norm')
        return log, diag_snapshot, term


# ---------------------------------------------------------------------------
def build_seed(seed, truth_states, times, dt, ground_stations, params,
               delta_v_eci, manoeuvre_state, t_star_true, cls, mode='FGO-G'):
    """Replicates mc_fgo.run_fgo_seed's setup exactly (same RNG draw order).

    mode='FGO-B' skips the delta-v / t* draws, exactly as mc_fgo does, so the
    RNG stream stays aligned: FGO-B and FGO-G for the same seed therefore share
    identical measurements and identical x0, and are directly comparable.
    """
    np.random.seed(seed)

    measurements, R = simulate_measurements(
        truth_states, times, ground_stations,
        params['measurement_noise_deg'],
        use_range=params['use_range'],
        range_noise_m=params['range_noise_m'],
    )

    x0 = truth_states[0].copy()
    x0_err = np.zeros(6)
    x0_err[:3] = np.random.normal(0, params['initial_pos_error'], 3)
    x0_err[3:] = np.random.normal(0, params['initial_vel_error'], 3)
    x0 += x0_err

    if mode == 'FGO-G' and delta_v_eci is not None:
        dv_noise = np.random.normal(0, params['dv_initial_error'], 3)
        dv_guess = delta_v_eci + dv_noise
        t_star_guess_err = np.random.normal(0, params['t_star_initial_error'])
        t_star_guess = t_star_true + t_star_guess_err
        manoeuvres = [{'delta_v': dv_guess, 't_star': t_star_guess}]
    else:
        manoeuvres = None
        t_star_guess_err = None

    n_man = 0 if manoeuvres is None else len(manoeuvres)
    # p0_scale multiplies the PRIOR sigmas only, leaving the sampled error
    # untouched. p0_scale == 1 is the correct, self-consistent case: the prior
    # covariance equals the distribution the perturbation was drawn from.
    # Anything else deliberately mis-specifies the prior, which is the point of
    # the robustness arm of the study. A very large value (1e6) effectively
    # removes the prior, since S_P0_inv -> 0.
    ps = params.get('p0_scale', 1.0)
    P0 = build_P0(n_man,
                  params['initial_pos_error'] * ps,
                  params['initial_vel_error'] * ps,
                  None if params.get('dv_initial_error') is None
                  else params['dv_initial_error'] * ps,
                  None if params.get('t_star_initial_error') is None
                  else params['t_star_initial_error'] * ps)

    fgo = cls(measurements, R, params['q_pos_ric'], params['q_vel_ric'],
              ground_stations, dt, x0=x0, P0=P0,
              use_range=params['use_range'], manoeuvres=manoeuvres,
              epsilon=params['epsilon'])
    return fgo, t_star_guess_err, la.norm(x0_err[:3])


def main():
    print("=" * 78)
    print("TIER 0 DIAGNOSTICS  --  angles-only, FGO-G, deltaRIC0.5 one_rev")
    print("=" * 78)

    truth_data = mc_fgo.propagate_truth(CONFIG, TAG)
    (truth_states, times, dt, delta_v_ric, delta_v_eci,
     manoeuvre_state, t_star_true) = truth_data

    cp, ground_stations = load_config_parameters(CONFIG)

    # ---------------- TEST 1 ----------------
    print("\n" + "-" * 78)
    print("TEST 1: is t* on the dt grid?")
    print("-" * 78)
    print(f"  N steps         = {len(truth_states)}")
    print(f"  dt              = {dt} s")
    print(f"  t_star_true     = {t_star_true} s")
    print(f"  t_star_true/dt  = {t_star_true/dt}")
    print(f"  t_star_true%dt  = {t_star_true % dt}")
    on_grid = abs((t_star_true % dt)) < 1e-9 or abs((t_star_true % dt) - dt) < 1e-9
    print(f"  ON GRID         = {on_grid}")
    dvn = la.norm(delta_v_eci)
    print(f"  |delta_v|       = {dvn:.6f} m/s")
    print(f"  0.5*|dv|*dt     = {0.5*dvn*dt:.3f} m   <- predicted boundary floor")

    # ---------------- Q units ----------------
    print("\n" + "-" * 78)
    print("Q units check")
    print("-" * 78)
    qp = np.array(cp['process_noise_pos'], dtype=float)
    qv = np.array(cp['process_noise_vel'], dtype=float)
    print(f"  q_pos_ric              = {qp}")
    print(f"  as variance -> sigma   = {np.sqrt(qp)} m")
    print(f"  as sigma    -> sigma   = {qp} m")
    print(f"  calibration RMS (q/5)  = {qp/5} m")
    print(f"  q_vel_ric              = {qv}")
    print(f"  as variance -> sigma   = {np.sqrt(qv)} m/s")
    print(f"  pos inflation if squared-by-mistake = {np.sqrt(qp)/qp}")
    print(f"  vel inflation if squared-by-mistake = {np.sqrt(qv)/qv}")

    # ---------------- TEST 2 ----------------
    results = {}
    for noise_name, noise_arcsec in [("10arcsec", 10.0), ("2arcsec", 2.0)]:
        params = {
            'q_pos_ric': qp,
            'q_vel_ric': qv,
            'use_range': False,                     # ANGLES ONLY
            'measurement_noise_deg': noise_arcsec * ARCSEC,
            'range_noise_m': cp['range_noise_m'],
            'initial_pos_error': cp['initial_pos_error'],
            'initial_vel_error': cp['initial_vel_error'],
            'dv_initial_error': cp['dv_initial_error'],
            't_star_initial_error': cp['t_star_initial_error'],
            'epsilon': cp['epsilon'],
            'max_iterations': cp['max_iterations'],
        }

        for seed in (3, 4):
            key = f"{noise_name}_seed{seed}"
            print("\n" + "=" * 78)
            print(f"TEST 2: {key}  (angles-only, FGO-G, eps={params['epsilon']})")
            print("=" * 78)

            fgo, tstar_err0, pos_err0 = build_seed(
                seed, truth_states, times, dt, ground_stations, params,
                delta_v_eci, manoeuvre_state, t_star_true, InstrumentedFGO)

            print(f"  initial |x0 pos err| = {pos_err0:.1f} m")
            print(f"  initial t* err       = {tstar_err0:+.2f} s")

            t0 = time.perf_counter()
            log, diag, term = fgo.opt_logged(max_iters=params['max_iterations'])
            rt = time.perf_counter() - t0

            print(f"\n  M.diagonal() at iteration 0  (min / median / max):")
            for blk in ('pos', 'vel', 'dv', 'tstar'):
                if blk in diag:
                    lo, md, hi = diag[blk]
                    print(f"    {blk:>6}: {lo:.4e} / {md:.4e} / {hi:.4e}")
            tstar_diag = diag['tstar'][1]
            print(f"\n  t* diagonal = {tstar_diag:.4e}")

            lam_max = max(r['lambda_after'] for r in log)
            print(f"  lambda max reached = {lam_max:.4e}")
            print(f"  lambda/tstar_diag  = {lam_max/tstar_diag:.4e}"
                  f"   {'<-- LAMBDA DOMINATES t* COLUMN' if lam_max > tstar_diag else ''}")

            print(f"\n  iter |     cost   |  lambda  | scale  | ratio |"
                  f"  cost_dyn  | cost_meas |   t*")
            for r in log:
                rr = f"{r['ratio']:.2f}" if r['ratio'] is not None else "  -- "
                print(f"  {r['iter']:>4} | {r['cost']:.4e} | {r['lambda_before']:.2e} |"
                      f" {r['best_scale']:.4f} | {rr:>5} |"
                      f" {r['cost_dyn']:.3e} | {r['cost_meas']:.3e} |"
                      f" {r['tstar']:.2f}")

            errors = fgo.states - truth_states
            pos_rms = float(np.sqrt(np.mean(la.norm(errors[:, :3], axis=1) ** 2)))
            dv_est_ric = eci_to_ric(fgo.man_params[0:3], manoeuvre_state)
            dv_err = la.norm(dv_est_ric - delta_v_ric)
            tstar_err = float(fgo.man_params[3] - t_star_true)

            print(f"\n  --> termination = {term}, iters = {len(log)}, {rt:.1f}s")
            print(f"  --> pos RMS     = {pos_rms:.2f} m")
            print(f"  --> |dv| err    = {dv_err:.4f} m/s")
            print(f"  --> t* err      = {tstar_err:+.2f} s")

            results[key] = {
                'pos_rms': pos_rms, 'dv_err': dv_err, 'tstar_err': tstar_err,
                'termination': term, 'n_iters': len(log), 'runtime': rt,
                'lambda_max': lam_max, 'diag': diag,
                'tstar_err0': float(tstar_err0), 'pos_err0': float(pos_err0),
                'log': log,
            }

    with open(f"{OUT}/tier0_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'case':<20} {'pos_rms(m)':>12} {'t*_err(s)':>11} {'dv_err':>9} "
          f"{'term':>14} {'iters':>6}")
    for k, v in results.items():
        print(f"{k:<20} {v['pos_rms']:>12.2f} {v['tstar_err']:>11.2f} "
              f"{v['dv_err']:>9.4f} {v['termination']:>14} {v['n_iters']:>6}")


if __name__ == '__main__':
    main()
