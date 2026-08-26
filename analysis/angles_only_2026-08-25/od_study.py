#!/usr/bin/env python3
"""Isolation study: pure orbit determination without manoeuvre estimation.

Motivation: t* enters the cost through ~20 of 19,898 residual rows (0.1%) but
displaces 87% of the trajectory, and accounts for ~44 m of the ~53 m error
budget (ANGLES_ONLY_FGO.md 4.2b). The OD core cannot be characterised while one
weakly-observed parameter dominates the error. The original implementation was
validated with angles+range; removing range removed the only direct observation
of the range dimension, so base OD needs re-validating.

  Phase 1  RIC0   FGO-B   the angles-only OD floor (no manoeuvre anywhere)
  Phase 2  RIC0   FGO-G   cost of estimating a manoeuvre that does not exist
  Phase 3  RIC0.5 FGO-B   unmodelled manoeuvre (deliberate mis-specification)
  Phase 3b RIC0.5 FGO-G   for completeness

Runs at maxit=300 and extracts the 50-iteration result from the recorded curve,
so both budgets cost one run.

NOTE ON DIVERGENCE: cost_dyn > 1e3 is NOT a divergence test in Phase 3 -- a huge
dynamics cost is the correct answer to an unmodelled 0.866 m/s burn. Raw
quantities are recorded and the criterion is applied per-phase in analysis.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, json, argparse
from multiprocessing import Pool
import numpy as np, numpy.linalg as la
import scipy.sparse as sp, scipy.sparse.linalg as spla

sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from Orbit_FGO import (SatelliteOrbitFGO, eci_to_ric,
                       CONV_REL_PRED, STALL_WINDOW, STALL_REL_TOL)
from truth_cache import load_truth
from tier0_diag import build_seed

OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1 / 3600
CONFIGS = {'RIC0':   "configs/config_geo_one_rev_deltaRIC0.yml",
           'RIC0.5': "configs/config_geo_one_rev_deltaRIC0.5.yml"}


def rms(v):
    return float(np.sqrt(np.mean(v ** 2)))


class CurveFGO(SatelliteOrbitFGO):
    """Mirrors the shipped opt() exactly, recording pos RMS per iteration."""

    def opt_curve(self, max_iters, truth, tstar_idx, damping=True):
        finished, num_iters, lambda_reg = False, 0, 1e-6
        cost_history, curve = [], []
        while not finished:
            L = self.create_L()
            y = self.create_y()
            current_cost = float(y.T @ y)
            M = L.T @ L
            A = M + lambda_reg * sp.eye(M.shape[0]) if damping else M
            try:
                delta_x = spla.spsolve(A, L.T @ y)
            except Exception:
                lambda_reg *= 10
                continue

            scale, best_scale, best_cost = 1.0, 0, current_cost
            for _ in range(20):
                try:
                    ny = self.create_y(self.add_delta(delta_x * scale))
                    nc = float(ny.T @ ny)
                    if nc < best_cost:
                        best_cost, best_scale = nc, scale
                    py = y - L @ (delta_x * scale)
                    pc = float(py.T @ py)
                    r = ((current_cost - nc) / (current_cost - pc)
                         if pc > 0 and current_cost > pc else 0)
                    if r > 0.25 and nc < current_cost:
                        break
                except Exception:
                    pass
                scale *= 0.5
                if scale < 1e-10:
                    break

            rel_pred_red = None
            if best_scale > 0:
                self.update_state(delta_x * best_scale)
                lambda_reg = max(lambda_reg * 0.5, 1e-10)
                py = y - L @ (delta_x * best_scale)
                rel_pred_red = (current_cost - float(py.T @ py)) / current_cost
            else:
                lambda_reg *= 10

            num_iters += 1
            cost_history.append(best_cost)
            if len(cost_history) > STALL_WINDOW + 1:
                cost_history.pop(0)

            e = self.states - truth
            pe = la.norm(e[:, :3], axis=1)
            curve.append({'iter': num_iters - 1, 'cost': current_cost,
                          'pos_rms': rms(pe), 'scale': float(best_scale)})

            if rel_pred_red is not None and rel_pred_red < CONV_REL_PRED:
                finished = True
            if len(cost_history) == STALL_WINDOW + 1 and cost_history[0] > 0:
                if (cost_history[0] - best_cost) / cost_history[0] < STALL_REL_TOL:
                    finished = True
            if num_iters >= max_iters or lambda_reg > 1e10:
                finished = True

        term = ('max_iters' if num_iters >= max_iters else
                'lambda_blowup' if lambda_reg > 1e10 else 'converged')
        return curve, term


def job(a):
    cfgname, mode, noise, seed, maxit, damping = a
    cfg = CONFIGS[cfgname]
    truth, times, dt, dvr, dve, mst, ts = load_truth(cfg, 'delta' + cfgname)
    cp, gs = load_config_parameters(cfg)
    params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
              'q_vel_ric': np.array(cp['process_noise_vel'], float),
              'use_range': False, 'measurement_noise_deg': noise * ARCSEC,
              'range_noise_m': cp['range_noise_m'],
              'initial_pos_error': cp['initial_pos_error'],
              'initial_vel_error': cp['initial_vel_error'],
              'dv_initial_error': cp['dv_initial_error'],
              't_star_initial_error': cp['t_star_initial_error'],
              'epsilon': cp['epsilon'], 'max_iterations': maxit}
    fgo, _, pos_err0 = build_seed(seed, truth, times, dt, gs, params,
                                  dve, mst, ts, CurveFGO, mode=mode)
    tidx = int(round(ts / dt))
    e0 = fgo.states - truth
    pos_rms_init = rms(la.norm(e0[:, :3], axis=1))

    t0 = time.perf_counter()
    curve, term = fgo.opt_curve(maxit, truth, tidx, damping=damping)
    rt = time.perf_counter() - t0

    yv = fgo.create_y(); n_dyn = 6 * (fgo.N - 1)
    n_meas = fgo.N * fgo.n_stations * fgo.meas_per_station
    e = fgo.states - truth
    pe = la.norm(e[:, :3], axis=1); ve = la.norm(e[:, 3:], axis=1)
    row = {'config': cfgname, 'mode': mode, 'noise': noise, 'seed': seed,
           'damping': damping,
           'pos_rms': rms(pe), 'vel_rms': rms(ve),
           'pos_rms_pre':  rms(pe[:tidx + 1]),
           'pos_rms_post': rms(pe[tidx + 1:]),
           'pos_rms_init': pos_rms_init,
           'pos_rms_at50': curve[min(49, len(curve) - 1)]['pos_rms'],
           'iters_at50_capped': min(50, len(curve)),
           'cost_dyn': float(yv[:n_dyn] @ yv[:n_dyn]),
           'cost_meas': float(yv[n_dyn:n_dyn + n_meas] @ yv[n_dyn:n_dyn + n_meas]),
           'cost_prior': float(yv[-fgo.n_prior:] @ yv[-fgo.n_prior:]),
           'n_iters': len(curve), 'termination': term, 'runtime': rt,
           'improved': bool(rms(pe) < pos_rms_init)}
    if mode == 'FGO-G' and fgo.n_manoeuvres > 0:
        dv_ric = eci_to_ric(fgo.man_params[0:3], mst)
        row['dv_err'] = float(la.norm(dv_ric - dvr))
        row['tstar_err'] = float(fgo.man_params[3] - ts)
    print(f"{cfgname:<7}{mode:<7}{noise:>4.0f}\" s{seed:<3} "
          f"rms={row['pos_rms']:9.2f}m (@50 {row['pos_rms_at50']:9.2f}) "
          f"pre={row['pos_rms_pre']:8.2f} post={row['pos_rms_post']:9.2f} "
          f"it={row['n_iters']:>3} {term:<10} ({rt:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--phases', nargs='*', default=['1', '2', '3', '3b'])
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(1, 11)))
    ap.add_argument('--noises', type=float, nargs='*', default=[10.0, 2.0, 1.0])
    ap.add_argument('--maxit', type=int, default=300)
    ap.add_argument('--no-damping', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()

    PHASE = {'1': ('RIC0', 'FGO-B'), '2': ('RIC0', 'FGO-G'),
             '3': ('RIC0.5', 'FGO-B'), '3b': ('RIC0.5', 'FGO-G')}
    jobs = [(PHASE[p][0], PHASE[p][1], n, s, a.maxit, not a.no_damping)
            for p in a.phases for n in a.noises for s in a.seeds]
    with Pool(a.workers) as pool:
        rows = pool.map(job, jobs)
    json.dump(rows, open(f"{OUT}/od_{a.tag}.json", "w"), indent=2, default=float)

    print("\n" + "=" * 104)
    print("ISOLATION STUDY  (converged = terminated on its own, not max_iters)")
    print("=" * 104)
    for p in a.phases:
        cfgname, mode = PHASE[p]
        print(f"\n--- Phase {p}: {cfgname} / {mode} ---")
        print(f"{'noise':>7}{'conv':>7}{'mean':>10}{'median':>10}{'max':>11}"
              f"{'mean@50':>10}{'iters':>7}{'pre':>9}{'post':>10}")
        for n in a.noises:
            sel = [r for r in rows if r['config'] == cfgname and r['mode'] == mode
                   and r['noise'] == n]
            conv = [r for r in sel if r['termination'] == 'converged']
            use = conv if conv else sel
            print(f"{n:>6.0f}\"{len(conv):>5}/{len(sel):<2}"
                  f"{np.mean([r['pos_rms'] for r in use]):>10.1f}"
                  f"{np.median([r['pos_rms'] for r in use]):>10.1f}"
                  f"{np.max([r['pos_rms'] for r in use]):>11.1f}"
                  f"{np.mean([r['pos_rms_at50'] for r in use]):>10.1f}"
                  f"{np.mean([r['n_iters'] for r in sel]):>7.0f}"
                  f"{np.mean([r['pos_rms_pre'] for r in use]):>9.1f}"
                  f"{np.mean([r['pos_rms_post'] for r in use]):>10.1f}")
