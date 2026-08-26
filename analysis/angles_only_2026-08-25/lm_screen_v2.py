#!/usr/bin/env python3
"""LM variant screen, re-run on post-fix code (2026-08-26).

Supersedes lm_variants.py. That screen rejected B/C/D, but it ran under the
BROKEN stall counter (10 consecutive iterations below 1e-6, which killed any
damped variant at exactly iteration 10) and on the PRE-Q-fix geometry. Both
have since changed, so those rejections carry no weight for the current code.

All variants here use the CURRENT termination criteria, so the only thing
varying is the damping form, the lambda update, and the line search.

  A  current   lam*I        always shrink /2      line search
  B  diag      lam*diag(M)  always shrink /2      line search
  C  nielsenI  lam*I        gain-ratio (Nielsen)  line search   <- lambda rule ALONE
  D  nielsenD  lam*diag(M)  gain-ratio (Nielsen)  line search   <- the pairing TODO requires
  E  textbook  lam*diag(M)  gain-ratio (Nielsen)  none, L/y reused on reject
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

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"; TAG = "deltaRIC0.5"
OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1 / 3600
LAM_FLOOR, LAM_CEIL = 1e-10, 1e10

VARIANTS = {
    'A_current':  dict(damping='identity', rule='shrink',  ls=True),
    'B_diag':     dict(damping='diag',     rule='shrink',  ls=True),
    'C_nielsenI': dict(damping='identity', rule='nielsen', ls=True),
    'D_nielsenD': dict(damping='diag',     rule='nielsen', ls=True),
    'E_textbook': dict(damping='diag',     rule='nielsen', ls=False),
}


class VariantFGO(SatelliteOrbitFGO):

    def _damp(self, M, lam, damping):
        if damping == 'identity':
            return M + lam * sp.eye(M.shape[0])
        d = M.diagonal()
        d = np.maximum(d, 1e-12 * d.max())
        return M + lam * sp.diags(d)

    def opt_variant(self, max_iters, damping, rule, ls, lam0, truth):
        lam, nu, num_iters = lam0, 2.0, 0
        cost_history, log = [], []
        finished, rebuild = False, True
        L = y = M = Lty = None
        current_cost = None

        while not finished:
            if rebuild:
                L = self.create_L()
                y = self.create_y()
                current_cost = float(y.T @ y)
                M = L.T @ L
                Lty = L.T @ y
                rebuild = False

            try:
                delta_x = spla.spsolve(self._damp(M, lam, damping).tocsc(), Lty)
            except Exception:
                lam = min(lam * 10, LAM_CEIL * 10)
                num_iters += 1
                if num_iters >= max_iters or lam > LAM_CEIL:
                    finished = True
                continue

            lam_before = lam

            if ls:
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
            else:
                best_scale = 1.0
                try:
                    ny = self.create_y(self.add_delta(delta_x))
                    best_cost = float(ny.T @ ny)
                except Exception:
                    best_cost = np.inf
                if not (best_cost < current_cost):
                    best_scale, best_cost = 0, current_cost

            # gain ratio for the step actually taken
            rho, rel_pred_red = None, None
            if best_scale > 0:
                py = y - L @ (delta_x * best_scale)
                pc = float(py.T @ py)
                denom = current_cost - pc
                rho = (current_cost - best_cost) / denom if denom > 0 else None
                rel_pred_red = denom / current_cost
                self.update_state(delta_x * best_scale)
                rebuild = True
            else:
                rebuild = bool(ls)   # textbook LM reuses L/y on a rejected step

            # ---- lambda update ----
            if rule == 'shrink':
                if best_scale > 0:
                    lam = max(lam * 0.5, LAM_FLOOR)
                else:
                    lam *= 10
            else:  # nielsen
                if best_scale > 0 and rho is not None and rho > 0:
                    lam = max(lam * max(1.0 / 3.0, 1 - (2 * rho - 1) ** 3), LAM_FLOOR)
                    nu = 2.0
                else:
                    lam *= nu
                    nu *= 2

            num_iters += 1
            cost_history.append(best_cost)
            if len(cost_history) > STALL_WINDOW + 1:
                cost_history.pop(0)

            e = self.states - truth
            log.append({'iter': num_iters - 1, 'cost': current_cost,
                        'lam': lam_before, 'scale': float(best_scale),
                        'rho': float(rho) if rho is not None else None,
                        'pos_rms': float(np.sqrt(np.mean(
                            la.norm(e[:, :3], axis=1) ** 2)))})

            if rel_pred_red is not None and rel_pred_red < CONV_REL_PRED:
                finished = True
            if len(cost_history) == STALL_WINDOW + 1 and cost_history[0] > 0:
                if (cost_history[0] - best_cost) / cost_history[0] < STALL_REL_TOL:
                    finished = True
            if num_iters >= max_iters or lam > LAM_CEIL:
                finished = True

            if best_scale > 0:
                current_cost = best_cost

        term = ('max_iters' if num_iters >= max_iters else
                'lambda_blowup' if lam > LAM_CEIL else 'converged')
        return log, term


def job(a):
    noise, seed, vname, lam0, maxit = a
    v = VARIANTS[vname]
    truth, times, dt, dvr, dve, mst, ts = load_truth(CONFIG, TAG)
    cp, gs = load_config_parameters(CONFIG)
    params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
              'q_vel_ric': np.array(cp['process_noise_vel'], float),
              'use_range': False, 'measurement_noise_deg': noise * ARCSEC,
              'range_noise_m': cp['range_noise_m'],
              'initial_pos_error': cp['initial_pos_error'],
              'initial_vel_error': cp['initial_vel_error'],
              'dv_initial_error': cp['dv_initial_error'],
              't_star_initial_error': cp['t_star_initial_error'],
              'epsilon': cp['epsilon'], 'max_iterations': maxit}
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts, VariantFGO)
    t0 = time.perf_counter()
    log, term = fgo.opt_variant(maxit, v['damping'], v['rule'], v['ls'], lam0, truth)
    rt = time.perf_counter() - t0

    yv = fgo.create_y(); n_dyn = 6 * (fgo.N - 1)
    n_meas = fgo.N * fgo.n_stations * fgo.meas_per_station
    e = fgo.states - truth
    dv_ric = eci_to_ric(fgo.man_params[0:3], mst)
    row = {'noise': noise, 'seed': seed, 'variant': vname, 'lam0': lam0,
           'maxit': maxit,
           'pos_rms': float(np.sqrt(np.mean(la.norm(e[:, :3], axis=1) ** 2))),
           'cost_dyn': float(yv[:n_dyn] @ yv[:n_dyn]),
           'tstar_err': float(fgo.man_params[3] - ts),
           'dv_err': float(la.norm(dv_ric - dvr)),
           'n_iters': len(log), 'termination': term, 'runtime': rt,
           'lam_max': max(r['lam'] for r in log),
           'lam_final': log[-1]['lam'],
           'median_scale': float(np.median([r['scale'] for r in log])),
           'n_rejected': int(sum(1 for r in log if r['scale'] == 0))}
    print(f"{noise:>4.0f}\" s{seed:<3}{vname:<12} lam0={lam0:.0e} "
          f"rms={row['pos_rms']:9.2f}m dyn={row['cost_dyn']:.2e} "
          f"it={row['n_iters']:>3} lam_max={row['lam_max']:.1e} "
          f"rej={row['n_rejected']:>2} {term:<12} ({rt:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', type=int, nargs='*', default=[1, 3, 5, 6, 8, 10])
    ap.add_argument('--noises', type=float, nargs='*', default=[10.0, 2.0])
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--lam0', type=float, nargs='*', default=[1e-6])
    ap.add_argument('--maxit', type=int, default=50)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()

    jobs = [(n, s, v, l, a.maxit) for n in a.noises for s in a.seeds
            for v in a.variants for l in a.lam0]
    with Pool(a.workers) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(f"{OUT}/lmv2_{a.tag}.json", "w"), indent=2, default=float)

    print("\n" + "=" * 100)
    print(f"maxit={a.maxit}   pos RMS (m), '*' = diverged (cost_dyn > 1e3)")
    print("=" * 100)
    for n in a.noises:
        print(f"\n--- {n:.0f} arcsec ---")
        hdr = f"{'variant':<13}{'lam0':>8}" + "".join(f"{'s'+str(s):>12}" for s in a.seeds)
        print(hdr + f"{'div':>6}{'iters':>7}")
        for v in a.variants:
            for l in a.lam0:
                line = f"{v:<13}{l:>8.0e}"
                ndiv = 0; its = []
                for s in a.seeds:
                    r = next(x for x in rows if x['noise'] == n and x['seed'] == s
                             and x['variant'] == v and x['lam0'] == l)
                    d = r['cost_dyn'] > 1e3
                    ndiv += d
                    its.append(r['n_iters'])
                    line += f"{r['pos_rms']:>11.1f}" + ("*" if d else " ")
                print(line + f"{ndiv:>6}{int(np.mean(its)):>7}")
