#!/usr/bin/env python3
"""Both arms, 10 seeds, max_iters=300, angles-only. Records the full
convergence curve so a 50-iteration budget can be evaluated against the
converged answer from the same run."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, json
from multiprocessing import Pool
import numpy as np, numpy.linalg as la, scipy.sparse as sp, scipy.sparse.linalg as spla

sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from Orbit_FGO import eci_to_ric
from truth_cache import load_truth
from tier0_diag import InstrumentedFGO, build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"; TAG = "deltaRIC0.5"
OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1/3600
MAXIT = 300
CHECK = [25, 50, 100, 200]


class CurveFGO(InstrumentedFGO):
    def opt_curve(self, max_iters, truth):
        num_iters, lam, stalled = 0, 1e-6, 0
        curve = []
        finished = False
        while not finished:
            L = self.create_L(); y = self.create_y()
            current_cost = float(y.T @ y)
            M = L.T @ L
            try:
                delta_x = spla.spsolve(M + lam * sp.eye(M.shape[0]), L.T @ y)
            except Exception:
                lam *= 10; continue
            scale, best_scale, best_cost = 1.0, 0, current_cost
            for _ in range(20):
                try:
                    ny = self.create_y(self.add_delta(delta_x * scale))
                    nc = float(ny.T @ ny)
                    if nc < best_cost: best_cost, best_scale = nc, scale
                    py = y - L @ (delta_x * scale); pc = float(py.T @ py)
                    ratio = (current_cost - nc) / (current_cost - pc) if pc > 0 else 0
                    if ratio > 0.25 and nc < current_cost:
                        best_scale = scale; break
                except Exception: pass
                scale *= 0.5
                if scale < 1e-10: break
            if best_scale > 0:
                self.update_state(delta_x * best_scale)
                lam = max(lam * 0.5, 1e-10)
            else:
                lam *= 10
            yv = self.create_y(); n_dyn = 6 * (self.N - 1)
            e = self.states - truth
            curve.append({'iter': num_iters,
                          'cost': current_cost,
                          'cost_dyn': float(yv[:n_dyn] @ yv[:n_dyn]),
                          'pos_rms': float(np.sqrt(np.mean(la.norm(e[:, :3], axis=1) ** 2))),
                          'scale': float(best_scale),
                          'tstar': float(self.man_params[3])})
            num_iters += 1
            stalled = stalled + 1 if (current_cost - best_cost) / current_cost < 1e-6 else 0
            if best_scale > 0 and la.norm(delta_x * best_scale) < 1e-3: finished = True
            if num_iters >= max_iters or lam > 1e10 or stalled >= 10: finished = True
        term = ('max_iters' if num_iters >= max_iters else 'lambda_blowup'
                if lam > 1e10 else 'stalled' if stalled >= 10 else 'step_norm')
        return curve, term


def job(args):
    noise, seed = args
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
              'epsilon': cp['epsilon'], 'max_iterations': MAXIT}
    fgo, ts0, p0 = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts, CurveFGO)
    t0 = time.perf_counter()
    curve, term = fgo.opt_curve(MAXIT, truth)
    rt = time.perf_counter() - t0
    dv_ric = eci_to_ric(fgo.man_params[0:3], mst)
    conv = next((c['iter'] for c in curve if c['cost_dyn'] < 100), None)
    row = {'noise': noise, 'seed': seed, 'termination': term, 'n_iters': len(curve),
           'runtime': rt, 'pos_rms': curve[-1]['pos_rms'],
           'cost_dyn': curve[-1]['cost_dyn'],
           'tstar_err': float(fgo.man_params[3] - ts),
           'dv_err': float(la.norm(dv_ric - dvr)),
           'iters_to_dyn100': conv, 'curve': curve}
    for c in CHECK:
        i = min(c - 1, len(curve) - 1)
        row[f'pos_rms@{c}'] = curve[i]['pos_rms']
        row[f'cost_dyn@{c}'] = curve[i]['cost_dyn']
    print(f"{noise:>4.0f}\" seed {seed:>2}: final={row['pos_rms']:8.2f}m "
          f"@50={row['pos_rms@50']:9.2f}m dyn={row['cost_dyn']:.2e} "
          f"conv@{conv} it={len(curve)} {term} ({rt:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    jobs = [(n, s) for n in (10.0, 2.0) for s in range(1, 11)]
    with Pool(10) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(f"{OUT}/sweep300.json", "w"), indent=2, default=float)
    print("\nDONE")
