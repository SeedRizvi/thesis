#!/usr/bin/env python3
"""Fixed-lambda sweep with diag(M) damping, 50-iteration budget.

cond_test showed the un-damped GN step has ||dx|| ~ 5e3 while a diag-damped
step at lam=1e-6 has ||dx|| ~ 3.7 and achieves a far larger cost reduction.
Variant B still failed because its lambda decayed to 1e-10 and the damping
vanished. So: hold lambda FIXED and find the useful magnitude.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time, json
from multiprocessing import Pool
import numpy as np, numpy.linalg as la, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from Orbit_FGO import eci_to_ric
from truth_cache import load_truth
from tier0_diag import InstrumentedFGO, build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"; ARCSEC = 1/3600
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 7, 8]
LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
# converged reference from the 300-iteration run
REF = {1: 112.53, 7: 94.05, 8: 120.54}


class FixedLamFGO(InstrumentedFGO):
    def opt_fixed(self, max_iters, lam, truth, damping='diag'):
        stalled, num_iters = 0, 0
        curve, finished = [], False
        while not finished:
            L = self.create_L(); y = self.create_y()
            cost = float(y.T @ y); M = L.T @ L
            if damping == 'diag':
                d = M.diagonal(); d = np.maximum(d, 1e-12*d.max())
                A = M + lam * sp.diags(d)
            else:
                A = M + lam * sp.eye(M.shape[0])
            try:
                dx = spla.spsolve(A.tocsc(), L.T @ y)
            except Exception:
                break
            scale, best_scale, best_cost = 1.0, 0, cost
            for _ in range(20):
                try:
                    ny = self.create_y(self.add_delta(dx*scale)); nc = float(ny.T@ny)
                    if nc < best_cost: best_cost, best_scale = nc, scale
                    py = y - L @ (dx*scale); pc = float(py.T@py)
                    r = (cost-nc)/(cost-pc) if pc > 0 else 0
                    if r > 0.25 and nc < cost: best_scale = scale; break
                except Exception: pass
                scale *= 0.5
                if scale < 1e-10: break
            if best_scale > 0: self.update_state(dx*best_scale)
            yv = self.create_y(); n_dyn = 6*(self.N-1)
            e = self.states - truth
            curve.append({'iter': num_iters, 'cost': cost,
                          'cost_dyn': float(yv[:n_dyn]@yv[:n_dyn]),
                          'pos_rms': float(np.sqrt(np.mean(la.norm(e[:,:3],axis=1)**2))),
                          'scale': float(best_scale), 'dxnorm': float(la.norm(dx))})
            num_iters += 1
            stalled = stalled+1 if (cost-best_cost)/cost < 1e-6 else 0
            if best_scale > 0 and la.norm(dx*best_scale) < 1e-3: finished = True
            if num_iters >= max_iters or stalled >= 10: finished = True
        term = ('max_iters' if num_iters >= max_iters
                else 'stalled' if stalled >= 10 else 'step_norm')
        return curve, term


def job(a):
    seed, lam, damping = a
    truth, times, dt, dvr, dve, mst, ts = load_truth(CONFIG, "deltaRIC0.5")
    cp, gs = load_config_parameters(CONFIG)
    params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
              'q_vel_ric': np.array(cp['process_noise_vel'], float),
              'use_range': False, 'measurement_noise_deg': 2.0*ARCSEC,
              'range_noise_m': cp['range_noise_m'],
              'initial_pos_error': cp['initial_pos_error'],
              'initial_vel_error': cp['initial_vel_error'],
              'dv_initial_error': cp['dv_initial_error'],
              't_star_initial_error': cp['t_star_initial_error'],
              'epsilon': cp['epsilon'], 'max_iterations': 50}
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts, FixedLamFGO)
    t0 = time.perf_counter()
    curve, term = fgo.opt_fixed(50, lam, truth, damping)
    rt = time.perf_counter()-t0
    last = curve[-1]
    row = {'seed': seed, 'lam': lam, 'damping': damping, 'pos_rms': last['pos_rms'],
           'cost_dyn': last['cost_dyn'], 'termination': term, 'n_iters': len(curve),
           'runtime': rt, 'tstar_err': float(fgo.man_params[3]-ts),
           'median_scale': float(np.median([c['scale'] for c in curve])),
           'median_dxnorm': float(np.median([c['dxnorm'] for c in curve])),
           'ref': REF[seed]}
    print(f"seed {seed} {damping:>4} lam={lam:.0e}: pos_rms={last['pos_rms']:9.2f}m "
          f"(ref {REF[seed]:.1f}) dyn={last['cost_dyn']:.2e} {term:<10} "
          f"it={len(curve):>3} medscale={row['median_scale']:.4f} "
          f"med|dx|={row['median_dxnorm']:.2e} ({rt:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    jobs = [(s, l, 'diag') for s in SEEDS for l in LAMS]
    jobs += [(s, 1e-6, 'identity') for s in SEEDS]   # baseline-equivalent control
    with Pool(8) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(f"{OUT}/lam_sweep.json", "w"), indent=2, default=float)
    print("\n" + "="*80)
    print(f"{'lam':>8} " + "".join(f"{'seed '+str(s):>14}" for s in SEEDS))
    for l in LAMS:
        line = f"{l:>8.0e} "
        for s in SEEDS:
            r = next(x for x in rows if x['seed']==s and x['lam']==l and x['damping']=='diag')
            line += f"{r['pos_rms']:>13.1f}m"
        print(line)
    line = f"{'A base':>8} "
    for s in SEEDS:
        r = next(x for x in rows if x['seed']==s and x['damping']=='identity')
        line += f"{r['pos_rms']:>13.1f}m"
    print(line)
    print(f"{'ref@300':>8} " + "".join(f"{REF[s]:>13.1f}m" for s in SEEDS))
