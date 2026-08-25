#!/usr/bin/env python3
"""Measure candidate termination quantities per iteration so thresholds are
chosen from data, not guessed.

  rel_pred = (cost - pred_cost_at_accepted_scale)/cost   <- model's promised gain
  rel_act  = (cost - next_cost)/cost                     <- realised gain
  cum10    = cumulative relative gain over last 10 iters
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys, json
from multiprocessing import Pool
import numpy as np, numpy.linalg as la, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from truth_cache import load_truth
from tier0_diag import InstrumentedFGO, build_seed
CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"; ARCSEC = 1/3600
OUT = os.path.dirname(os.path.abspath(__file__))


class ProbeFGO(InstrumentedFGO):
    def probe(self, max_iters, truth):
        lam, n, stalled, rows = 1e-6, 0, 0, []
        hist = []
        finished = False
        while not finished:
            L = self.create_L(); y = self.create_y()
            cost = float(y.T @ y); M = L.T @ L
            try:
                dx = spla.spsolve(M + lam*sp.eye(M.shape[0]), L.T @ y)
            except Exception:
                lam *= 10; continue
            scale, best_scale, best_cost = 1.0, 0, cost
            for _ in range(20):
                try:
                    ny = self.create_y(self.add_delta(dx*scale)); nc = float(ny.T@ny)
                    if nc < best_cost: best_cost, best_scale = nc, scale
                    py = y - L @ (dx*scale); pc = float(py.T@py)
                    r = (cost-nc)/(cost-pc) if pc > 0 else 0
                    if r > 0.25 and nc < cost: break
                except Exception: pass
                scale *= 0.5
                if scale < 1e-10: break
            if best_scale > 0:
                py = y - L @ (dx*best_scale); pc = float(py.T@py)
                rel_pred = (cost - pc)/cost
                self.update_state(dx*best_scale)
                lam = max(lam*0.5, 1e-10)
            else:
                rel_pred = 0.0
                lam *= 10
            rel_act = (cost - best_cost)/cost
            hist.append(cost)
            cum10 = ((hist[-11]-cost)/hist[-11]) if len(hist) > 10 else None
            e = self.states - truth
            rows.append({'iter': n, 'cost': cost, 'rel_pred': float(rel_pred),
                         'rel_act': float(rel_act),
                         'cum10': float(cum10) if cum10 is not None else None,
                         'stepnorm': float(la.norm(dx*best_scale)) if best_scale > 0 else 0.0,
                         'scale': float(best_scale),
                         'pos_rms': float(np.sqrt(np.mean(la.norm(e[:,:3],axis=1)**2)))})
            n += 1
            stalled = stalled+1 if rel_act < 1e-6 else 0
            if n >= max_iters or lam > 1e10: finished = True
        return rows


def job(seed):
    truth, times, dt, dvr, dve, mst, ts = load_truth(CONFIG, "deltaRIC0.5")
    cp, gs = load_config_parameters(CONFIG)
    p = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
         'q_vel_ric': np.array(cp['process_noise_vel'], float), 'use_range': False,
         'measurement_noise_deg': 2.0*ARCSEC, 'range_noise_m': cp['range_noise_m'],
         'initial_pos_error': cp['initial_pos_error'],
         'initial_vel_error': cp['initial_vel_error'],
         'dv_initial_error': cp['dv_initial_error'],
         't_star_initial_error': cp['t_star_initial_error'],
         'epsilon': cp['epsilon'], 'max_iterations': 160}
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, p, dve, mst, ts, ProbeFGO)
    # run WITHOUT any early termination, to 160 iters, so we see the whole curve
    return seed, fgo.probe(160, truth)


if __name__ == '__main__':
    with Pool(3) as pool:
        res = pool.map(job, [1, 3, 7])
    json.dump({str(s): r for s, r in res}, open(f"{OUT}/term_probe.json","w"),
              indent=2, default=float)
    # baseline stop points from sweep300: s3 stops at 19, s1 at 141, s7 at 64
    STOP = {1: 141, 3: 19, 7: 64}
    for s, rows in res:
        print(f"\n=== seed {s} (baseline stops at iter {STOP[s]}, "
              f"converged pos_rms {rows[min(len(rows)-1,159)]['pos_rms']:.1f} m) ===")
        print(f"{'iter':>5}{'pos_rms':>10}{'rel_act':>11}{'rel_pred':>11}"
              f"{'cum10':>11}{'stepnorm':>11}{'scale':>8}")
        marks = sorted(set([0,1,2,5,10,15,18,19,20,25,30,40,50,60,64,80,100,120,140,141,159]))
        for i in marks:
            if i >= len(rows): continue
            r = rows[i]
            c = f"{r['cum10']:.2e}" if r['cum10'] is not None else "   --   "
            m = "  <-- baseline stop" if i == STOP[s] else ""
            print(f"{r['iter']:>5}{r['pos_rms']:>10.1f}{r['rel_act']:>11.2e}"
                  f"{r['rel_pred']:>11.2e}{c:>11}{r['stepnorm']:>11.2e}"
                  f"{r['scale']:>8.4f}{m}")
