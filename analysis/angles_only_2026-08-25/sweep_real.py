#!/usr/bin/env python3
"""Sweep that calls the REAL Orbit_FGO.opt(), so repo changes are exercised.

Iteration count is recovered by counting create_L calls (exactly one per
opt() iteration) rather than by instrumenting opt() itself.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time, json, argparse
from multiprocessing import Pool
import numpy as np, numpy.linalg as la
sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from Orbit_FGO import SatelliteOrbitFGO, eci_to_ric
from truth_cache import load_truth
from tier0_diag import build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"; TAG = "deltaRIC0.5"
OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1/3600


class CountingFGO(SatelliteOrbitFGO):
    _n_L = 0
    def create_L(self):
        self._n_L += 1
        return super().create_L()


def job(a):
    noise, seed, maxit = a
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
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts, CountingFGO)
    fgo._n_L = 0
    t0 = time.perf_counter()
    fgo.opt(max_iters=maxit, verbose=False)      # <-- the real solver
    rt = time.perf_counter() - t0
    yv = fgo.create_y(); n_dyn = 6 * (fgo.N - 1)
    n_meas = fgo.N * fgo.n_stations * fgo.meas_per_station
    e = fgo.states - truth
    dv_ric = eci_to_ric(fgo.man_params[0:3], mst)
    row = {'noise': noise, 'seed': seed, 'maxit': maxit,
           'pos_rms': float(np.sqrt(np.mean(la.norm(e[:, :3], axis=1) ** 2))),
           'cost_dyn': float(yv[:n_dyn] @ yv[:n_dyn]),
           'cost_meas': float(yv[n_dyn:n_dyn+n_meas] @ yv[n_dyn:n_dyn+n_meas]),
           'tstar_err': float(fgo.man_params[3] - ts),
           'dv_err': float(la.norm(dv_ric - dvr)),
           'n_iters': fgo._n_L, 'runtime': rt}
    print(f"{noise:>4.0f}\" s{seed:<3} maxit={maxit:<4} pos_rms={row['pos_rms']:9.2f}m "
          f"dyn={row['cost_dyn']:.2e} t*={row['tstar_err']:+8.2f}s "
          f"it={row['n_iters']:>3} ({rt:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(1, 11)))
    ap.add_argument('--noises', type=float, nargs='*', default=[10.0, 2.0])
    ap.add_argument('--maxits', type=int, nargs='*', default=[50])
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    jobs = [(n, s, m) for m in a.maxits for n in a.noises for s in a.seeds]
    with Pool(a.workers) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(f"{OUT}/real_{a.tag}.json", "w"), indent=2, default=float)
    for m in a.maxits:
        for n in a.noises:
            sel = [r['pos_rms'] for r in rows if r['maxit'] == m and r['noise'] == n]
            print(f"\nmaxit={m} noise={n}\": mean {np.mean(sel):8.2f} m   "
                  f"median {np.median(sel):8.2f} m   max {np.max(sel):9.2f} m")
