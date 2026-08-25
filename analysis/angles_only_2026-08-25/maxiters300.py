#!/usr/bin/env python3
"""Control: is the failure a RATE problem or a LOCAL MINIMUM?
Rerun the three failing seeds at 2" with a 6x iteration budget."""
import os, sys, time, json
import numpy as np, numpy.linalg as la
sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from Orbit_FGO import eci_to_ric
import mc_fgo
from tier0_diag import InstrumentedFGO, build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"
OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1/3600
truth, times, dt, dvr, dve, mst, ts = mc_fgo.propagate_truth(CONFIG, "deltaRIC0.5")
cp, gs = load_config_parameters(CONFIG)
params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
          'q_vel_ric': np.array(cp['process_noise_vel'], float),
          'use_range': False, 'measurement_noise_deg': 2.0*ARCSEC,
          'range_noise_m': cp['range_noise_m'],
          'initial_pos_error': cp['initial_pos_error'],
          'initial_vel_error': cp['initial_vel_error'],
          'dv_initial_error': cp['dv_initial_error'],
          't_star_initial_error': cp['t_star_initial_error'],
          'epsilon': cp['epsilon'], 'max_iterations': 300}
rows = []
for seed in (1, 7, 8):
    fgo, ts0, p0 = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts,
                              InstrumentedFGO)
    t0 = time.perf_counter(); log, diag, term = fgo.opt_logged(max_iters=300)
    rt = time.perf_counter()-t0
    err = fgo.states - truth
    pos_rms = float(np.sqrt(np.mean(la.norm(err[:, :3], axis=1)**2)))
    last = log[-1]
    rows.append({'seed': seed, 'pos_rms': pos_rms, 'cost_dyn': last['cost_dyn'],
                 'cost_meas': last['cost_meas'], 'termination': term,
                 'n_iters': len(log), 'runtime': rt,
                 'tstar_err': float(fgo.man_params[3]-ts),
                 'median_scale': float(np.median([r['best_scale'] for r in log])),
                 'log': log})
    print(f"seed {seed}: pos_rms={pos_rms:9.2f} m  dyn={last['cost_dyn']:.3e}  "
          f"t*err={fgo.man_params[3]-ts:+8.2f}s  {term:<12} it={len(log):>3} "
          f"({rt:.0f}s)", flush=True)
    # cost trajectory every 25 iters
    for r in log[::25]:
        print(f"    it{r['iter']:>4} cost={r['cost']:.4e} dyn={r['cost_dyn']:.3e} "
              f"scale={r['best_scale']:.4f} lam={r['lambda_before']:.1e} "
              f"t*={r['tstar']:.1f}", flush=True)
json.dump(rows, open(f"{OUT}/maxiters300.json", "w"), indent=2, default=float)
print("\nBaseline @50 iters was: seed1 765.04m/1.56e3, seed7 1031.44m/3.06e3, seed8 511.74m/6.97e3")
