#!/usr/bin/env python3
"""Is the bottleneck the conditioning of the normal equations?

At a given iterate, solve the SAME system three ways and compare:
  (1) spsolve(M + lam*I,        L'y)   <- what the code does
  (2) spsolve(M + lam*diag(M),  L'y)   <- variant B
  (3) lsmr(L, y)                       <- least-squares on L, never forms L'L

If the solve is well conditioned, (1) and (2) must agree closely (both are
<=1e-6 relative perturbations) and all three should give similar steps.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys, time
import numpy as np, numpy.linalg as la, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from truth_cache import load_truth
from tier0_diag import InstrumentedFGO, build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"; ARCSEC = 1/3600
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


def best_scale_for(fgo, delta_x, y, L, current_cost):
    scale, best_scale, best_cost = 1.0, 0.0, current_cost
    for _ in range(20):
        try:
            ny = fgo.create_y(fgo.add_delta(delta_x * scale))
            nc = float(ny.T @ ny)
            if nc < best_cost: best_cost, best_scale = nc, scale
            py = y - L @ (delta_x * scale); pc = float(py.T @ py)
            ratio = (current_cost - nc)/(current_cost - pc) if pc > 0 else 0
            if ratio > 0.25 and nc < current_cost:
                return scale, nc, ratio
        except Exception: pass
        scale *= 0.5
        if scale < 1e-10: break
    return best_scale, best_cost, None


for warmup in (0, 10, 30):
    fgo, _, _ = build_seed(1, truth, times, dt, gs, params, dve, mst, ts, InstrumentedFGO)
    if warmup:
        fgo.opt_logged(max_iters=warmup)
    print("=" * 78)
    print(f"SEED 1 @ 2 arcsec, after {warmup} baseline iterations")
    print("=" * 78)

    L = fgo.create_L(); y = fgo.create_y()
    cost = float(y.T @ y)
    M = (L.T @ L).tocsc()
    g = L.T @ y
    d = M.diagonal()
    print(f"  L shape {L.shape}, nnz {L.nnz}")
    print(f"  cost = {cost:.6e}")
    print(f"  M diagonal: min {d.min():.4e}  max {d.max():.4e}  "
          f"spread {d.max()/max(d.min(),1e-300):.3e}")

    t0 = time.perf_counter()
    out = spla.lsmr(L, y, atol=1e-12, btol=1e-12, maxiter=20000)
    dx3, istop, itn, normr, normar, normA, condA, normx = out[:8]
    print(f"  lsmr: itn={itn} istop={istop} normA={normA:.4e} "
          f"condA(L)={condA:.4e}  -> implied cond(M)=cond(L)^2={condA**2:.4e} "
          f"({time.perf_counter()-t0:.1f}s)")
    print(f"  double precision eps = 2.22e-16; cond(M)*eps = {condA**2*2.22e-16:.3e}"
          f"   (>1 means the normal-equations solve retains no significant digits)")

    lam = 1e-6
    dx1 = spla.spsolve((M + lam*sp.eye(M.shape[0])).tocsc(), g)
    dd = np.maximum(d, 1e-12*d.max())
    dx2 = spla.spsolve((M + lam*sp.diags(dd)).tocsc(), g)

    def nres(A, x):
        return la.norm(A @ x - g)/la.norm(g)
    print(f"\n  normal-equation residual ||(M+lamI)dx - L'y||/||L'y|| = "
          f"{nres(M + lam*sp.eye(M.shape[0]), dx1):.3e}")
    print(f"  ||dx1||={la.norm(dx1):.4e}  ||dx2||={la.norm(dx2):.4e}  "
          f"||dx3(lsmr)||={la.norm(dx3):.4e}")
    print(f"  rel diff dx1 vs dx2 = {la.norm(dx1-dx2)/la.norm(dx1):.3e}"
          f"   <- both are <=1e-6 relative perturbations of the same system")
    print(f"  rel diff dx1 vs dx3 = {la.norm(dx1-dx3)/la.norm(dx1):.3e}")

    print(f"\n  {'solver':<22}{'scale':>9}{'new cost':>14}{'reduction':>12}")
    for name, dx in (("spsolve M+lam*I", dx1), ("spsolve M+lam*diag", dx2),
                     ("lsmr on L", dx3)):
        s, nc, r = best_scale_for(fgo, dx, y, L, cost)
        print(f"  {name:<22}{s:>9.5f}{nc:>14.6e}{100*(cost-nc)/cost:>11.2f}%")
    print()
