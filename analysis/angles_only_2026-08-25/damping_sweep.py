#!/usr/bin/env python3
"""Does damping help this problem AT ALL?  (2026-08-26)

lm_screen_v2.py compared damping FORMS at equal lambda0, which is meaningless:
post-Q the median M diagonal is 3.6e9, so lam*I at 1e-6 is relative damping
2.8e-16 (below machine eps -- i.e. pure Gauss-Newton) while lam*diag at 1e-6 is
relative damping 1e-6. Nine orders apart.

Here damping is parametrised by a RELATIVE strength tau, so the two forms are
directly comparable:

    diag      :  M + tau * diag(M)
    identity  :  M + tau * median(diag(M)) * I
    tau = 0   :  pure Gauss-Newton (control)

tau is held FIXED for the run, so the question is the clean one: is there any
damping level that beats no damping? Everything else -- line search,
termination criteria -- is exactly the shipped opt().
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


class TauFGO(SatelliteOrbitFGO):

    def opt_tau(self, max_iters, tau, form, truth):
        num_iters, finished = 0, False
        cost_history, log = [], []

        while not finished:
            L = self.create_L()
            y = self.create_y()
            current_cost = float(y.T @ y)
            M = L.T @ L
            Lty = L.T @ y

            if tau <= 0:
                A = M
            elif form == 'diag':
                d = M.diagonal()
                A = M + tau * sp.diags(np.maximum(d, 1e-12 * d.max()))
            else:
                A = M + (tau * float(np.median(M.diagonal()))) * sp.eye(M.shape[0])

            try:
                delta_x = spla.spsolve(A.tocsc(), Lty)
            except Exception:
                break

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
                py = y - L @ (delta_x * best_scale)
                rel_pred_red = (current_cost - float(py.T @ py)) / current_cost
                self.update_state(delta_x * best_scale)

            num_iters += 1
            cost_history.append(best_cost)
            if len(cost_history) > STALL_WINDOW + 1:
                cost_history.pop(0)

            e = self.states - truth
            log.append({'iter': num_iters - 1, 'cost': current_cost,
                        'scale': float(best_scale),
                        'dxnorm': float(la.norm(delta_x * best_scale)),
                        'pos_rms': float(np.sqrt(np.mean(
                            la.norm(e[:, :3], axis=1) ** 2)))})

            if rel_pred_red is not None and rel_pred_red < CONV_REL_PRED:
                finished = True
            if len(cost_history) == STALL_WINDOW + 1 and cost_history[0] > 0:
                if (cost_history[0] - best_cost) / cost_history[0] < STALL_REL_TOL:
                    finished = True
            if num_iters >= max_iters:
                finished = True

        term = 'max_iters' if num_iters >= max_iters else 'converged'
        return log, term


def job(a):
    noise, seed, tau, form, maxit = a
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
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts, TauFGO)
    t0 = time.perf_counter()
    log, term = fgo.opt_tau(maxit, tau, form, truth)
    rt = time.perf_counter() - t0

    yv = fgo.create_y(); n_dyn = 6 * (fgo.N - 1)
    e = fgo.states - truth
    row = {'noise': noise, 'seed': seed, 'tau': tau, 'form': form, 'maxit': maxit,
           'pos_rms': float(np.sqrt(np.mean(la.norm(e[:, :3], axis=1) ** 2))),
           'cost_dyn': float(yv[:n_dyn] @ yv[:n_dyn]),
           'tstar_err': float(fgo.man_params[3] - ts),
           'n_iters': len(log), 'termination': term, 'runtime': rt,
           'median_scale': float(np.median([r['scale'] for r in log])),
           'median_dxnorm': float(np.median([r['dxnorm'] for r in log]))}
    print(f"{noise:>4.0f}\" s{seed:<3}{form:<9} tau={tau:.0e} "
          f"rms={row['pos_rms']:9.2f}m dyn={row['cost_dyn']:.2e} "
          f"it={row['n_iters']:>3} medscale={row['median_scale']:.4f} "
          f"med|dx|={row['median_dxnorm']:.2e} {term} ({rt:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', type=int, nargs='*', default=[1, 3, 5, 6, 8])
    ap.add_argument('--noises', type=float, nargs='*', default=[2.0])
    ap.add_argument('--taus', type=float, nargs='*',
                    default=[0.0, 1e-12, 1e-9, 1e-6, 1e-3])
    ap.add_argument('--forms', nargs='*', default=['diag', 'identity'])
    ap.add_argument('--maxit', type=int, default=50)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()

    jobs = [(n, s, t, f, a.maxit) for n in a.noises for s in a.seeds
            for t in a.taus for f in a.forms if not (t == 0.0 and f == 'identity')]
    with Pool(a.workers) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(f"{OUT}/tau_{a.tag}.json", "w"), indent=2, default=float)

    print("\n" + "=" * 96)
    print(f"maxit={a.maxit}   pos RMS (m), '*' = diverged (cost_dyn > 1e3)")
    print("baseline (shipped opt(), tau ~ 2.8e-16 i.e. pure GN) for reference:")
    print("  2\": s1 40.2  s3 69.2  s5 226.3*  s6 444.8*  s8 10651.1*")
    print("=" * 96)
    for n in a.noises:
        print(f"\n--- {n:.0f} arcsec ---")
        print(f"{'form':<10}{'tau':>9}" + "".join(f"{'s'+str(s):>12}" for s in a.seeds)
              + f"{'div':>6}{'iters':>7}")
        for f in a.forms:
            for t in a.taus:
                if t == 0.0 and f == 'identity':
                    continue
                sel = [next((x for x in rows if x['noise'] == n and x['seed'] == s
                             and x['tau'] == t and x['form'] == f), None)
                       for s in a.seeds]
                if any(x is None for x in sel):
                    continue
                line = f"{('GN' if t==0 else f):<10}{t:>9.0e}"
                ndiv = sum(x['cost_dyn'] > 1e3 for x in sel)
                for x in sel:
                    line += f"{x['pos_rms']:>11.1f}" + ("*" if x['cost_dyn'] > 1e3 else " ")
                print(line + f"{ndiv:>6}{int(np.mean([x['n_iters'] for x in sel])):>7}")
