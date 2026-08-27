#!/usr/bin/env python3
"""Configurable MC harness: epoch x epsilon x damping x mode x noise x seeds.

Serves both the 2x2 configuration check and the large robustness MC that
follows, so the two are measured by identical code.

  --damping / --no-damping   shipped solver (vestigial lambda*I) vs pure
                             Gauss-Newton. ANGLES_ONLY_FGO.md 4.2 shows the
                             shipped damping is 2.8e-16 relative, i.e. below
                             machine epsilon, and that removing it helps at the
                             BASELINE epoch. Untested at the later epochs, where
                             convergence is already 10/10 -- hence the 2x2.
  --eps                      Gaussian pulse width. 5.7: accuracy is flat over
                             the quadrature-valid range but epsilon=100 converges
                             2-3x faster than 30 at the baseline epoch.

Convergence is 'terminated on its own', not max_iters. With zero failures in N
seeds the 95% upper bound on the failure rate is about 3/N (rule of three), so
N=10 only supports "<30%", N=100 supports "<3%".
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, json, argparse, itertools
from multiprocessing import Pool
import numpy as np, numpy.linalg as la

sys.path.insert(0, '/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from Orbit_FGO import eci_to_ric
from truth_cache import load_truth
from tier0_diag import build_seed
from od_study import CurveFGO, rms

OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1 / 3600
C = "configs/config_geo_one_rev_deltaRIC0.5"
EPOCHS = {
    'base':    (f"{C}.yml",         'deltaRIC0.5'),          # 13% pre
    'pre030':  (f"{C}_pre030.yml",  'deltaRIC0.5_pre030'),   # 26%
    'pre040':  (f"{C}_pre040.yml",  'deltaRIC0.5_pre040'),   # 40%
    'pre045':  (f"{C}_pre045.yml",  'deltaRIC0.5_pre045'),   # 39%
    'pre0575': (f"{C}_pre0575.yml", 'deltaRIC0.5_pre0575'),  # 50%
    'RIC0':    ("configs/config_geo_one_rev_deltaRIC0.yml", 'deltaRIC0'),
    'RIC0_pre040': ("configs/config_geo_one_rev_deltaRIC0_pre040.yml",
                    'deltaRIC0_pre040'),
    # delta-v magnitude / direction variants, all at the 40% epoch so |dv|
    # effects are not confounded with the baseline geometry.
    'RIC1_pre040': ("configs/config_geo_one_rev_deltaRIC1_pre040.yml",
                    'deltaRIC1_pre040'),    # |dv| = 1.7321, same direction as RIC0.5
    'I02_pre040':  ("configs/config_geo_one_rev_deltaI0.2_pre040.yml",
                    'deltaI0.2_pre040'),    # |dv| = 0.2, in-track
    'C02_pre040':  ("configs/config_geo_one_rev_deltaC0.2_pre040.yml",
                    'deltaC0.2_pre040'),    # |dv| = 0.2, cross-track
}


def job(a):
    epoch, mode, noise, seed, eps, damping, maxit, use_range = a
    cfg, tag = EPOCHS[epoch]
    truth, times, dt, dvr, dve, mst, ts = load_truth(cfg, tag)
    cp, gs = load_config_parameters(cfg)
    params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
              'q_vel_ric': np.array(cp['process_noise_vel'], float),
              'use_range': use_range, 'measurement_noise_deg': noise * ARCSEC,
              'range_noise_m': cp['range_noise_m'],
              'initial_pos_error': cp['initial_pos_error'],
              'initial_vel_error': cp['initial_vel_error'],
              'dv_initial_error': cp['dv_initial_error'],
              't_star_initial_error': cp['t_star_initial_error'],
              'epsilon': eps if eps is not None else cp['epsilon'],
              'max_iterations': maxit}
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts,
                           CurveFGO, mode=mode)
    tidx = int(round(ts / dt))
    t0 = time.perf_counter()
    curve, term = fgo.opt_curve(maxit, truth, tidx, damping=damping)
    rt = time.perf_counter() - t0

    e = fgo.states - truth
    pe = la.norm(e[:, :3], axis=1)
    row = {'epoch': epoch, 'mode': mode, 'noise': noise, 'seed': seed,
           'epsilon': params['epsilon'], 'damping': damping,
           'use_range': use_range,
           'pre_steps': tidx, 'N': int(fgo.N),
           'pos_rms': rms(pe), 'pos_rms_pre': rms(pe[:tidx + 1]),
           'pos_rms_post': rms(pe[tidx + 1:]),
           'pos_rms_at50': curve[min(49, len(curve) - 1)]['pos_rms'],
           'n_iters': len(curve), 'termination': term, 'runtime': rt}
    if mode == 'FGO-G' and fgo.n_manoeuvres > 0:
        dv_est = eci_to_ric(fgo.man_params[0:3], mst)
        row['tstar_err'] = float(fgo.man_params[3] - ts)
        row['dv_err'] = float(la.norm(dv_est - dvr))
        # store the VECTOR, not just its norm: if the manoeuvre parameters are
        # absorbing a deterministic dynamics mismatch, the estimate is
        # systematic across seeds (|mean(v)| ~ mean(|v|)); if they are fitting
        # measurement noise it is random (|mean(v)| << mean(|v|)).
        row['dv_est_R'], row['dv_est_I'], row['dv_est_C'] = map(float, dv_est)
        row['tstar_est'] = float(fgo.man_params[3])
    return row


def summarise(rows, keys, title):
    print("\n" + "=" * 112); print(title); print("=" * 112)
    hdr = "".join(f"{k:>10}" for k in keys)
    print(hdr + f"{'conv':>9}{'p_fail<=':>10}{'pos_rms':>10}{'pre':>8}{'post':>8}"
                f"{'|t*err|':>9}{'|dv|err':>9}{'it mean':>7}{'it med':>7}"
                f"{'it p95':>7}{'it max':>7}{'sec/run':>8}{'>50it':>7}")
    seen = sorted({tuple(r[k] for k in keys) for r in rows},
                  key=lambda t: [str(x) for x in t])
    for combo in seen:
        s = [r for r in rows if tuple(r[k] for k in keys) == combo]
        cv = [r for r in s if r['termination'] == 'converged']
        u = cv if cv else s
        nf = len(s) - len(cv)
        # rule of three: 95% upper bound when zero failures
        pb = f"{3.0/len(s)*100:.1f}%" if nf == 0 else f"({nf} fail)"
        te = ([abs(r['tstar_err']) for r in u if 'tstar_err' in r] or [float('nan')])
        dv = ([r['dv_err'] for r in u if 'dv_err' in r] or [float('nan')])
        print("".join(f"{str(c):>10}" for c in combo)
              + f"{len(cv):>5}/{len(s):<3}{pb:>10}"
              + f"{np.mean([r['pos_rms'] for r in u]):>10.1f}"
              + f"{np.mean([r['pos_rms_pre'] for r in u]):>8.1f}"
              + f"{np.mean([r['pos_rms_post'] for r in u]):>8.1f}"
              + f"{np.nanmean(te):>9.1f}{np.nanmean(dv):>9.4f}"
              + f"{np.mean([r['n_iters'] for r in s]):>7.0f}"
              + f"{np.median([r['n_iters'] for r in s]):>7.0f}"
              + f"{np.percentile([r['n_iters'] for r in s], 95):>7.0f}"
              + f"{np.max([r['n_iters'] for r in s]):>7.0f}"
              + f"{np.mean([r['runtime'] for r in s]):>8.1f}"
              + f"{int(sum(r['n_iters'] > 50 for r in s)):>7}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--epochs', nargs='*', default=['pre040'])
    ap.add_argument('--modes', nargs='*', default=['FGO-G'])
    ap.add_argument('--noises', type=float, nargs='*', default=[2.0])
    ap.add_argument('--seeds', type=int, default=10, help='seeds 1..N')
    ap.add_argument('--eps', type=float, nargs='*', default=[None])
    ap.add_argument('--damping', nargs='*', default=['on'],
                    choices=['on', 'off'], help="'on' = shipped, 'off' = pure GN")
    ap.add_argument('--maxit', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--range', action='store_true',
                    help='enable range measurements (uses range_noise_m from '
                         'the config). NOTE: this changes the number of RNG '
                         'draws in simulate_measurements, so range and '
                         'angles-only runs are NOT paired with each other; '
                         'FGO-B vs FGO-G WITHIN an arm still is.')
    ap.add_argument('--group', nargs='*', default=None,
                    help='columns to group the summary by')
    a = ap.parse_args()

    eps = [None if e is None or e < 0 else e for e in a.eps]
    jobs = [(ep, md, n, s, e, d == 'on', a.maxit, a.range)
            for ep, md, n, e, d in itertools.product(
                a.epochs, a.modes, a.noises, eps, a.damping)
            for s in range(1, a.seeds + 1)]
    print(f"{len(jobs)} runs on {a.workers} workers\n", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as p:
        rows = p.map(job, jobs)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min")
    json.dump(rows, open(f"{OUT}/mc_{a.tag}.json", "w"), indent=2, default=float)

    keys = a.group or [k for k, v in (('epoch', a.epochs), ('mode', a.modes),
                                      ('noise', a.noises), ('epsilon', eps),
                                      ('damping', a.damping)) if len(v) > 1]
    summarise(rows, keys or ['epoch'],
              f"MC '{a.tag}'   ('p_fail<=' is the 95% rule-of-three bound "
              f"when there are zero failures)")
