#!/usr/bin/env python3
"""Two diagnostics on manoeuvre estimation (2026-08-26). FGO-G throughout.

STUDY 'eps' -- does the Gaussian pulse width set a floor on t* precision?
    |t*_err| improves as roughly sqrt(noise), not linearly (105.9 / 52.7 / 25.0 s
    at 10"/2"/1"), so it is partly noise-limited and partly floored by something
    else. The residual at 1 arcsec (25.0 s) is suspiciously close to epsilon=30 s.
    Sweep epsilon and see whether |t*_err| tracks it.
    Values ABOVE 30 are included deliberately: a two-sided trend is far more
    convincing than one-sided, and the upper half is quadrature-safe. epsilon=12
    is the validity floor (ANGLES_ONLY_FGO.md 6 needs epsilon >~ dt/5 = 12 s)
    and is flagged as marginal.

STUDY 'epoch' -- how much of the error is scenario geometry?
    The hinge dumps its cost on the pre-manoeuvre arc, which is only 13% of the
    arc in the baseline (5.4). Move the burn later, holding total arc length at
    1.15 days so N and the dt grid are unchanged.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, json, argparse
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
BASE = "configs/config_geo_one_rev_deltaRIC0.5.yml"
EPOCHS = [('base', BASE, 'deltaRIC0.5'),
          ('pre030', "configs/config_geo_one_rev_deltaRIC0.5_pre030.yml",
           'deltaRIC0.5_pre030'),
          ('pre045', "configs/config_geo_one_rev_deltaRIC0.5_pre045.yml",
           'deltaRIC0.5_pre045'),
          ('pre0575', "configs/config_geo_one_rev_deltaRIC0.5_pre0575.yml",
           'deltaRIC0.5_pre0575')]


def job(a):
    label, cfg, tag, noise, seed, eps, maxit = a
    truth, times, dt, dvr, dve, mst, ts = load_truth(cfg, tag)
    cp, gs = load_config_parameters(cfg)
    params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
              'q_vel_ric': np.array(cp['process_noise_vel'], float),
              'use_range': False, 'measurement_noise_deg': noise * ARCSEC,
              'range_noise_m': cp['range_noise_m'],
              'initial_pos_error': cp['initial_pos_error'],
              'initial_vel_error': cp['initial_vel_error'],
              'dv_initial_error': cp['dv_initial_error'],
              't_star_initial_error': cp['t_star_initial_error'],
              'epsilon': eps if eps is not None else cp['epsilon'],
              'max_iterations': maxit}
    fgo, _, _ = build_seed(seed, truth, times, dt, gs, params, dve, mst, ts,
                           CurveFGO, mode='FGO-G')
    tidx = int(round(ts / dt))
    t0 = time.perf_counter()
    curve, term = fgo.opt_curve(maxit, truth, tidx)
    rt = time.perf_counter() - t0

    e = fgo.states - truth
    pe = la.norm(e[:, :3], axis=1)
    dv_ric = eci_to_ric(fgo.man_params[0:3], mst)
    row = {'label': label, 'noise': noise, 'seed': seed,
           'epsilon': params['epsilon'], 't_star_true': ts,
           'pre_steps': tidx, 'N': int(fgo.N),
           'pos_rms': rms(pe), 'pos_rms_pre': rms(pe[:tidx + 1]),
           'pos_rms_post': rms(pe[tidx + 1:]),
           'tstar_err': float(fgo.man_params[3] - ts),
           'dv_err': float(la.norm(dv_ric - dvr)),
           'n_iters': len(curve), 'termination': term, 'runtime': rt}
    print(f"{label:<8}{noise:>3.0f}\" s{seed:<3} eps={params['epsilon']:<6g} "
          f"rms={row['pos_rms']:8.2f} pre={row['pos_rms_pre']:8.2f} "
          f"post={row['pos_rms_post']:7.2f} t*err={row['tstar_err']:+8.1f} "
          f"it={row['n_iters']:>3} {term:<10} ({rt:.0f}s)", flush=True)
    return row


def report(rows, key, levels, title, extra=None):
    print("\n" + "=" * 104); print(title); print("=" * 104)
    for n in sorted({r['noise'] for r in rows}, reverse=True):
        print(f"\n--- {n:.0f} arcsec ---")
        hdr = (f"{key:>10}{'conv':>7}{'pos_rms':>10}{'pre':>9}{'post':>9}"
               f"{'|t*err|':>10}{'t* med':>9}{'|dv|err':>10}{'iters':>7}")
        print(hdr + (f"{extra:>10}" if extra else ""))
        for lv in levels:
            s = [r for r in rows if r['noise'] == n and r[key] == lv]
            if not s:
                continue
            cv = [r for r in s if r['termination'] == 'converged']
            u = cv if cv else s
            te = np.abs([r['tstar_err'] for r in u])
            line = (f"{lv:>10}{len(cv):>4}/{len(s):<2}"
                    f"{np.mean([r['pos_rms'] for r in u]):>10.1f}"
                    f"{np.mean([r['pos_rms_pre'] for r in u]):>9.1f}"
                    f"{np.mean([r['pos_rms_post'] for r in u]):>9.1f}"
                    f"{te.mean():>10.1f}{np.median(te):>9.1f}"
                    f"{np.mean([r['dv_err'] for r in u]):>10.4f}"
                    f"{np.mean([r['n_iters'] for r in s]):>7.0f}")
            if extra == 'pre%':
                line += f"{100*u[0]['pre_steps']/u[0]['N']:>10.1f}"
            print(line)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=['eps', 'epoch'])
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(1, 11)))
    ap.add_argument('--noises', type=float, nargs='*', default=[2.0, 1.0])
    ap.add_argument('--maxit', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()

    if a.study == 'eps':
        EPS = [12.0, 20.0, 30.0, 60.0, 100.0]
        jobs = [('base', BASE, 'deltaRIC0.5', n, s, e, a.maxit)
                for n in a.noises for s in a.seeds for e in EPS]
    else:
        jobs = [(lb, cf, tg, n, s, None, a.maxit)
                for lb, cf, tg in EPOCHS for n in a.noises for s in a.seeds]

    print(f"{len(jobs)} runs\n")
    with Pool(a.workers) as p:
        rows = p.map(job, jobs)
    json.dump(rows, open(f"{OUT}/{a.study}_sweep.json", "w"), indent=2, default=float)

    if a.study == 'eps':
        report(rows, 'epsilon', EPS,
               "STUDY 1: does the Gaussian pulse width floor t* precision?\n"
               "    If |t*err| tracks epsilon, Option A is justified and quantified.\n"
               "    If flat, Option A will not move t* precision.\n"
               "    NOTE epsilon=12 is the quadrature validity floor (dt/5) -- marginal.")
    else:
        report(rows, 'label', [e[0] for e in EPOCHS],
               "STUDY 2: manoeuvre epoch moved later, total arc held at 1.15 days.\n"
               "    Tests how much of the error is scenario geometry (the 5.4 hinge).",
               extra='pre%')
