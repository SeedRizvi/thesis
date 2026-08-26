#!/usr/bin/env python3
"""Phase 4: is the prior the binding constraint on the pre-manoeuvre arc?

Motivated by ANGLES_ONLY_FGO.md 5.4: the manoeuvre parameters insert a hinge at
t* that partially decouples the trajectory. The post arc holds 87% of the
measurements and determines itself; the pre arc has 13% and must be pinned by
that plus the prior. Two things changed since the original validation -- a prior
was added, and range was removed. Range gave direct observability that could
override a poor initial estimate; now the prior *enforces* it instead.

  PREDICTION: strengthening the prior improves pos_rms_pre substantially and
  pos_rms_post barely, and the effect is larger for the hinge configurations
  (RIC0/FGO-G, RIC0.5/FGO-G) than for RIC0/FGO-B, which has no hinge.

Two independent knobs:

  sigma_scale  multiplies initial_pos_error AND initial_vel_error, so it moves
               the sampled perturbation and the prior together. This is the
               SCENARIO axis -- "the initial state is better known" -- and it is
               the self-consistent case (prior == sampling distribution).
               Draws are paired across levels: numpy scales the same standard
               normals, so sigma_scale=0.5 gives exactly half the error of 1.0
               for the same seed.

  p0_scale     multiplies the PRIOR sigmas only, leaving the sampled error
               alone. This deliberately MIS-SPECIFIES the prior, which is the
               robustness axis. p0_scale=1e6 effectively removes the prior.

  4a scenario  sigma_scale in {1, 0.5, 0.25},  p0_scale = 1
  4b misspec   sigma_scale = 1,                p0_scale in {0.25, 0.5, 1, 2, 4}
  4c ablation  sigma_scale = 1,                p0_scale = 1e6
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
from od_study import CurveFGO, CONFIGS, rms

OUT = os.path.dirname(os.path.abspath(__file__)); ARCSEC = 1 / 3600
CASES = [('RIC0', 'FGO-B'), ('RIC0', 'FGO-G'), ('RIC0.5', 'FGO-G')]


def job(a):
    cfgname, mode, noise, seed, sig, ps, maxit = a
    cfg = CONFIGS[cfgname]
    truth, times, dt, dvr, dve, mst, ts = load_truth(cfg, 'delta' + cfgname)
    cp, gs = load_config_parameters(cfg)
    params = {'q_pos_ric': np.array(cp['process_noise_pos'], float),
              'q_vel_ric': np.array(cp['process_noise_vel'], float),
              'use_range': False, 'measurement_noise_deg': noise * ARCSEC,
              'range_noise_m': cp['range_noise_m'],
              'initial_pos_error': cp['initial_pos_error'] * sig,
              'initial_vel_error': cp['initial_vel_error'] * sig,
              'dv_initial_error': cp['dv_initial_error'],
              't_star_initial_error': cp['t_star_initial_error'],
              'p0_scale': ps,
              'epsilon': cp['epsilon'], 'max_iterations': maxit}
    fgo, _, pos_err0 = build_seed(seed, truth, times, dt, gs, params,
                                  dve, mst, ts, CurveFGO, mode=mode)
    tidx = int(round(ts / dt))
    e0 = fgo.states - truth
    init = rms(la.norm(e0[:, :3], axis=1))
    t0 = time.perf_counter()
    curve, term = fgo.opt_curve(maxit, truth, tidx)
    rt = time.perf_counter() - t0

    e = fgo.states - truth
    pe = la.norm(e[:, :3], axis=1)
    row = {'config': cfgname, 'mode': mode, 'noise': noise, 'seed': seed,
           'sigma_scale': sig, 'p0_scale': ps,
           'pos_rms': rms(pe), 'pos_rms_pre': rms(pe[:tidx + 1]),
           'pos_rms_post': rms(pe[tidx + 1:]),
           'pos_rms_init': init, 'x0_pos_err': float(pos_err0),
           'pos_rms_at50': curve[min(49, len(curve) - 1)]['pos_rms'],
           'n_iters': len(curve), 'termination': term, 'runtime': rt}
    if mode == 'FGO-G' and fgo.n_manoeuvres > 0:
        row['tstar_err'] = float(fgo.man_params[3] - ts)
        row['dv_err'] = float(la.norm(eci_to_ric(fgo.man_params[0:3], mst) - dvr))
    print(f"{cfgname:<7}{mode:<7}{noise:>3.0f}\" s{seed:<3} "
          f"sig={sig:<5} p0={ps:<7} rms={row['pos_rms']:8.2f} "
          f"pre={row['pos_rms_pre']:8.2f} post={row['pos_rms_post']:8.2f} "
          f"it={row['n_iters']:>3} {term:<10} ({rt:.0f}s)", flush=True)
    return row


def summarise(rows, key, levels, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for cfgname, mode in CASES:
        print(f"\n--- {cfgname} / {mode} ---")
        print(f"{key:>10}{'conv':>7}{'mean':>10}{'pre':>10}{'post':>10}"
              f"{'iters':>7}{'d(pre)':>9}{'d(post)':>9}")
        ref = None
        for lv in levels:
            sel = [r for r in rows if r['config'] == cfgname and r['mode'] == mode
                   and r[key] == lv]
            if not sel:
                continue
            cv = [r for r in sel if r['termination'] == 'converged']
            use = cv if cv else sel
            m = np.mean([r['pos_rms'] for r in use])
            pre = np.mean([r['pos_rms_pre'] for r in use])
            post = np.mean([r['pos_rms_post'] for r in use])
            it = np.mean([r['n_iters'] for r in sel])
            if ref is None:
                ref = (pre, post)
                dp = dq = ""
            else:
                dp = f"{100*(pre-ref[0])/ref[0]:>+8.0f}%"
                dq = f"{100*(post-ref[1])/ref[1]:>+8.0f}%"
            print(f"{lv:>10}{len(cv):>4}/{len(sel):<2}{m:>10.1f}{pre:>10.1f}"
                  f"{post:>10.1f}{it:>7.0f}{dp:>9}{dq:>9}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(1, 11)))
    ap.add_argument('--noise', type=float, default=2.0)
    ap.add_argument('--maxit', type=int, default=300)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()

    SIG = [1.0, 0.5, 0.25]
    P0S = [0.25, 0.5, 1.0, 2.0, 4.0, 1e6]
    jobs = []
    for c, m in CASES:
        for s in a.seeds:
            for sig in SIG:                       # 4a scenario (p0 self-consistent)
                jobs.append((c, m, a.noise, s, sig, 1.0, a.maxit))
            for ps in P0S:                        # 4b misspec + 4c ablation
                if ps == 1.0:
                    continue                      # already covered by sig=1.0
                jobs.append((c, m, a.noise, s, 1.0, ps, a.maxit))
    print(f"{len(jobs)} runs\n")
    with Pool(a.workers) as pool:
        rows = pool.map(job, jobs)
    json.dump(rows, open(f"{OUT}/phase4_{a.tag}.json", "w"), indent=2, default=float)

    summarise([r for r in rows if r['p0_scale'] == 1.0], 'sigma_scale', SIG,
              "4a SCENARIO: initial uncertainty reduced (sampled error AND prior "
              "together, self-consistent).\n    Deltas are vs sigma_scale=1.0.")
    summarise([r for r in rows if r['sigma_scale'] == 1.0], 'p0_scale', P0S,
              "4b/4c MIS-SPECIFICATION: prior sigmas scaled, sampled error fixed.\n"
              "    p0_scale=1 is correct; 1e6 is effectively no prior. "
              "Deltas are vs p0_scale=0.25.")
