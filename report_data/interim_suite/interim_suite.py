#!/usr/bin/env python3
"""Runs the interim FGO/BLS/EKF comparison suite and writes interim_suite.json.

Five scenarios x three manoeuvre configs x {FGO, BLS, EKF} x {-B, -G} x N seeds.
Q is recalibrated per scenario (5x per-step RMS mismatch) before the solves.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import time

import sys

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
os.chdir(ROOT)
sys.path.insert(0, os.path.abspath(ROOT))

import mc_fgo
import mc_bls
import mc_ekf
mc_fgo.ENABLE_PLOTTING = False
mc_bls.ENABLE_PLOTTING = False
mc_ekf.ENABLE_PLOTTING = False

from fgo_pipeline import load_config_parameters
from Orbit_FGO import SatelliteOrbitFGO
from Orbit_BLS import SatelliteOrbitBLS
from calibrate_q_ric import measure_ric_mismatch

BASE = 'configs/config_geo_one_rev_delta{}.yml'
CFGS = ['RIC0', 'I0.2', 'RIC0.5']
MAX_ITERS = 300
CFG_DIR = '/tmp/interim_suite_cfg'

SCEN = {
    'baseline':  dict(title='Baseline — 2-body + J2, 1.15 day arc'),
    'arc25':     dict(title='Longer propagation — 2-body + J2, 2.5 day arc',
                      mjd_end=59350.00, pm_duration=1.5),
    'lunisolar': dict(title='Third-body — Sun + Moon + SRP, 1.15 day arc',
                      sun=True, moon=True, srp=True),
    'sun':       dict(title='Third-body — Sun + SRP, 1.15 day arc',
                      sun=True, moon=False, srp=True),
    'moon':      dict(title='Third-body — Moon + SRP, 1.15 day arc',
                      sun=False, moon=True, srp=True),
}
SCEN_ORDER = ['baseline', 'arc25', 'lunisolar', 'sun', 'moon']

ARC_LENGTHS = [0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
EPS_VALUES = [20, 25, 30, 45, 60, 100, 150, 200, 300]
MJD0 = 59349.00


def register_eps_sweep(values=None):
    """Gaussian pulse width scenarios. Epsilon enters the estimator only, not truth."""
    names = []
    for e in (values or EPS_VALUES):
        k = f'eps{e:g}'
        SCEN[k] = dict(title=f'epsilon = {e:g} s', epsilon=float(e))
        names.append(k)
    return names


def register_arc_sweep(lengths=None):
    """Arc-length scenarios, manoeuvre held at 40% of each arc."""
    names = []
    for L in (lengths or ARC_LENGTHS):
        k = f'arc{L:.2f}'
        SCEN[k] = dict(title=f'Arc length — 2-body + J2, {L:g} day arc',
                       mjd_end=MJD0 + 0.4 * L, pm_duration=0.6 * L)
        names.append(k)
    return names

# Capture the solver instance so rigid_dev and iteration counts survive the call.
_LAST = {}
_orig_opt, _orig_run = SatelliteOrbitFGO.opt, SatelliteOrbitBLS.run


def _opt(self, *a, **k):
    _LAST['obj'] = self
    return _orig_opt(self, *a, **k)


def _run(self, *a, **k):
    _LAST['obj'] = self
    return _orig_run(self, *a, **k)


SatelliteOrbitFGO.opt = _opt
SatelliteOrbitBLS.run = _run


def rigid_dev(fgo):
    """RMS gap between the FGO trajectory and a rigid propagation of its own solution."""
    s = np.zeros_like(fgo.states)
    s[0] = fgo.states[0]
    for i in range(1, fgo.N):
        s[i] = fgo.prop_one_timestep(s[i - 1], (i - 1) * fgo.dt)
    d = np.linalg.norm(fgo.states[:, :3] - s[:, :3], axis=1)
    return float(np.sqrt(np.mean(d ** 2)))


def build_config(scen, cfg, q_pos=None, q_vel=None):
    """Write the scenario/config YAML variant and return its path."""
    s = SCEN[scen]
    with open(BASE.format(cfg)) as f:
        c = yaml.safe_load(f)

    if 'mjd_end' in s:
        c['scenario_parameters']['MJD_end'] = s['mjd_end']
        c['manoeuvre_parameters']['pm_duration'] = s['pm_duration']

    t = c['propagator_truth_settings']
    sun, moon, srp = s.get('sun', False), s.get('moon', False), s.get('srp', False)
    t['third_body_attraction'] = bool(sun or moon)
    t['third_body_sun'] = bool(sun)
    t['third_body_moon'] = bool(moon)
    t['solar_radiation_pressure'] = bool(srp)
    if srp:
        t['srpCoef'] = 1.5
        t['srpArea'] = 20.0

    if q_pos is not None:
        c['fgo_parameters']['process_noise_position'] = [float(v) for v in q_pos]
        c['fgo_parameters']['process_noise_velocity'] = [float(v) for v in q_vel]
    c['fgo_parameters']['max_iterations'] = MAX_ITERS
    if 'epsilon' in s:
        c['manoeuvre_parameters']['epsilon'] = s['epsilon']

    os.makedirs(CFG_DIR, exist_ok=True)
    path = os.path.join(CFG_DIR, f'{scen}_{cfg}.yml')
    with open(path, 'w') as f:
        yaml.safe_dump(c, f)
    return path


def calibrate(scen):
    """Q = 5x per-step RMS dynamics mismatch, per RIC axis.

    Measured over the whole arc the estimator sees, not just the pre-manoeuvre
    segment: the config's own MJD_end stops at the burn, and for the perturbed
    scenarios that window over-estimates Q by ~12%.
    """
    path = build_config(scen, 'RIC0')
    with open(path) as f:
        c = yaml.safe_load(f)
    sp = c['scenario_parameters']
    sp['MJD_end'] = (sp['MJD_start'] + (sp['MJD_end'] - sp['MJD_start'])
                     + c['manoeuvre_parameters']['pm_duration'])
    path = os.path.join(CFG_DIR, f'{scen}_cal.yml')
    with open(path, 'w') as f:
        yaml.safe_dump(c, f)
    pos_err, vel_err, _ = measure_ric_mismatch(path, dt_val=60.0)
    q_pos = 5 * np.sqrt(np.mean(pos_err ** 2, axis=0))
    q_vel = 5 * np.sqrt(np.mean(vel_err ** 2, axis=0))
    return q_pos, q_vel


TRUTH = {}
PARAMS = {}


def run_one(job):
    scen, cfg, est, mode, seed = job
    S, T, dt, dv_ric, dv_eci, man_state, t_star = TRUTH[(scen, cfg)]
    params = PARAMS[(scen, cfg)]
    fn = {'FGO': mc_fgo.run_fgo_seed, 'BLS': mc_bls.run_bls_seed,
          'EKF': mc_ekf.run_ekf_seed}[est]
    _LAST.pop('obj', None)
    t0 = time.perf_counter()
    r = fn(seed, S, T, dt, params['_gs'], params, dv_ric, dv_eci, man_state,
           t_star, f'{est}-{mode}', cfg)
    obj = _LAST.get('obj')
    out = dict(scen=scen, cfg=cfg, est=est, mode=mode, seed=seed,
               runtime=r['runtime_s'], pos_rms=r['pos_rms'], vel_rms=r['vel_rms'],
               n_iters=getattr(obj, 'num_iters', None) if est != 'EKF' else None,
               converged=getattr(obj, 'converged', None) if est != 'EKF' else None,
               dv_err=r['dv_err_norm'], tstar_err=r['t_star_error'],
               rigid_dev=rigid_dev(obj) if est == 'FGO' else None,
               wall=time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--scenarios', nargs='*', default=None)
    ap.add_argument('--arc-sweep', action='store_true')
    ap.add_argument('--eps-sweep', action='store_true')
    ap.add_argument('--eps', nargs='*', type=float, default=None,
                    help='epsilon values to sweep (default: %s)' % EPS_VALUES)
    ap.add_argument('--baseline-q', action='store_true',
                    help="use the 2-body+J2 Q for every scenario, instead of "
                         "recalibrating to the scenario's own (deliberately "
                         "unmodelled) forces")
    ap.add_argument('--arcs', nargs='*', type=float, default=None)
    ap.add_argument('--merge-into', default=None,
                    help='merge results into this existing JSON instead of replacing it')
    ap.add_argument('--modes', nargs='*', default=['B', 'G'])
    ap.add_argument('--configs', nargs='*', default=CFGS)
    ap.add_argument('--out', default='report_data/interim_suite/interim_suite.json')
    a = ap.parse_args()
    if a.eps_sweep:
        a.scenarios = register_eps_sweep(a.eps)
    elif a.arc_sweep:
        a.scenarios = register_arc_sweep(a.arcs)
    elif a.scenarios is None:
        a.scenarios = SCEN_ORDER

    qbase = None
    meta = {}
    if a.baseline_q:
        qb_pos, qb_vel = calibrate('baseline')
        print(f'[Q] baseline Q applied to every scenario: q_vel = '
              f'[{qb_vel[0]:.4e}, {qb_vel[1]:.4e}, {qb_vel[2]:.4e}]', flush=True)
    for scen in a.scenarios:
        q_pos, q_vel = calibrate(scen)
        if a.baseline_q:
            q_pos, q_vel = qb_pos, qb_vel
        if qbase is None:
            qbase = q_vel.copy()
        mult = float(np.mean(q_vel / qbase))
        meta[scen] = dict(q_pos=q_pos.tolist(), q_vel=q_vel.tolist(), q_mult=mult)
        print(f'[Q] {scen:10s} mult = {mult:6.1f}x   q_vel = '
              f'[{q_vel[0]:.4e}, {q_vel[1]:.4e}, {q_vel[2]:.4e}]', flush=True)

        for cfg in a.configs:
            path = build_config(scen, cfg, q_pos, q_vel)
            TRUTH[(scen, cfg)] = mc_fgo.propagate_truth(path, f'is_{scen}_{cfg}')
            cp, gs = load_config_parameters(path)
            PARAMS[(scen, cfg)] = {
                'q_pos_ric': np.array(cp['process_noise_pos'], dtype=float),
                'q_vel_ric': np.array(cp['process_noise_vel'], dtype=float),
                'use_range': cp['use_range'],
                'measurement_noise_deg': cp['measurement_noise_deg'],
                'range_noise_m': cp['range_noise_m'],
                'initial_pos_error': cp['initial_pos_error'],
                'initial_vel_error': cp['initial_vel_error'],
                'dv_initial_error': cp['dv_initial_error'],
                't_star_initial_error': cp['t_star_initial_error'],
                'epsilon': cp['epsilon'],
                'max_iterations': MAX_ITERS,
                'gmst0': cp['gmst0'],
                '_gs': gs,
            }
            N = len(TRUTH[(scen, cfg)][0])
            print(f'      truth {scen}/{cfg}: N = {N}', flush=True)
        meta[scen]['N'] = N

    jobs = [(scen, cfg, est, mode, seed)
            for scen in a.scenarios for cfg in a.configs
            for est in ('FGO', 'BLS', 'EKF') for mode in a.modes
            for seed in range(1, a.seeds + 1)]
    print(f'\n{len(jobs)} runs on {a.workers} workers\n', flush=True)

    mp.set_start_method('fork', force=True)
    rows = []
    t0 = time.perf_counter()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows.append(r)
            el = time.perf_counter() - t0
            eta = el / i * (len(jobs) - i)
            print(f'[{i:4d}/{len(jobs)}] {r["scen"]:10s} {r["cfg"]:7s} '
                  f'{r["est"]}-{r["mode"]} s{r["seed"]:<3d} '
                  f'pos={r["pos_rms"]:10.2f}  {r["wall"]:5.1f}s  '
                  f'elapsed {el/60:5.1f}m  ETA {eta/60:5.1f}m', flush=True)

    if a.merge_into and os.path.exists(a.merge_into):
        prev = json.load(open(a.merge_into))
        keys = {(r['scen'], r['cfg'], r['est'], r['mode'], r['seed']) for r in rows}
        rows += [r for r in prev['runs']
                 if (r['scen'], r['cfg'], r['est'], r['mode'], r['seed']) not in keys]
        meta = {**prev['meta'], **meta}
        a.out = a.merge_into
    with open(a.out, 'w') as f:
        json.dump({'meta': meta, 'runs': rows}, f)
    print(f'\nwrote {a.out}  ({len(rows)} runs, {(time.perf_counter()-t0)/60:.1f} min)')


if __name__ == '__main__':
    main()
