#!/usr/bin/env python3
"""Renders interim_suite.json into interim_fgo_vs_bls.md."""

import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
JSON = os.path.join(ROOT, 'report_data/interim_suite/interim_suite.json')
ARC_JSON = os.path.join(ROOT, 'report_data/interim_suite/arc_sweep.json')
BO_JSON = os.path.join(ROOT, 'report_data/interim_suite/blackout_sweep.json')
BO_LS_JSON = os.path.join(ROOT, 'report_data/interim_suite/blackout_lunisolar.json')
OUT = os.path.join(ROOT, 'interim_fgo_vs_bls.md')

SCEN_ORDER = ['baseline', 'arc25', 'lunisolar', 'sun', 'moon']
TITLES = {
    'baseline':  'Baseline — 2-body + J2, 1.15 day arc',
    'arc25':     'Longer propagation — 2-body + J2, 2.5 day arc',
    'lunisolar': 'Third-body — Sun + Moon + SRP, 1.15 day arc',
    'sun':       'Third-body — Sun + SRP, 1.15 day arc',
    'moon':      'Third-body — Moon + SRP, 1.15 day arc',
}
CFGS = ['RIC0', 'I0.2', 'RIC0.5']
DV_TRUE = {'RIC0': 0.0, 'I0.2': 0.2, 'RIC0.5': np.sqrt(3 * 0.5 ** 2)}

HEADER = """# Interim results — FGO vs BLS vs EKF

{nseeds} seeds per cell. Angles-only, 2 arcsec, 40% manoeuvre epoch, dt = 60 s, epsilon = 100, pure Gauss-Newton, max_iterations = 300.
SRP where present: A = 20 m^2, Cr = 1.5, satMass = 1000 kg. Q recalibrated per scenario (5x per-step RMS mismatch).

Measurement geometry uses GMST at MJD_start for the ECI-to-ECEF rotation. Earlier revisions of this document used `theta = omega_earth * t`, which placed the stations 233 degrees away in longitude and put the satellite below the horizon for the whole arc. Every number here has been re-measured since that fix.

`-B` = manoeuvre not estimated. `-G` = manoeuvre estimated (dv, t*). On `RIC0` the true delta-v is zero, so the dv column is spurious delta-v, not an error against a real burn.

Values are mean ± standard deviation over seeds. `conv` = seeds terminating on the solver's own criteria (EKF is a filter: not applicable), `iters` contains `mean/max`.

`rigid_dev` = how far the FGO's estimated trajectory strays from a plain propagation of its own starting state. Zero means it produced exactly what BLS could have, large means it used process noise to bend off that path.
"""

HDR_B = ('| config | est | pos RMS (m) | vel RMS (m/s) | iters | conv | '
         'runtime (s) | rigid_dev (m) |\n|---|---|---|---|---|---|---|---|')
HDR_G = ('| config | est | pos RMS (m) | vel RMS (m/s) | dv err (m/s) | dv % | '
         '\\|t* err\\| (s) | t* signed (s) | iters | conv | runtime (s) | '
         'rigid_dev (m) |\n|---|---|---|---|---|---|---|---|---|---|---|---|')


def fmt(v, p):
    return f'{v:.{p}f}'


def row(rs, cfg, est, mode):
    pos = np.array([r['pos_rms'] for r in rs])
    vel = np.array([r['vel_rms'] for r in rs])
    rt = np.mean([r['runtime'] for r in rs])
    if est == 'EKF':
        it, conv = '—', '—'
    else:
        n = [r['n_iters'] for r in rs]
        it = f'{np.mean(n):.1f} / {max(n)}'
        conv = f'{sum(1 for r in rs if r["converged"])}/{len(rs)}'
    rd = [r['rigid_dev'] for r in rs if r['rigid_dev'] is not None]
    rds = fmt(np.mean(rd), 2) if rd else '—'

    cells = [cfg, est, f'{fmt(pos.mean(), 2)} ± {fmt(pos.std(), 2)}',
             f'{fmt(vel.mean(), 5)} ± {fmt(vel.std(), 5)}']
    if mode == 'G':
        dv = np.array([r['dv_err'] for r in rs])
        ts = np.array([r['tstar_err'] for r in rs])
        pct = '—' if DV_TRUE[cfg] == 0 else f'{100 * dv.mean() / DV_TRUE[cfg]:.1f}%'
        cells += [f'{fmt(dv.mean(), 5)} ± {fmt(dv.std(), 5)}', pct,
                  f'{fmt(np.abs(ts).mean(), 1)} ± {fmt(np.abs(ts).std(), 1)}',
                  f'{ts.mean():+.1f}']
    cells += [it, conv, fmt(rt, 1), rds]
    return '| ' + ' | '.join(cells) + ' |'


ARC_CFG = 'RIC0.5'
ARC_EST = ('FGO', 'BLS')
ARC_HDR = ('| arc (d) | N | est | pos RMS (m) | vel RMS (m/s) | dv err (m/s) | dv % | '
           '\\|t* err\\| (s) | t* signed (s) | iters | conv | runtime (s) | '
           'rigid_dev (m) | BLS/FGO pos |'
           '\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')


def arc_section():
    """Arc-length sweep table: one manoeuvre config, FGO vs BLS, -G only."""
    if not os.path.exists(ARC_JSON):
        return []
    d = json.load(open(ARC_JSON))
    runs = [r for r in d['runs']
            if r['cfg'] == ARC_CFG and r['est'] in ARC_EST]
    if not runs:
        return []
    meta = d['meta']
    nseeds = len({r['seed'] for r in runs})
    arcs = sorted({r['scen'] for r in runs}, key=lambda x: float(x[3:]))

    out = ['## Arc-length sweep — 2-body + J2, manoeuvre estimated\n',
           f'{ARC_CFG}, `-G` only, {nseeds} seeds per cell, manoeuvre held at 40% of '
           'each arc. Q recalibrated per arc length (flat to 4 significant figures '
           'across the sweep, so the trend is arc length and not a Q artefact). '
           '`BLS/FGO pos` is the ratio of the two position RMS values at that arc.\n',
           ARC_HDR]
    for a in arcs:
        base = np.mean([r['pos_rms'] for r in runs
                        if r['scen'] == a and r['est'] == 'FGO'])
        for est in ARC_EST:
            rs = [r for r in runs if r['scen'] == a and r['est'] == est]
            if not rs:
                continue
            pos = np.array([r['pos_rms'] for r in rs])
            vel = np.array([r['vel_rms'] for r in rs])
            dv = np.array([r['dv_err'] for r in rs])
            ts = np.array([r['tstar_err'] for r in rs])
            n = [r['n_iters'] for r in rs]
            rd = [r['rigid_dev'] for r in rs if r['rigid_dev'] is not None]
            out.append('| ' + ' | '.join([
                f'{float(a[3:]):.2f}', str(meta[a]['N']), est,
                f'{pos.mean():.2f} ± {pos.std():.2f}',
                f'{vel.mean():.5f} ± {vel.std():.5f}',
                f'{dv.mean():.5f} ± {dv.std():.5f}',
                f'{100 * dv.mean() / DV_TRUE[ARC_CFG]:.1f}%',
                f'{np.abs(ts).mean():.1f} ± {np.abs(ts).std():.1f}',
                f'{ts.mean():+.1f}',
                f'{np.mean(n):.1f} / {max(n)}',
                f'{sum(1 for r in rs if r["converged"])}/{len(rs)}',
                f'{np.mean([r["runtime"] for r in rs]):.1f}',
                f'{np.mean(rd):.2f}' if rd else '—',
                f'{pos.mean() / base:.2f}',
            ]) + ' |')
    out.append('')
    return out


BO_CFG = 'RIC0.5'
BO_WINDOWS = [('bo_pre', '10-40% (ends at burn)'),
              ('bo_span', '25-55% (straddles burn)'),
              ('bo_post', '60-90% (after burn)')]
BO_HDR = ('| blackout | est | pos RMS (m) | vel RMS (m/s) | dv err (m/s) | dv % | '
          '\\|t* err\\| (s) | t* signed (s) | iters | conv | runtime (s) | '
          'rigid_dev (m) | BLS/FGO pos |'
          '\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')


def _bo_rows(src, label, ctrl_scen=None):
    """One window's FGO/BLS/EKF rows."""
    rows = []
    fgo = [r for r in src if r['est'] == 'FGO']
    base = np.mean([r['pos_rms'] for r in fgo]) if fgo else None
    for est in ('FGO', 'BLS', 'EKF'):
        rs = [r for r in src if r['est'] == est]
        if not rs:
            continue
        pos = np.array([r['pos_rms'] for r in rs])
        vel = np.array([r['vel_rms'] for r in rs])
        dv = np.array([r['dv_err'] for r in rs])
        ts = np.array([r['tstar_err'] for r in rs])
        n = [r['n_iters'] for r in rs if r['n_iters'] is not None]
        rd = [r['rigid_dev'] for r in rs if r['rigid_dev'] is not None]
        rows.append('| ' + ' | '.join([
            label, est,
            f'{pos.mean():.2f} ± {pos.std():.2f}',
            f'{vel.mean():.5f} ± {vel.std():.5f}',
            f'{dv.mean():.5f} ± {dv.std():.5f}',
            f'{100 * dv.mean() / DV_TRUE[BO_CFG]:.1f}%',
            f'{np.abs(ts).mean():.1f} ± {np.abs(ts).std():.1f}',
            f'{ts.mean():+.1f}',
            f'{np.mean(n):.1f} / {max(n)}' if n else '—',
            f'{sum(1 for r in rs if r["converged"])}/{len(rs)}' if n else '—',
            f'{np.mean([r["runtime"] for r in rs]):.1f}',
            f'{np.mean(rd):.2f}' if rd else '—',
            f'{pos.mean() / base:.2f}',
        ]) + ' |')
    return rows


def blackout_section():
    """Measurement-dropout tables: clean dynamics and lunisolar, with controls."""
    if not (os.path.exists(BO_JSON) and os.path.exists(BO_LS_JSON)):
        return []
    main = json.load(open(JSON))['runs']
    blocks = [('2-body + J2', BO_JSON, 'baseline', ''),
              ('Sun + Moon + SRP', BO_LS_JSON, 'lunisolar', '_lunisolar')]
    nseeds = len({r['seed'] for r in json.load(open(BO_JSON))['runs']})

    out = ['## Measurement dropout\n',
           f'{BO_CFG}, `-G` only, {nseeds} seeds per cell, 1.15 day arc. The blackout '
           'removes every station for a contiguous window given as a fraction of the '
           'arc; the manoeuvre is at 40%. `no blackout` is the control row, taken from '
           'the corresponding full-data scenario above. `BLS/FGO pos` is the ratio of '
           'the two position RMS values in that window.\n',
           'Read the dv and t* columns together: a plausible dv magnitude fitted at a '
           'badly wrong t* is not a good manoeuvre estimate.\n']

    for title, path, ctrl, suffix in blocks:
        runs = json.load(open(path))['runs']
        out.append(f'### {title}\n')
        out.append(BO_HDR)
        ctrl_rows = [r for r in main if r['scen'] == ctrl and r['cfg'] == BO_CFG
                     and r['mode'] == 'G']
        out += _bo_rows(ctrl_rows, 'none')
        for key, label in BO_WINDOWS:
            src = [r for r in runs if r['scen'] == key + suffix]
            if src:
                out += _bo_rows(src, label)
        out.append('')
    return out


def main():
    d = json.load(open(JSON))
    runs, meta = d['runs'], d['meta']
    nseeds = len({r['seed'] for r in runs})

    out = [HEADER.format(nseeds=nseeds)]
    for scen in SCEN_ORDER:
        sr = [r for r in runs if r['scen'] == scen]
        if not sr:
            continue
        m = meta[scen]
        out.append(f'## {TITLES[scen]}\n')
        out.append(f"N = {m['N']} steps · Q = {m['q_mult']:.1f}x baseline\n")
        for mode, hdr, label in (('B', HDR_B, '-B  (manoeuvre not estimated)'),
                                 ('G', HDR_G, '-G  (manoeuvre estimated)')):
            out.append(f'### {label}\n')
            out.append(hdr)
            for cfg in CFGS:
                for est in ('FGO', 'BLS', 'EKF'):
                    rs = [r for r in sr if r['cfg'] == cfg and r['est'] == est
                          and r['mode'] == mode]
                    if rs:
                        out.append(row(rs, cfg, est, mode))
            out.append('')
        if scen == 'arc25':
            out += arc_section()
    out += blackout_section()
    open(OUT, 'w').write('\n'.join(out) + '\n')
    print(f'wrote {OUT}  ({len(runs)} runs, {nseeds} seeds)')


if __name__ == '__main__':
    main()
