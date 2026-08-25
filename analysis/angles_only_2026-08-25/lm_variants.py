#!/usr/bin/env python3
"""
LM variant screen on the three failing seeds (1, 7, 8) at 2 arcsec, angles-only.

  A  current:  M + lambda*I,        line search, lambda always shrinks /2
  B  scaled :  M + lambda*diag(M),  line search, lambda always shrinks /2
  C  reactive: M + lambda*diag(M),  line search, lambda responds to backtracking
  D  textbook: M + lambda*diag(M),  NO line search, Nielsen rho-based update,
               L and y reused on a rejected step

Scratchpad only. Orbit_FGO.py is untouched.
"""
import os, sys, time, json
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, '/home/z5363026/thesis')
os.chdir('/home/z5363026/thesis')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fgo_pipeline import load_config_parameters
from Orbit_FGO import eci_to_ric
import mc_fgo
from truth_cache import load_truth
from tier0_diag import InstrumentedFGO, build_seed

CONFIG = "configs/config_geo_one_rev_deltaRIC0.5.yml"
TAG = "deltaRIC0.5"
OUT = os.path.dirname(os.path.abspath(__file__))
ARCSEC = 1.0 / 3600.0
SEEDS = [1, 7, 8]
LAM_FLOOR = 1e-10
LAM_CEIL = 1e10


class LMVariantFGO(InstrumentedFGO):

    def _damped(self, M, lam, damping):
        if damping == 'identity':
            return M + lam * sp.eye(M.shape[0])
        d = M.diagonal()
        d = np.maximum(d, 1e-12 * d.max())
        return M + lam * sp.diags(d)

    # ---- A / B / C: keep the line search, vary damping + lambda rule ----
    def opt_ls(self, max_iters, damping, lam_rule):
        num_iters, lam, stalled = 0, 1e-6, 0
        log = []
        finished = False
        while not finished:
            L = self.create_L()
            y = self.create_y()
            current_cost = float(y.T @ y)
            M = L.T @ L

            try:
                delta_x = spla.spsolve(self._damped(M, lam, damping), L.T @ y)
            except Exception:
                lam *= 10
                continue

            scale, best_scale, best_cost, acc_ratio = 1.0, 0, current_cost, None
            for _ in range(20):
                try:
                    next_y = self.create_y(self.add_delta(delta_x * scale))
                    next_cost = float(next_y.T @ next_y)
                    if next_cost < best_cost:
                        best_cost, best_scale = next_cost, scale
                    pred_y = y - L @ (delta_x * scale)
                    pred_cost = float(pred_y.T @ pred_y)
                    ratio = ((current_cost - next_cost) / (current_cost - pred_cost)
                             if pred_cost > 0 else 0)
                    if ratio > 0.25 and next_cost < current_cost:
                        best_scale, acc_ratio = scale, float(ratio)
                        break
                except Exception:
                    pass
                scale *= 0.5
                if scale < 1e-10:
                    break

            lam_before = lam
            if best_scale > 0:
                self.update_state(delta_x * best_scale)
                if lam_rule == 'always_shrink':
                    lam = max(lam * 0.5, LAM_FLOOR)
                elif lam_rule == 'backtrack':
                    # heavy backtracking means the step was too long: grow lambda.
                    if best_scale >= 1.0 and acc_ratio is not None and acc_ratio > 0.75:
                        lam = max(lam / 10.0, LAM_FLOOR)
                    elif best_scale < 0.25:
                        lam = min(lam * 10.0, LAM_CEIL)
            else:
                lam *= 10

            log.append({'iter': num_iters, 'cost': current_cost,
                        'lambda': lam_before, 'scale': float(best_scale),
                        'ratio': acc_ratio,
                        'tstar': float(self.man_params[3])})
            num_iters += 1
            stalled = stalled + 1 if (current_cost - best_cost) / current_cost < 1e-6 else 0
            if best_scale > 0 and la.norm(delta_x * best_scale) < 1e-3:
                finished = True
            if num_iters >= max_iters or lam > LAM_CEIL or stalled >= 10:
                finished = True
        term = ('max_iters' if num_iters >= max_iters else
                'lambda_blowup' if lam > LAM_CEIL else
                'stalled' if stalled >= 10 else 'step_norm')
        return log, term

    # ---- D: textbook LM, no line search, Nielsen update, L reused on reject ----
    def opt_nielsen(self, max_iters):
        num_iters, lam, nu, stalled = 0, 1e-6, 2.0, 0
        log = []
        finished = False
        rebuild = True
        L = y = M = None
        current_cost = None
        while not finished:
            if rebuild:
                L = self.create_L()
                y = self.create_y()
                current_cost = float(y.T @ y)
                M = L.T @ L
                rebuild = False

            try:
                delta_x = spla.spsolve(self._damped(M, lam, 'diag'), L.T @ y)
            except Exception:
                lam *= nu
                nu *= 2
                num_iters += 1
                if num_iters >= max_iters or lam > LAM_CEIL:
                    finished = True
                continue

            next_y = self.create_y(self.add_delta(delta_x))
            next_cost = float(next_y.T @ next_y)
            pred_y = y - L @ delta_x
            pred_cost = float(pred_y.T @ pred_y)
            denom = current_cost - pred_cost
            rho = (current_cost - next_cost) / denom if denom > 0 else -1.0

            lam_before, accepted = lam, False
            if rho > 0 and next_cost < current_cost:
                self.update_state(delta_x)
                lam = max(lam * max(1.0 / 3.0, 1 - (2 * rho - 1) ** 3), LAM_FLOOR)
                nu = 2.0
                accepted, rebuild = True, True
                improvement = (current_cost - next_cost) / current_cost
            else:
                lam *= nu
                nu *= 2
                improvement = 0.0

            log.append({'iter': num_iters, 'cost': current_cost,
                        'lambda': lam_before, 'scale': 1.0 if accepted else 0.0,
                        'ratio': float(rho),
                        'tstar': float(self.man_params[3])})
            num_iters += 1
            stalled = stalled + 1 if improvement < 1e-6 else 0
            if accepted and la.norm(delta_x) < 1e-3:
                finished = True
            if num_iters >= max_iters or lam > LAM_CEIL or stalled >= 10:
                finished = True
        term = ('max_iters' if num_iters >= max_iters else
                'lambda_blowup' if lam > LAM_CEIL else
                'stalled' if stalled >= 10 else 'step_norm')
        return log, term


VARIANTS = {
    'A_current':  dict(kind='ls', damping='identity', lam_rule='always_shrink'),
    'B_diagscale': dict(kind='ls', damping='diag',    lam_rule='always_shrink'),
    'C_reactive':  dict(kind='ls', damping='diag',    lam_rule='backtrack'),
    'D_nielsen':   dict(kind='nielsen'),
}


def main():
    truth_states, times, dt, delta_v_ric, delta_v_eci, manoeuvre_state, t_star_true = \
        load_truth(CONFIG, TAG)
    cp, gs = load_config_parameters(CONFIG)
    params = {
        'q_pos_ric': np.array(cp['process_noise_pos'], float),
        'q_vel_ric': np.array(cp['process_noise_vel'], float),
        'use_range': False,
        'measurement_noise_deg': 2.0 * ARCSEC,
        'range_noise_m': cp['range_noise_m'],
        'initial_pos_error': cp['initial_pos_error'],
        'initial_vel_error': cp['initial_vel_error'],
        'dv_initial_error': cp['dv_initial_error'],
        't_star_initial_error': cp['t_star_initial_error'],
        'epsilon': cp['epsilon'],
        'max_iterations': 50,
    }

    rows = []
    for seed in SEEDS:
        for vname, vcfg in VARIANTS.items():
            fgo, ts0, p0 = build_seed(seed, truth_states, times, dt, gs, params,
                                      delta_v_eci, manoeuvre_state, t_star_true,
                                      LMVariantFGO)
            t0 = time.perf_counter()
            if vcfg['kind'] == 'ls':
                log, term = fgo.opt_ls(50, vcfg['damping'], vcfg['lam_rule'])
            else:
                log, term = fgo.opt_nielsen(50)
            rt = time.perf_counter() - t0

            yv = fgo.create_y()
            n_dyn = 6 * (fgo.N - 1)
            n_meas = fgo.N * fgo.n_stations * fgo.meas_per_station
            cost_dyn = float(yv[:n_dyn] @ yv[:n_dyn])
            cost_meas = float(yv[n_dyn:n_dyn + n_meas] @ yv[n_dyn:n_dyn + n_meas])

            err = fgo.states - truth_states
            pos_rms = float(np.sqrt(np.mean(la.norm(err[:, :3], axis=1) ** 2)))
            dv_ric = eci_to_ric(fgo.man_params[0:3], manoeuvre_state)

            row = {'seed': seed, 'variant': vname, 'pos_rms': pos_rms,
                   'cost_dyn': cost_dyn, 'cost_meas': cost_meas,
                   'tstar_err': float(fgo.man_params[3] - t_star_true),
                   'dv_err': float(la.norm(dv_ric - delta_v_ric)),
                   'termination': term, 'n_iters': len(log), 'runtime': rt,
                   'lambda_max': max(r['lambda'] for r in log),
                   'lambda_final': log[-1]['lambda'],
                   'median_scale': float(np.median([r['scale'] for r in log])),
                   'n_rejected': int(sum(1 for r in log if r['scale'] == 0))}
            rows.append(row)
            print(f"seed {seed} {vname:<12} pos_rms={pos_rms:9.2f} m  "
                  f"dyn={cost_dyn:.3e}  t*err={row['tstar_err']:+8.2f}s  "
                  f"{term:<12} it={len(log):>3}  lam_max={row['lambda_max']:.1e}  "
                  f"rej={row['n_rejected']:>2}  ({rt:.0f}s)", flush=True)

    with open(f"{OUT}/lm_variants.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    print("\n" + "=" * 96)
    print("LM VARIANT SCREEN  --  2 arcsec, angles-only, failing seeds")
    print("=" * 96)
    print(f"{'variant':<13}" + "".join(f"{'seed '+str(s):>26}" for s in SEEDS))
    for vname in VARIANTS:
        line = f"{vname:<13}"
        for s in SEEDS:
            r = next(x for x in rows if x['seed'] == s and x['variant'] == vname)
            line += f"{r['pos_rms']:>10.1f} m dyn={r['cost_dyn']:.1e}"
        print(line)


if __name__ == '__main__':
    main()
