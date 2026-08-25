# Angles-only solver investigation, 2026-08-25

Scripts and raw results behind every number in `ANGLES_ONLY_FGO.md`.
Run from the repo root with the project venv. All are scratch-quality: they
import from the repo root via `sys.path` and hardcode
`configs/config_geo_one_rev_deltaRIC0.5.yml`.

## The harness worth keeping

- `sweep_real.py`  — the main one. Calls the REAL `Orbit_FGO.opt()`, so repo
  changes are actually exercised. 10 seeds x {10", 2"} x {50, 300 iters} across
  10 cores in ~8 min. Iteration count is recovered by counting `create_L` calls
  rather than by instrumenting `opt()`.
      python analysis/angles_only_2026-08-25/sweep_real.py --tag mytest \
             --maxits 50 300 --workers 10
  Validated against the serial baseline: reproduces it exactly.
- `truth_cache.py` — loads truth from `out/*.csv` instead of re-propagating.
  Avoids `mc_fgo.propagate_truth`'s fixed `/tmp` path, which races when two
  processes run concurrently (see TODO.md).
- `tier0_diag.py`  — `InstrumentedFGO` + `build_seed`. **`sweep_real.py`
  imports `build_seed` from here**, so the two travel together. `build_seed`
  replicates `mc_fgo.run_fgo_seed`'s RNG draw order exactly, which is what makes
  seeds comparable with the Monte Carlo drivers.

## One-off diagnostics

- `term_probe.py`    — per-iteration `rel_pred` / `rel_act` / windowed cumulative.
                       This is where `CONV_REL_PRED = 1e-8` came from.
- `cond_test.py`     — solves one system three ways (spsolve+lam*I,
                       spsolve+lam*diag, lsmr on L) and compares steps.
- `lm_variants.py`   — the four LM variants A/B/C/D. Start here for the LM work,
                       but read ANGLES_ONLY_FGO.md 6 first: B/C/D were all
                       measured much worse, and all were confounded by the
                       then-broken stall counter.
- `lam_sweep.py`     — fixed-lambda sweep with diag damping.
- `maxiters300.py`   — the control that showed the failures were budget, not
                       local minima.
- `prof.py`          — profiles `create_L` / `F_man_mat` / `create_y`.
- `seed_sweep.py`, `seed_sweep_rounded.py`, `sweep300.py` — earlier sweeps that
  reimplement `opt()` rather than calling it. Superseded by `sweep_real.py`;
  kept only because `sweep300.json` is the pre-fix baseline.

## Results

`real_step1/2/3.json` are the regressions after each fix (line-search,
termination, Q units). `sweep300.json` is the pre-fix baseline. The `.json`
files carry per-iteration curves; the `.log` files are the readable summaries.
