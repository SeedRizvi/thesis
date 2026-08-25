# Angles-Only FGO — Current Status

Status as of 2026-08-25. All numbers below are measured on
`configs/config_geo_one_rev_deltaRIC0.5.yml`, angles-only, FGO-G, with prior,
10 seeds, unless stated otherwise.

**Every number in this document can be reproduced from
`analysis/angles_only_2026-08-25/`**, which holds the scripts that produced them
and the raw per-iteration results. See section 8.

**This revision supersedes the 2026-08-24 version.** Its central claim — that
2 arcsec causes a catastrophic instability in 4 of 10 seeds — was wrong. The
failures were an iteration-budget artefact. See section 2.

---

## 1. Where things stand

Angles-only estimation works, and sharper measurements help uniformly.

Converged (`max_iters` raised until every seed terminates on its own),
after the Q units fix, over the 9 seeds that converge in both versions:

| | before Q fix | after Q fix | |
|---|---|---|---|
| 10 arcsec mean | 232.96 m | **160.02 m** | -31% |
| 2 arcsec mean | 126.29 m | **55.53 m** | -56% |
| 2 arcsec median | 121.3 m | **50.6 m** | -58% |

The 10" -> 2" improvement is **-65%** after the fix (-46% before it).

Every seed improves from 10" to 2". There is no instability.

---

## 2. The "catastrophic failure" was the iteration budget

`max_iterations: 50` was too small for roughly a third of 2 arcsec seeds.
The same 20 runs, scored at iteration 50 versus at convergence (pre-Q-fix):

| budget | 10" mean | 2" mean | apparent effect of sharper measurements |
|---|---|---|---|
| 50 iters | 321.1 m | 323.8 m | **+0.8%** — no benefit, 3 blow-ups |
| converged | 234.3 m | 125.7 m | **-46%** — uniform benefit |

Same code, same seeds, same data. Only the cap differs.

The three "failures" at 2 arcsec converge cleanly given budget:

| seed | @50 iters | converged | iters needed |
|---|---|---|---|
| 1 | 765.04 m, `dyn 1.6e3` | 112.53 m, `dyn 1.0e1` | 141 |
| 7 | 1031.44 m, `dyn 3.1e3` | 94.05 m, `dyn 1.3e1` | 64 |
| 8 | 511.74 m, `dyn 7.0e3` | 120.54 m, `dyn 1.3e1` | 72 |

All three land in the same basin as the seeds that converge inside 50
iterations, at the *good* end of the distribution. There is no second basin and
no local minimum. The 10 arcsec arm is affected too: seed 7 reads 1089.91 m at
50 iterations and 221.56 m converged, so the previously published 246 m median
was itself depressed by an unconverged seed.

**Caveat: `cost_dyn` alone does not indicate convergence.** `create_init_state`
builds the initial trajectory with the same propagator the dynamics residual
uses, so that trajectory is dynamically self-consistent by construction
(`cost_dyn` ~ 1e-6) while being ~24 km from truth — a 1 m/s initial velocity
error integrates to ~99 km over the 27.6 h arc. Low `cost_dyn` is only
meaningful alongside low position error.

---

## 3. Defects found and fixed

### 3.1 Q was a standard deviation consumed as a variance (HIGH impact)

`calibrate_q_ric.py` emits `5 * sqrt(mean(err**2))` — five times the RMS
mismatch in **metres**, a standard deviation. `compute_S_Q_inv` placed it
directly into the diagonal of Q, where entries are variances.

```
effective sigma was sqrt(5.222e-4) = 0.0229 m, not 5.222e-4 m
position: 43.8x / 52.3x / 67.9x too loose   (R / I / C)
velocity: 239.6x / 287.5x / 372.0x too loose
```

Cross-check confirming the intended reading: `q_pos/5 = 1.044e-4 m` matches the
measured smooth-flight per-step mismatch of 1.35e-4 m. Read as a variance, the
implied sigma would have been 170x the mismatch it was calibrated against.

The inflation factor is `1/sqrt(q)`, so it differed per axis and **flattened the
R:I:C anisotropy** that `calibrate_q_ric.py` exists to measure: position
1:0.701:0.415 became 1:0.837:0.644, velocity likewise. A deliberate safety
factor would have been uniform.

Fixed by squaring in `compute_S_Q_inv` **and** in `Orbit_EKF.compute_Q`, which
carries its own duplicate of the same code. Configs now state that the values
are standard deviations.

**Effect:** large accuracy gain (section 1) but a markedly stiffer problem —
see section 4.

**Note:** this changes EKF and BLS results too. Those Monte Carlos have not been
re-run.

### 3.2 Termination criteria unsuited to the problem

Both tests in `opt()` were replaced:

- `la.norm(delta_x * best_scale) < 1e-3` summed metres, m/s and seconds over
  9,946 variables — dimensionally incoherent, and so tight it rarely fired.
- `stalled`: 10 *consecutive* iterations each below 1e-6 relative cost. This
  terminated any well-damped variant at exactly iteration 10, before it could
  do anything.

Replaced with `CONV_REL_PRED` on the model's predicted relative cost reduction
(dimensionless, hence invariant to how Q and R scale the whitened rows) plus a
*cumulative* windowed stagnation backstop. Measured separation justifying the
1e-8 threshold:

```
still making progress : rel_pred = 1e-3 .. 1e-1
converged             : rel_pred = 1e-11 .. 1e-13
```

**Effect:** identical answers (max change 0.01 m over 40 runs), 7-10% fewer
iterations, 16-22% less CPU.

### 3.3 Line search could discard a better step

The acceptance branch overwrote `best_scale` unconditionally, so a smaller
scale passing the ratio test could replace a larger scale that had already
achieved a lower cost. Fixed by deleting the assignment; the tracking above it
was already correct.

**Effect:** fired on 4 of 20 runs but changed only the path, never the
destination (0/20 seeds changed position error), costing 1-6 extra iterations.
Kept as a correctness fix, not an improvement.

---

## 4. What remains broken

### 4.1 The solver diverges on stiff seeds

After the Q fix:

| budget | 10" diverged | 2" diverged |
|---|---|---|
| 50 iters | 3/10 | 3/10 |
| 300 iters | 1/10 (seed 8) | 1/10 (seed 8) |

Seed 8 fails in both arms — 300 iterations ending at `cost_dyn` 6.0e4 (10") and
9.4e5 (2"), against ~1e-2 for converged seeds. It converged in 23 iterations
before the Q fix. Iteration counts rose broadly (10" s10 18->150, 2" s6
48->187), which is the expected cost of dynamics rows 44x-240x stronger.

### 4.2 `opt()` is not Levenberg-Marquardt

Measured across 20+ runs: `lambda_max = 5.0e-07`. Lambda never rises above its
initial 1e-6 and reaches the 1e-10 floor by iteration ~13. The growth branch
only fires when `best_scale == 0` — the line search failing across all 20
halvings — which never happened (`rejections = 0` in every run). **The solver
is Gauss-Newton with backtracking; the damping is vestigial.**

Curvature at iteration 0 (2 arcsec, `M.diagonal()`):

```
pos   median  5.18e+03
vel   median  9.49e+06
dv    median  1.35e+06
t*            2.53e+02     <- 3.7e4x smaller than velocity
```

So if lambda ever did grow, `lambda * I` would annihilate `t*` first — the one
parameter being estimated. Any fix to the lambda update must come with
curvature-scaled damping, not after it.

### 4.3 The objective is a long flat valley

For seed 1 at 2 arcsec, between iteration 50 and convergence:

```
cost      11519  ->  ~10000     (-13%)
pos RMS     765  ->     112     (-85%)
```

At iteration 50 the whitened residual RMS is already 0.76 sigma — the
measurements are essentially satisfied while the trajectory is 765 m wrong.
This explains why Gauss-Newton crawls, why cost-based termination is a poor
proxy, why results differ slightly between machines (BLAS/SuperLU rounding),
and why sharpening 10" -> 2" helps: it steepens the valley.

### 4.4 The impulse model is still mis-specified

Unchanged from the previous revision and still the floor on accuracy. The truth
propagator applies an **instantaneous** delta-v at t*; the FGO models it as a
Gaussian of width epsilon = 30 s **centred** on t*, so half the impulse is
applied before the manoeuvre occurs.

```
total dynamics cost at truth = 4.4566e+06
  within +/-3 steps of t*    = 100.00%
per-step position mismatch:
  median (smooth flight)     = 1.35e-04 m
  at t*                      = 36.67 m
  step before t*             = 8.96 m
```

**Not an integration error.** Sub-stepping converges almost immediately
(36.6695 m at `n_timesteps=1` -> 36.1250 m at 1000); only 0.54 m of 36.67 m
(1.5%) was integration error.

**New finding: t\* sits exactly on the dt grid.** `t_star_true = 12960.0 s`
= 216 x 60 exactly, because `mc_fgo.propagate_truth` takes it as the last epoch
of the pre-manoeuvre propagation. A symmetric pulse centred on a grid node puts
half its impulse a full step early:

```
0.5 * |dv| * dt = 0.5 * 0.866025 * 60 = 25.981 m
```

which is precisely the "narrowing alone plateaus at ~26 m" result. **The
plateau is a grid-boundary artefact, not a fundamental limit of narrow pulses.**

The fix still requires narrow AND causal together. With the pulse centred at
`t* + delay`, the residual error is just the timing delay, `|dv| * delay`,
confirmed to ~5% across the measured range:

| epsilon | delay | measured | 0.866 x delay |
|---|---|---|---|
| 10 | 30 | 26.05 m | 25.98 |
| 5 | 15 | 13.06 m | 12.99 |
| 3 | 9 | 7.87 m | 7.79 |
| 1 | 3 | 2.67 m | 2.60 |
| 0.5 | 1.5 | 1.37 m | 1.30 |

---

## 5. Option A — analytic impulse integration (still the plan)

Integrate the impulse in closed form instead of letting RK4 sample it. The
continuous-time model is unchanged; only the integration scheme changes.

Over `[t0, t1]`, with `z = (t - t_c)/epsilon`, `t_c = t* + delay`:

```
dv_applied = dv * [Phi(z1) - Phi(z0)]

dr_applied = dv * epsilon * [z1*Phi(z1) + phi(z1) - z1*Phi(z0) - phi(z0)]
```

Verified against quadrature to 1e-14 and derived independently. The Jacobians
are then closed-form and **exact for the discrete map** (under splitting,
`x1 = RK4_grav(x0) + Delta(p)`, so `dx1/dp = dDelta/dp` with no chain rule
through the integrator):

```
d(dv_applied)/d(dv) = [Phi(z1) - Phi(z0)] * I3
d(dv_applied)/dt*   = -(dv/epsilon) * [phi(z1) - phi(z0)]
d(dr_applied)/dt*   =  dv * { ((t1-t0)/epsilon)*phi(z0) - [Phi(z1) - Phi(z0)] }
```

### Why RK4 cannot resolve a narrow pulse

With `n_timesteps = 1`, RK4 evaluates the dynamics at only **three distinct
times** (`t`, `t + dt/2`, `t + dt`); k2 and k3 share a time, and the impulse
depends on time only. Their weights add, so the applied impulse is

```
dv_applied = (dt/6) * [g(t0) + 4*g(tm) + g(t1)] * dv
```

— exactly **Simpson's rule**. At epsilon = 30, dt = 60 that is adequate (the
measured 0.54 m). At epsilon = 1 the three nodes are 30 s apart on a 6 s pulse,
so the result is unrelated to the true integral. This is also why the `t*`
Jacobian is currently a finite difference *of a quadrature* rather than of the
true increment.

### Cost

`F_man_mat` is **26.1% of `create_L`** (0.235 s of 0.899 s), i.e. 9.6-18.9% of a
full iteration depending on backtracking. Real, but a constant factor — it does
not change iteration counts.

### Caveats to handle

- **Operator splitting** neglects impulse/gravity coupling within a step.
  Estimated ~1 mm against Q's sigma. **Measure it** against a heavily
  sub-stepped reference; do not assume. Fallback is Strang splitting, which
  costs the exact-Jacobian property.
- **`FD_TSTAR_STEP = 0.01`** disappears entirely under analytic Jacobians.
- **The causal offset is a known bias**: estimated t* is offset from the
  physical burn time by `delay`. Document or subtract it.

Keep `orbital_dynamics` computing the full field behind an `include_man` flag —
it is required as the reference for the splitting-error measurement, and it
keeps `a_total = a_2body + a_J2 + a_man` intact in the source and the
manuscript.

---

## 6. Hypotheses tested and rejected

Recorded so they are not retried.

| hypothesis | result |
|---|---|
| The prior causes the 2" failures | No. Failures occur with and without it; the prior cuts mean error 3.1x and the worst case ~5x. |
| Failure correlates with initial-guess quality | No. corr = -0.110 (dv), -0.161 (t*), -0.052 (pos RMS at 10"). |
| Sub-stepping fixes the manoeuvre-step error | No. 0.54 m of 36.67 m. |
| It's a local minimum / second basin | No. All seeds reach the same basin given budget. |
| lambda grows and freezes t* | No. lambda never grows at all. |
| `lambda*diag(M)` damping helps | **Much worse**: 2669 / 11315 / 2211 m vs 765 / 1031 / 512 m baseline. |
| Reactive lambda (grow on backtracking) | **Much worse** — but confounded with diag damping and the broken stall counter. Untested standalone. |
| Textbook Nielsen LM, no line search | **Much worse**: 5613 / 35636 / 5891 m. Same confounds. |
| Fixed lambda with diag damping | **Much worse** at every value 1e-6..1.0. Runs at lambda >= 1e-2 died at exactly iteration 10 on the stall counter, so those rows are uninformative. |
| Normal equations lose all precision | No. `||(M+lam*I)dx - L'y|| / ||L'y|| = 9.2e-13`; spsolve is accurate. But LSMR needs >20,000 iterations without converging, implying cond(L) ~ 5e4, cond(M) ~ 3e9 — badly conditioned, not catastrophically. |

---

## 7. Open questions

1. **Can the solver be made to converge within 50 iterations?** This is the
   binding operational constraint. Target: eliminate divergences at
   `max_iters = 50`, with accuracy on converged seeds matching the post-Q
   figures (10" 160 m, 2" 55.5 m).
2. **Why does seed 8 diverge post-Q even at 300 iterations?** New failure mode,
   not a slow-convergence case.
3. **What happens at 1 arcsec?** Untested. Historical target ~80 m; the post-Q
   2" result of 55.5 m already beats it.
4. **Does Option A change the floor?** Only measurable once the solver
   converges reliably, or model improvement and convergence luck are confounded.
5. **Should the Q magnitude be revisited?** The units are now right; whether 5x
   RMS is the right margin is a separate, deliberate question.
6. **Alternatives raised with supervisor** (not investigated): triangulating
   ground-station angles into a position region as the measurement, or
   spacecraft-to-spacecraft angles.

---

## 8. Tooling and reproducing these results

Everything is in **`analysis/angles_only_2026-08-25/`** (see its README for a
file-by-file description). The three that matter:

- `sweep_real.py` — the main harness. 10 seeds x {10", 2"} x {50, 300
  iterations} across 10 cores in ~8 minutes, calling the **real**
  `Orbit_FGO.opt()` so that repo changes are actually exercised. Validated to
  reproduce the serial baseline exactly.

      python analysis/angles_only_2026-08-25/sweep_real.py --tag mytest \
             --maxits 50 300 --workers 10

- `tier0_diag.py` — supplies `build_seed`, which `sweep_real.py` imports. It
  replicates `mc_fgo.run_fgo_seed`'s RNG draw order exactly, which is what makes
  these seed numbers comparable with the Monte Carlo drivers.
- `truth_cache.py` — loads truth from `out/*.csv` rather than re-propagating,
  and sidesteps the fixed `/tmp` path that makes `mc_fgo.propagate_truth` race
  under parallel use.

Raw results: `sweep300.json` is the pre-fix baseline; `real_step1/2/3.json` are
the regressions after the line-search, termination and Q-units fixes
respectively. The JSON files carry per-iteration curves, the `.log` files the
readable summaries.

Caveat recorded in the README: `seed_sweep.py`, `seed_sweep_rounded.py` and
`sweep300.py` **reimplement** `opt()` instead of calling it. They are superseded
by `sweep_real.py` and will not reflect solver changes made after 2026-08-25.

`check_jacobian` (verifies `create_L` against a finite difference of
`create_y`, exploiting `y(x + d) ~= y(x) - L*d`) was **not** carried over and
should be added to the repo permanently (see TODO.md) — it is essential before
Option A changes how the impulse enters both functions. Baseline on current
code:

```
short arc, 1300 cols exhaustive : max rel discrepancy 1.07e-08
one_rev,   9946 cols exhaustive : max rel discrepancy 1.09e-08,  0 cols > 1e-3
```
