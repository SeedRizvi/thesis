# Angles-Only FGO — Current Issue and Future Work

Status as of 2026-08-24. Written as a handoff; all numbers below are measured,
not estimated.

---

## 1. Where things stand

Angles-only estimation works. The large improvement this session came from the
solver fixes (Cholesky whitening, LM termination, prior), not from the
measurement model.

deltaRIC0.5, one revolution, FGO-G, angles-only, position RMS:

| angular noise | median | mean | notes |
|---|---|---|---|
| 10 arcsec (historical, pre-fixes) | ~85 km | — | what motivated this work |
| 10 arcsec (current) | 246 m | 321 m | 10 seeds, with prior |
| 2 arcsec (current) | 164 m | 495 m | 10 seeds, with prior, **4/10 unstable** |
| 1 arcsec | not tested | — | historical memory of ~80 m |

FGO-B (no manoeuvre estimation) behaves cleanly at 2 arcsec: -61% versus
10 arcsec, consistent across all seeds, no failures. The instability is
specific to FGO-G, i.e. to estimating `(delta-v, t*)` from angles alone.

Ground-station geometry (Rocky Point -> Siding Spring, with real altitudes) was
tested in isolation: -4.7% for FGO-B, -7.9% for FGO-G. Marginal. Not a priority.

---

## 2. The issue

### Symptom

At 2 arcsec, 4 of 10 seeds degrade badly (+340%, +291%, +108%, +108%) while the
other 6 improve substantially (-7% to -72%). Failure does not correlate with the
initial guess quality:

```
corr(|guess dv|,     paired change) = -0.110
corr(|guess t*|,     paired change) = -0.161
corr(pos_rms at 10", paired change) = -0.052
```

### The prior is NOT the cause

A full 2x2 ({prior, no prior} x {10", 2"}) over 10 seeds:

| | mean | median | failures >50% worse |
|---|---|---|---|
| no prior, 10"->2" | +762% | -44.5% | 2/10 |
| prior, 10"->2"    | +60%  | -15.7% | 4/10 |

Failures occur in both arms. The no-prior arm contains the single worst result
of the whole study (seed 4, 11,928 m). The prior reduces mean error 2.8x at
10 arcsec and 3.1x at 2 arcsec, and cuts the worst case ~5x. It trades a little
median performance for a much shorter tail — the same shrinkage behaviour seen
in the EKF P0 fix and the BLS prior work.

### Root cause: the Gaussian impulse does not match the truth manoeuvre

The truth propagator applies an **instantaneous** delta-v at t*. The FGO models
it as a **Gaussian of width epsilon = 30 s centred on t***. Because the pulse is
centred on t*, half the impulse is applied *before* the manoeuvre actually
occurs.

Evaluating the FGO's dynamics residuals at the TRUE trajectory with TRUE
manoeuvre parameters:

```
total dynamics cost at truth = 4.4566e+06
  within +/-3 steps of t*    = 4.4566e+06  (100.00%)
  everywhere else            = 9.7e-03

per-step position mismatch:
  median (smooth flight)     = 1.35e-04 m   <- Q is correctly calibrated here
  at t*                      = 36.67 m
  step before t*             = 8.96 m
```

`q_pos_ric[0] = 5.222e-04` is used as a variance, so sigma ~ 0.0229 m. The 36.67 m
mismatch is therefore a **~1600 sigma** violation of the process noise model at
that one step.

**This is NOT an integration error.** Sub-stepping the propagation converges
almost immediately:

```
n_timesteps   prop_dt (s)   err at t* step (m)
          1         60.00              36.6695
          5         12.00              36.1254
       1000          0.06              36.1250   <- converged
```

Only 0.54 m of 36.67 m (1.5%) was integration error. The remaining 98.5% is
model mismatch: a symmetric pulse centred on a step boundary always puts half
its impulse on the wrong side, regardless of how finely it is integrated or how
narrow it is made.

(An earlier note in TODO.md described this as an under-integration problem
fixable by `use_substep`. That was wrong — sub-stepping buys half a metre.)

### Why it only bites at low noise

Sharpening the measurements 5x makes `S_R_inv` 5x larger, so measurement rows
gain weight against dynamics rows — the only route through which delta-v and t*
enter the problem:

```
||L_dyn||_F / ||L_meas||_F :  4.670e+06 at 10"   ->   9.339e+05 at 2"
```

At 10 arcsec the dynamics term dominates, so the optimiser simply satisfies the
dynamics and treats the measurements as a weak nudge. At 2 arcsec the
measurements carry 25x more weight in the cost and there is genuine tension
between what they say and what the mis-specified dynamics insist on.

Cost at the converged solution, 2 arcsec:

| seed | outcome | cost dyn | cost meas | total |
|---|---|---|---|---|
| 1 | FAIL | 1.54e+03 | 9953 | 1.15e+04 |
| 4 | FAIL | 3.65e+04 | 9824 | 4.63e+04 |
| 3 | improve | 10.6 | 9790 | 9.80e+03 |
| 5 | improve | 7.9 | 9922 | 9.93e+03 |

Failing seeds abandon dynamic consistency (100-3600x higher dynamics cost) for
negligible measurement gain.

### Two effects are stacked

1. **Displaced objective.** Truth carries a 4.46e6 dynamics penalty, so the
   cost function's minimum is not at truth.
2. **Convergence failure.** Seed 4 converged to a solution 4.7x worse *in its
   own objective* than seed 3, on a comparable problem. That is a solver
   failure on top of the displaced objective.

Fixing the impulse model addresses (1) only. (2) is untested — the LM changes
and the lambda shrink rate (currently /2 while growth is *10) are the obvious
suspects and have not been investigated.

---

## 3. What fixes the model mismatch

The fix requires **narrow AND causal** together. Neither alone works:

- narrowing alone plateaus at ~26 m (symmetric pulse always splits at the boundary)
- shifting alone makes it worse (a wide pulse at the wrong time)

Measured, with pulse centred at `t* + 3*epsilon`:

| epsilon | delay | peak step err | dyn cost @truth | post-man offset |
|---|---|---|---|---|
| **current (30, no shift)** | — | **36.67 m** | **4.46e+06** | 9.90 m |
| 10 | 30 s | 26.05 m | 2.09e+06 | 25.96 m |
| 5 | 15 s | 13.06 m | 5.27e+05 | 12.94 m |
| 3 | 9 s | 7.87 m | 1.91e+05 | 7.74 m |
| 1 | 3 s | 2.67 m | 2.20e+04 | 2.54 m |
| 0.5 | 1.5 s | 1.37 m | 5.79e+03 | 1.25 m |

The residual error is then just the timing delay: peak error ~= |delta-v| * delay.
It scales linearly with epsilon and goes to zero.

**The catch:** resolving a narrow pulse numerically needs `prop_dt <~ epsilon/5`,
i.e. `n_timesteps ~ 300/epsilon`. At epsilon = 1 s that is 300x the propagation
cost — `create_L` goes from 0.71 s to ~150 s, i.e. 25-125 min per solve.
Prohibitive.

---

## 4. Proposed solutions

### Option A — analytic impulse integration (recommended)

Split the dynamics: RK4 the gravity as now, apply the impulse in closed form.
The Gaussian's contribution over a step is exactly integrable, so epsilon can be
made arbitrarily small at **zero** extra cost.

Over `[t0, t1]`, with `z = (t - t*)/epsilon`, `Phi` the normal CDF, `phi` the PDF:

```
dv_applied = dv * [Phi(z1) - Phi(z0)]

dr_applied = dv * { epsilon*[(z1*Phi(z1) + phi(z1)) - (z0*Phi(z0) + phi(z0))]
                    - (t1 - t0)*Phi(z0) }
```

Both formulas verified against brute-force quadrature to 1e-14.

Code changes (three touches in `Orbit_FGO.py`):

1. `orbital_dynamics` — drop the impulse term, leave gravity only.
2. New `_impulse_increment(t0, t1)` returning `(dr, dv)` from the formulas above,
   using `t_star + impulse_delay` for the causal offset.
3. `prop_one_timestep` — RK4 the gravity, then add `dr` to position and `dv` to
   velocity.

`n_timesteps` stays 1, `create_L` stays at ~0.7 s, and all the sub-stepping
machinery becomes unnecessary.

**Caveats to handle:**

- **Operator splitting** neglects the coupling between the impulse-induced
  position change and the gravity felt within the step. Estimated ~1 mm against
  Q's 2.3 cm sigma. Measure it against a heavily sub-stepped reference rather
  than assuming.
- **`FD_TSTAR_STEP = 0.01 s`** was sized as 0.033% of epsilon = 30 s. At
  epsilon = 0.1 s it becomes 10% of the pulse width, far too coarse for the t*
  Jacobian. Tie it to epsilon instead of leaving it fixed.
- **The causal offset is a known bias.** With the pulse centred at
  `t* + 3*epsilon`, the estimated t* is offset from the physical burn time by
  `3*epsilon`. At epsilon = 0.1 s that is 0.3 s — negligible but it must be
  documented or subtracted, or reported t* errors carry a constant offset.

### Option B — sub-step only near t*

This is the existing `use_substep`, previously rejected on the grounds that it
requires detecting the manoeuvre in order to integrate it.

Worth revisiting: in FGO-G, `t*` is an **explicitly estimated parameter**. The
model already asserts where the burn is; refining the integration around your own
model's stated burn time is not detection, it is integrating the model you have
committed to. FGO-B needs none of this since it has no impulse to integrate.

Cheaper to implement than A, but leaves the sub-stepping machinery in place and
keeps a runtime cost near t*.

### Option C — moderate epsilon reduction only

epsilon = 5 s with `n_timesteps = 60`: mismatch 2.8x better, dynamics cost 8.5x
better, for ~30x runtime. Poor exchange rate. Listed for completeness.

### Rejected

- **Inflate Q near t***: statistically standard, but circular in the same way
  Option B was objected to, and rejected as a hack.
- **Generate truth with the same Gaussian**: removes the mismatch by
  construction, but a 30 s smeared burn defeats the point of demonstrating
  robustness against an impulsive manoeuvre.

---

## 5. Open questions for next session

1. **Does fixing the model fix the instability?** Option A makes truth far more
   plausible under the cost (dynamics cost at truth 4.46e6 -> ~2e4), but the
   convergence failure identified in section 2 is a separate effect. Re-run the
   10 seeds at 2 arcsec after implementing.
2. **What happens at 1 arcsec?** Untested. The historical ~80 m target is at
   1 arcsec, and the trend from 10 -> 2 suggests it is reachable. But the
   weighting shift becomes 5x more severe again, so the instability may worsen.
   Measure before deciding how much of Option A to build.
3. **Is the LM lambda shrink rate implicated?** Currently /2 on an accepted step
   while growth is *10. Symmetric *10 / /10 is the standard LM form and was
   decisive on short-arc angles-only (13,008 m at /2 vs 1,892 m at /10), but was
   not adopted because short-arc is the config built to favour BLS. This is the
   leading candidate for the convergence half of the problem. See TODO.md.
4. **Alternatives raised with supervisor** (not yet investigated): triangulating
   ground-station angles into a position region and feeding that as the
   measurement, or moving to spacecraft-to-spacecraft angles.

---

## 6. Validation tooling

A `check_jacobian` debugging tool was written this session (scratchpad only, not
in the repo). It verifies `create_L` against a finite difference of `create_y`,
exploiting `y(x + d) ~= y(x) - L*d`, so `dy/dx_j` must equal `-L[:, j]`.

Baseline results on the current code (both with and without the prior):

```
short arc, 1300 cols exhaustive : max rel discrepancy 1.07e-08
one_rev,   9946 cols exhaustive : max rel discrepancy 1.09e-08,  0 cols > 1e-3
```

Option A changes how the impulse enters both `create_y` and `create_L`, so this
check is essential after implementing it. The reference implementation the FGO
was ported from had an equivalent (`reference_fgo_.py:315 test_Jacobian`) which
was lost in the port — worth re-adding to the repo permanently.
