# Angles-Only FGO — Current Status

Status as of 2026-08-26. Angles-only, 10 seeds, with prior, unless stated
otherwise. Sections 1-4 are `configs/config_geo_one_rev_deltaRIC0.5.yml` with
FGO-G; section 5 spans `deltaRIC0` and `deltaRIC0.5` x {FGO-B, FGO-G}.

**Every number in this document can be reproduced from
`analysis/angles_only_2026-08-25/`**, which holds the scripts that produced them
and the raw per-iteration results. See section 9.

**This revision supersedes the 2026-08-24 version.** Its central claim — that
2 arcsec causes a catastrophic instability in 4 of 10 seeds — was wrong. The
failures were an iteration-budget artefact. See section 2.

---

## 1. Where things stand

**The orbit determination core is solved. Everything still open is specific to
manoeuvre estimation.**

Mean position RMS over converged seeds (section 5):

| noise | OD floor `RIC0/B` | with manoeuvre estimation `RIC0.5/G` |
|---|---|---|
| 10" | 91.1 m (9 iters, 10/10) | 160.0 m (82 iters, 9/10) |
| 2" | 29.8 m (6 iters, 10/10) | 55.5 m (78 iters, 9/10) |
| 1" | **18.1 m** (5 iters, 10/10) | **31.8 m** (27 iters, 10/10) |

Angles-only OD is clean: no divergences at any noise level, monotonic in noise,
and 5-9 iterations against a budget of 50. The 1 arcsec result comfortably
passes the historical ~80 m target.

Manoeuvre estimation costs 53-86% in accuracy and 3-13x in iterations, and it
is where every remaining failure lives. The three mechanisms are `t*`
observability (4.2b), the hinge that decouples the pre-manoeuvre arc (5.4), and
the mis-specified impulse model (4.4).

**But roughly half that cost is scenario geometry, not method.** The baseline
places the burn 3.6 h into a 27.6 h arc -- only 13% of the data before it.
Moving it to mid-arc, at the same total arc length (5.8):

| noise | burn at 13% | burn at 50% | |
|---|---|---|---|
| 2" | 55.5 m, 9/10, 78 iters | **24.8 m, 10/10, 28 iters** | -55% |
| 1" | 31.8 m, 10/10, 27 iters | **13.6 m, 10/10, 33 iters** | -57% |

At the symmetric epoch the manoeuvre-estimation penalty largely vanishes: 24.8 m
against an OD floor of 29.8 m at 2 arcsec. That comparison is not matched (the
floor was measured at the baseline epoch) and needs a `RIC0`/FGO-B control at
the symmetric epoch before it can be claimed.

Earlier progress on the `RIC0.5`/FGO-G configuration, for reference:

| | before Q fix | after Q fix | |
|---|---|---|---|
| 10 arcsec mean | 232.96 m | 160.02 m | -31% |
| 2 arcsec mean | 126.29 m | 55.53 m | -56% |

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

After the Q fix, on `RIC0.5`/FGO-G with the shipped solver:

| budget | 10" diverged | 2" diverged |
|---|---|---|
| 50 iters | 3/10 | 3/10 |
| 300 iters | 1/10 (seed 8) | 1/10 (seed 8) |

Three things measured since have narrowed this considerably:

- **It is confined to manoeuvre estimation.** `RIC0`/FGO-B converges 10/10 in
  5-9 iterations at every noise level (5.1). FGO-B on `RIC0.5` likewise
  converges 10/10 in 5-9 iterations (5.3). Only FGO-G is slow or unstable.
- **Removing the vestigial damping fixes most of it** — pure Gauss-Newton takes
  2 arcsec to 0/10 divergences at 300 iterations (4.2), including seed 8.
- **Prior strength is a convergence dial** (5.6c): a tighter prior converges
  faster and more reliably at an accuracy cost. `p0 = 0.25` is the only setting
  measured reaching 10/10 on `RIC0.5`/FGO-G, in 56 iterations, at +42% error.

So the divergences are not one defect. They are the combination of a
weakly-observable `t*`, damping that suppresses the informative directions, and
a deliberately loose prior.

### 4.2 Levenberg-Marquardt is inapplicable to this problem (RESOLVED 2026-08-26)

`opt()` is not LM: `lambda_max = 5.0e-07` across 20+ runs, lambda never rises
above its initial 1e-6, and the growth branch only fires when the line search
fails across all 20 halvings, which never happens. Post-Q the median `M`
diagonal is 3.6e9, so lambda = 1e-6 is a **relative** damping of 2.8e-16 —
below machine epsilon (2.2e-16). The shipped solver is *exactly* Gauss-Newton
with backtracking.

**That turns out to be the right answer, not a bug.** Damping was swept by
relative strength `tau` so the two forms are comparable
(`M + tau*diag(M)` vs `M + tau*median(diag(M))*I`), 2 arcsec, maxit 50:

```
form            tau          s1          s3          s5          s6          s8   div
GN            0e+00       40.2        69.2       105.0        25.2     15866.4*     1
diag          1e-12    20916.8*   168810.5*    62764.4*   111052.5*    20710.8*     5
diag          1e-09    23132.0*   169719.5*    62851.0*   111964.2*    21083.9*     5
diag          1e-06    24374.9    172340.8     64201.5    112585.9     21372.2      0
diag          1e-03    24376.2    172351.9     64205.6    112593.2     21373.5      0
identity      1e-09    24370.3    172316.0     64191.1    112568.2     21369.1      0
identity      1e-06    24376.2    172351.9     64205.6    112593.2     21373.5      0
```

Read the damped rows against the initial guess: seed 1 starts at **24376.2 m**,
and `tau = 1e-3` returns 24376.2 m after **one iteration**. Every damped
configuration converges back onto the starting point. Even `tau = 1e-12` —
one part in a trillion, relative — freezes the solve, implying the informative
directions have relative curvature below 1e-12, i.e. `cond(M) >~ 1e12`.

**Why LM fails.** In the eigenbasis of `M` the GN step along direction *i* is
`g_i/mu_i`; the LM step is `g_i/(mu_i + tau*d_i)`. Damping leaves steep
directions alone and suppresses shallow ones by `mu_i/(tau*d_i)`. LM's premise
is that low-curvature directions are where the quadratic model is least
trustworthy. **That premise is false here**: angles constrain direction but
barely constrain range, so range-like directions are intrinsically shallow;
`t*` sits at the bottom of the curvature spectrum (2.53e2 against velocity's
9.49e6); and section 4.3 states it outright — a 13% cost change corresponds to
an 85% error change, so the directions that reduce *error* are exactly those
that barely reduce *cost*. LM damps out precisely the directions that must move.

**The statistical argument is stronger and belongs in the manuscript.** The FGO
already carries a prior: `S_P0_inv` contributes rows to `L`. That is the
principled regularisation of weakly-observable directions, with a calibrated
covariance. Adding `lambda*I` on top is a second regulariser with no statistical
meaning — it pulls toward zero rather than toward the prior mean, weighted by
solver heuristics rather than any uncertainty model. It double-counts
regularisation and corrupts a correctly-specified Bayesian estimator.

**Why Gauss-Newton with a line search works: direction versus length.** A line
search scales the step but *preserves its direction*, keeping the
valley-following direction and taking a safe fraction of it. LM damping
*rotates* the step toward the gradient and toward high-curvature directions —
in a long flat valley, that means turning to point across the valley instead of
along it. GN's classic failure (an over-long step outside the linearisation's
validity) is real here, and the backtracking does genuine work (median accepted
scales 1/16 to 1/128 on hard seeds) — but that is a **length** problem, and the
line search is the correct instrument. LM applies a direction-changing remedy
to a length problem.

**Measured effect of removing the damping entirely** (10 seeds, both arms):

| | shipped: div / mean* / iters | pure GN: div / mean* / iters |
|---|---|---|
| 10" @50 | 3/10 · 145.5 m · 40 | 3/10 · **131.6 m** · 40 |
| 2" @50 | 3/10 · 52.8 m · 35 | **1/10** · 55.5 m · **24** |
| 10" @300 | 1/10 · 160.0 m · 82 | 1/10 · 159.4 m · 74 |
| 2" @300 | 1/10 · 55.5 m · 78 | **0/10** · **53.5 m** · **40** |

*over converged seeds only.

**At 2 arcsec / 300 iterations all ten seeds converge, mean 53.48 m, median
43.88 m** — the best result recorded for this configuration. Seed 8 goes from
2827.9 m diverged to 35.1 m converged. At 50 iterations the 2 arcsec arm goes
3 divergences to 1 (seeds 5 and 6 fixed) with iterations down 35 -> 24.

**Not uniformly better.** At 10 arcsec / 50 iterations GN fixes seed 10 but
breaks seed 2 (198.4 m converged -> 4362.6 m diverged); seed 2 converges fine
under GN at 300 iterations, so it is slower, not broken. The 10 arcsec arm at
50 iterations is a wash on divergence count, 3/10 either way. GN is clearly
better at 2 arcsec and neutral at 10 arcsec.

**Not yet applied to the repo.**

### 4.2b The t* error budget dominates the residual

`t*` enters the cost only through dynamics residuals within +/-3*epsilon = +/-90 s
of the manoeuvre — roughly 3-4 steps, i.e. **~20 of 19,898 residual rows
(0.1%)**. But it displaces the entire post-manoeuvre arc, which is 86,400 s of
the 99,360 s total (**87% of the trajectory**). One variable determined by 0.1%
of the cost sets most of the error.

Measured, converged runs only:

```
corr(|t*_err|, pos_rms) = +0.54 .. +0.70, consistent across both arms
                          and both iteration budgets

mean |t*_err| = 50.4 s  ->  0.866 m/s x 50.4 s = 44 m predicted displacement
observed position RMS                          = 53.5 m
```

The magnitude accounts for most of the error budget. `t*` is **unbiased but
imprecise**: mean signed error is only +8.0 s while the scatter is ~60 s, so the
issue is `t*` observability, not a systematic offset from the centred-pulse
mis-specification.

This is the concrete mechanism behind section 4.3's "13% cost change, 85% error
change", and it makes `t*` precision the highest-value lever on angles-only
FGO-G accuracy.

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

## 5. Isolation study: what each component costs

Motivation: `t*` dominates the error budget (4.2b), so the OD core cannot be
characterised while it is in play. And the original implementation was
validated with **angles + range**; removing range removed the only direct
observation of the range dimension, so base OD needed re-validating in its own
right.

Design: `RIC0` (no manoeuvre) and `RIC0.5` (0.866 m/s burn) x {FGO-B, FGO-G},
10 seeds, 10"/2"/1", `max_iters = 300` with the 50-iteration result read off
the recorded curve. RIC0 truth was checked for smoothness at the pre/post
propagation seam (deviates 0.18 m from the arc median; the worst step anywhere
is 7.1 m at an unrelated index, so there is no restart artefact). `build_seed`
skips the dv/t* RNG draws in FGO-B exactly as `mc_fgo` does, so FGO-B and FGO-G
for a given seed share identical measurements and identical x0.

### 5.1 The angles-only OD core is clean

`RIC0` / FGO-B:

| noise | converged | mean | median | max | iters |
|---|---|---|---|---|---|
| 10" | **10/10** | 91.1 m | 89.6 | 137.0 | **9** |
| 2" | **10/10** | 29.8 m | 29.6 | 34.6 | **6** |
| 1" | **10/10** | 18.1 m | 18.3 | 20.1 | **5** |

No divergences, monotonic in noise, and **5-9 iterations** against a budget of
50. The OD core was never the problem.

### 5.2 What manoeuvre estimation costs

Mean position RMS over converged seeds:

| noise | OD floor `RIC0/B` | false alarm `RIC0/G` | real manoeuvre `RIC0.5/G` | not estimating `RIC0.5/B` |
|---|---|---|---|---|
| 10" | 91.1 | 158.3 (+74%) | 160.0 (+76%) | 1705.1 (**11x** worse) |
| 2" | 29.8 | 45.6 (+53%) | 55.5 (+86%) | 1553.0 (**28x** worse) |
| 1" | 18.1 | 28.1 (+55%) | 31.8 (+76%) | 1492.7 (**47x** worse) |

- **Estimating a manoeuvre costs 53-86% whether or not one exists.** Most of the
  penalty is the *act of estimating*, not the manoeuvre.
- **Manoeuvre estimation buys 11-47x**, and the benefit grows as measurements
  sharpen — a quantitative justification for FGO-G.
- **1 arcsec works**: FGO-G on RIC0.5 gives 31.8 m, 10/10 converged, well past
  the historical ~80 m target. (Open question 3, answered.)

### 5.3 An unmodelled manoeuvre is a bias, not a divergence

`RIC0.5` / FGO-B converges 10/10 in 5-9 iterations at every noise level, and at
2 arcsec the spread across seeds is `std/mean = 0.1%` (min 1550.1, max 1556.1).
That is a **systematic bias**, near-identical on every seed — not instability.

The confirming signature is noise insensitivity:

```
RIC0.5/FGO-B   10" -> 1":  1705 -> 1493 m   (-12%)
RIC0/FGO-B     10" -> 1":    91 ->   18 m   (-80%)
```

Ten times better measurements buy 12%, because the error is model bias rather
than measurement variance. The superseded revision recorded FGO-B on RIC0.5 at
-61% from 10"->2"; it is now -9%, consistent with the Q fix turning a
partly-absorbable event into a hard bias (trend only -- the pre-Q absolute
numbers were never recorded).

### 5.4 The manoeuvre parameters act as a hinge, and the pre-arc pays

The false-alarm penalty falls **entirely on the pre-manoeuvre arc**:

| | `RIC0/B` pre -> `RIC0/G` pre | `RIC0/B` post -> `RIC0/G` post |
|---|---|---|
| 10" | 104.1 -> 344.2 (**3.3x**) | 86.2 -> 89.7 (unchanged) |
| 2" | 42.8 -> 103.5 (**2.4x**) | 26.7 -> 26.2 (unchanged) |
| 1" | 25.4 -> 60.6 (**2.4x**) | 16.4 -> 17.9 (unchanged) |

Mechanism: the manoeuvre parameters insert a **hinge** at `t*` that partially
decouples the two halves of the trajectory. The post arc is 1441 of 1657 steps
(87% of the measurements) and determines itself. The pre arc is only 216 steps
(13%), and once decoupled it must be pinned down by that 13% plus **the prior**.

This is compounded by two changes made since the original validation: there was
previously **no prior**, and there **was range**. Range gave direct
observability of the range dimension, which could override a poor initial
estimate. Now the prior actively *enforces* the initial guess while the
measurement type that could have overruled it has been removed. A large-error
prior is therefore far more consequential than it was.

### 5.5 Iterations

Mean iterations to convergence:

```
             RIC0/B   RIC0/G   RIC0.5/B   RIC0.5/G
   10"            9       15          9         82
    2"            6       10          6         78
    1"            5        8          5         27
```

The manoeuvre parameters are the entire convergence difficulty. Everything
except FGO-G-with-a-real-manoeuvre fits comfortably inside 50 iterations. At
1 arcsec even that mostly does: 9 of 10 seeds finish inside 50; seed 8 needs 97
(converging to 24.2 m, but reading 10579 m at iteration 50).

### 5.6 Phase 4: the prior is a convergence dial, not an accuracy lever

Two knobs, deliberately separated. `sigma_scale` moves the sampled perturbation
**and** the prior together (the self-consistent scenario axis: "the initial
state is better known"). `p0_scale` moves the prior **only**, leaving the
sampled error at sigma = 1000 m / 1.0 m/s per axis -- a deliberate
mis-specification. Seeds are exactly paired: numpy scales the same standard
normals, so seed 3 at sigma_scale 0.5 gets precisely half the error it gets at
1.0, and `p0_scale` changes nothing but P0. 240 runs, 2 arcsec.

**All significance below is a paired per-seed test; `*` marks > 2 SE.**

#### 5.6a Better initial knowledge helps the pre-arc only

sigma_scale 1.0 -> 0.25 (four times better initial knowledge):

| config | overall | pre | post |
|---|---|---|---|
| RIC0/FGO-B | -0.1% n.s. | -0.7% n.s. | +0.1% n.s. |
| RIC0/FGO-G | -6.6% n.s. | -11.6% n.s. | -0.1% n.s. |
| RIC0.5/FGO-G | **-10.9%** * | **-15.2%** * | **+0.00 +/- 0.07 m** n.s. |

The hinge prediction from 5.4 is confirmed, and the post-arc result is the
sharpest number in the study: **+0.00 +/- 0.07 m**. Four times better initial
knowledge changes the post-manoeuvre arc by nothing at all, while moving the
pre-arc 15%. `RIC0/FGO-B`, which has no hinge, is completely insensitive.

The benefit is real but modest, and only reaches significance for the
configuration that actually has a manoeuvre.

#### 5.6b A correctly specified prior is statistically neutral

`p0_scale` against the correct value, paired:

```
RIC0/FGO-G      p0=0.25   +58.0%  *      p0=2    -2.4%  n.s.
                p0=0.5    +11.3%  n.s.   p0=4    -2.9%  n.s.
RIC0.5/FGO-G    p0=0.25   +42.3%  *      p0=4    -5.0%  n.s.
                p0=0.5     +9.5%  n.s.   p0=1e6  -5.3%  n.s.
```

**Only the overconfident arm is significant.** Every setting at `p0 >= 1`,
including removing the prior outright, sits within ~2 SE of correct. There is no
evidence that the prior helps or hurts accuracy when honestly specified -- which
is expected, since it claims sigma ~ 1732 m on x0 while the measurements
determine x0 to ~40 m. It is roughly 40x looser than the data and carries almost
no information either way.

**Caution on reading the `p0 < 1` arm.** It samples the error at 1.0 m/s and
then tells the estimator to trust it to 0.25 m/s. Its degradation is arithmetic,
not a discovery, and it says nothing about whether the prior is useful. What it
*does* give is a sensitivity figure: **a four-times-optimistic prior costs
+42-58%.** Real covariances from TLEs or a prior OD are often optimistic, so the
practical guidance is to err conservative.

#### 5.6c What the prior is actually for

Convergence, not accuracy:

```
config            p0     conv   iters
RIC0/FGO-G      0.25    10/10       8
RIC0/FGO-G         1    10/10      10
RIC0/FGO-G         4    10/10      18
RIC0/FGO-G      1e+06     3/10     254
RIC0.5/FGO-G    0.25    10/10      56
RIC0.5/FGO-G       1     9/10      78
RIC0.5/FGO-G       4     9/10      64
RIC0.5/FGO-G    1e+06     9/10      69
```

Remove the prior from `RIC0/FGO-G` and convergence collapses to 3/10 at 254
iterations. The likely cause is structural: with `dv -> 0` the `t*` gradient
vanishes, so `t*` becomes unidentifiable and the solver wanders. `RIC0.5/FGO-G`,
where a real burn gives `t*` something to lock onto, still converges 9/10 with
no prior at all.

**So prior strength is a convergence-versus-accuracy dial.** Tighter converges
faster and more reliably at the cost of bias; looser is more accurate where it
converges but slower and less reliable. On `RIC0.5/FGO-G`, `p0 = 0.25` is the
only setting in the whole study reaching **10/10 convergence, in 56 iterations**
-- against 9/10 in 78 for the correct prior. That is directly relevant to the
50-iteration budget, and it is a genuine trade rather than a free win: it buys
convergence with +42% error.

### 5.7 The pulse width does NOT floor t* precision (Option A demoted)

`|t*_err|` improves as roughly sqrt(noise) rather than linearly, and the
residual at 1 arcsec (25.0 s) is close to epsilon = 30 s, which suggested the
pulse width might be the floor. Swept, 10 seeds:

```
2 arcsec                                        1 arcsec
  eps  conv  pos_rms  |t*err|  |dv|err  iters     conv  pos_rms  |t*err|  |dv|err  iters
   12 10/10     63.7    108.4   0.0833     10     9/10     50.7    111.3   0.1047     38
   20  8/10     57.0     85.8   0.0413     84     9/10     34.0     34.0   0.0376     87
   30  9/10     55.5     52.7   0.0172     78    10/10     31.8     25.0   0.0107     27
   60  9/10     54.7     51.3   0.0175     50    10/10     31.8     25.7   0.0105     20
  100 10/10     52.7     49.4   0.0173     38    10/10     31.7     25.7   0.0105     10
```

**In the quadrature-valid range (epsilon >= 30) t\* precision is flat.**
Widening the pulse 3.3x moves `|t*err|` 52.7 -> 49.4 s at 2 arcsec and
25.0 -> 25.7 s at 1 arcsec: nothing.

The degradation below 30 is **Simpson's rule failing, not a t\* effect**. The
sampling ratio `h/sigma = 30/epsilon` runs 0.3, 0.5, 1.0, 1.5, 2.5 as epsilon
drops, and `|dv|err` and `|t*err|` degrade *together* in exactly that pattern,
both plateauing once the quadrature is adequate. That is section 6's analysis
confirmed, not the epsilon hypothesis supported.

**The stronger argument:** the model mismatch *scales with* epsilon (4.4). At
epsilon = 100 the mismatch is far larger than at 30, yet t* precision is
identical. So t* precision is limited by neither the pulse width nor the model
mismatch.

**Consequence: Option A (section 6) is demoted.** It remains a correctness fix
-- it removes the mis-specification, deletes `FD_TSTAR_STEP`, and gives exact
Jacobians -- but it is **not** a route to better t*. Caveat: this sweep cannot
go below epsilon = 12 before RK4 breaks down, so epsilon << 1 is untested and
untestable without building Option A. Flatness over [30, 100] plus the
mismatch-scaling argument makes it unlikely to help, but that is not proof.

**Free convergence win:** epsilon = 100 gives identical accuracy with **2.1x
fewer iterations at 2 arcsec (78 -> 38) and 2.7x at 1 arcsec (27 -> 10)**, and
better convergence (10/10 vs 9/10). A wider pulse gives a smoother t* landscape.
Not yet confirmed in combination with 5.8.

### 5.8 The manoeuvre epoch is the dominant lever

The baseline places the burn 3.6 h into a 27.6 h arc -- only 13% of the data
before it. Moving the burn later, holding total arc length at 1.15 days so N and
the dt grid are unchanged (`MJD_end` + `pm_duration` = 1.15):

```
2 arcsec
  label    pre%   conv  pos_rms    pre   post  |t*err|  t* med  |dv|err   dv %  iters
  base      13%   9/10     55.5  133.1   26.0     52.7    50.2   0.0172  1.99%     78
  pre030    26%  10/10     29.7   41.7   23.2     33.9    29.2   0.0075  0.86%     48
  pre045    39%  10/10     26.0   28.8   22.8     32.9    31.0   0.0051  0.59%     33
  pre0575   50%  10/10     24.8   24.1   24.3     24.4    25.3   0.0051  0.59%     28

1 arcsec
  label    pre%   conv  pos_rms    pre   post  |t*err|  t* med  |dv|err   dv %  iters
  base      13%  10/10     31.8   72.4   17.8     25.0    23.5   0.0107  1.24%     27
  pre030    26%  10/10     19.1   25.4   15.9     17.7    13.1   0.0056  0.64%     26
  pre045    39%  10/10     15.0   15.6   14.0     18.6    16.3   0.0038  0.44%     22
  pre0575   50%  10/10     13.6   13.4   13.4     18.9    21.2   0.0039  0.45%     33

  |dv| true = 0.866025 m/s, so 'dv %' is the delta-v error as a percentage of it.
  |t*err| is the mean absolute error; 't* med' its median.
```

Paired per-seed against base, **all significant**: -47% to -58% in position RMS.
`|t*err|` improves 25-55% in the mean but does **not** reach significance at
n = 9-10 (e.g. 2 arcsec symmetric: -29.03 +/- 16.52 s).

The mechanism is 5.4's hinge, confirmed quantitatively -- the gain is almost
entirely the pre-arc:

| | pre-arc | post-arc |
|---|---|---|
| 2" base -> symmetric | 133.1 -> 24.1 (**-82%**) | 26.0 -> 24.3 (-7%) |
| 1" base -> symmetric | 72.4 -> 13.4 (**-81%**) | 17.8 -> 13.4 (-25%) |

Note the asymmetry, which rules out a simple arc-length explanation: the
post-arc *shortens* from 1440 to 828 steps yet its error stays flat, while the
pre-arc lengthens from 216 to 827 steps and its error falls 82%. Once the hinge
decouples the two halves, each must determine itself -- and 216 steps of
angles-only data is below the threshold where a GEO arc is well determined,
while ~828 steps is comfortably above it. At 50/50 both sides land at ~24 m.

**The manoeuvre parameters improve alongside the trajectory.** `|dv|err` falls
3.4x at 2 arcsec (0.0172 -> 0.0051 m/s, i.e. 1.99% -> 0.59% of the true
0.866 m/s) and 2.7x at 1 arcsec (0.0107 -> 0.0039, 1.24% -> 0.45%). `|t*err|`
falls 52.7 -> 24.4 s at 2 arcsec and 25.0 -> 18.9 s at 1 arcsec, though as noted
that one is not statistically resolved at n = 9-10. Convergence goes 9/10 in
78 iterations to **10/10 in 28** -- inside the 50-iteration budget.

Worth noting for the follow-ups: at 1 arcsec `|t*err|` is essentially flat past
26% pre-arc (17.7 / 18.6 / 18.9 s) while `|dv|err` keeps falling
(0.0056 / 0.0038 / 0.0039). If that survives more seeds it would mean `t*`
saturates at a floor the extra pre-arc cannot lower, whereas `dv` keeps
benefiting -- which bears directly on whether `t*` precision is
information-limited.

**Implication for the manuscript.** The baseline epoch is a legitimate hard
case, but if it is the only case reported the method is understated by ~2x. A
burn-position sweep is a stronger result than either extreme alone.

**Untested:** pre-arc fractions beyond 50%. If the mechanism is "both sides need
enough data", performance should worsen again symmetrically as the burn moves
late. If it keeps improving, the explanation is incomplete.

### 5.9 Predictions scorecard

| prediction | outcome |
|---|---|
| Phase 1 monotonic, no divergences | **Confirmed.** |
| FGO-G on RIC0 only slightly worse than FGO-B | **Partly.** +53-74% — same order, but more than "slight". |
| FGO-B on RIC0.5 will not diverge | **Confirmed**, and strongly: std/mean = 0.1%. |
| FGO-B on RIC0.5 stays in the hundreds of metres | **Wrong.** 1.5-1.7 km. |
| Q fix hurts FGO-B on RIC0.5 disproportionately | **Supported**, not proven — trend only (5.3). |
| Strengthening the prior improves the pre-arc, not the post-arc | **Confirmed** (5.6a). Post-arc moves +0.00 +/- 0.07 m under a 4x change. |
| Pulse width epsilon floors t* precision | **Wrong** (5.7). Flat over the quadrature-valid range; the sub-30 degradation is Simpson's rule failing. |
| Moving the burn later helps the pre-arc | **Confirmed** (5.8), -47% to -58% overall, all paired-significant. |
| `p0_scale = 1` is optimal for accuracy | **Wrong** (5.6b). Everything at `p0 >= 1` is statistically indistinguishable; only over-confidence is significant. An earlier reading of a "monotonic improvement as the prior weakens" was noise -- the paired test shows n.s. |

---

## 6. Option A — analytic impulse integration (DEMOTED, see 5.7)

**Status: demoted 2026-08-26.** The epsilon sweep (5.7) shows t* precision is
flat across the quadrature-valid range, so Option A will not improve t*. It is
worth doing as a correctness fix -- it removes the model mis-specification,
deletes `FD_TSTAR_STEP`, and makes the manoeuvre Jacobians exact -- but it is no
longer the headline plan.

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

## 7. Hypotheses tested and rejected

Recorded so they are not retried.

| hypothesis | result |
|---|---|
| The prior causes the 2" failures | No. Failures occur with and without it; the prior cuts mean error 3.1x and the worst case ~5x. |
| Failure correlates with initial-guess quality | No. corr = -0.110 (dv), -0.161 (t*), -0.052 (pos RMS at 10"). |
| Sub-stepping fixes the manoeuvre-step error | No. 0.54 m of 36.67 m. |
| It's a local minimum / second basin | No. All seeds reach the same basin given budget. |
| lambda grows and freezes t* | No. lambda never grows at all. |
| Any LM damping helps | **No — settled 2026-08-26.** Swept by relative strength: every tau >= 1e-12 returns the initial guess. See 4.2 for the mechanism. Superseded all the individual damping rows below. |
| `lambda*diag(M)` damping helps | Rejected twice, but BOTH screens were invalid: the first ran under the broken stall counter pre-Q-fix; the second compared forms at equal lambda0, which post-Q is a 3.6e9x difference in relative damping. The valid test is the tau sweep in 4.2. |
| Nielsen gain-ratio lambda rule | Worse than shipped at matched damping form (2" s3: 828.3 vs 69.2 m). Note a line search almost always accepts, so the rejection branch never fires and no gain-ratio rule can grow lambda while a line search sits in front of it. |
| Removing damping entirely (pure GN) | **Helps.** 2" @300 goes 1/10 -> 0/10 divergences, mean 55.5 -> 53.5 m, iters 78 -> 40. Neutral at 10 arcsec. See 4.2. |
| Normal equations lose all precision | No. `||(M+lam*I)dx - L'y|| / ||L'y|| = 9.2e-13`; spsolve is accurate. But LSMR needs >20,000 iterations without converging, implying cond(L) ~ 5e4, cond(M) ~ 3e9 — badly conditioned, not catastrophically. |

---

## 8. Open questions

1. **Can the solver be made to converge within 50 iterations?** Largely yes,
   by scenario and epsilon rather than by the solver: the symmetric epoch gives
   10/10 in 28 iterations at 2 arcsec (5.8), and epsilon = 100 gives 10/10 in
   38 (5.7). Neither has been tested in combination. LM is ruled out (4.2).
2. ~~Why does seed 8 diverge post-Q even at 300 iterations?~~ **Answered:** it
   was the damping. Under pure GN seed 8 converges to 35.1 m at 2 arcsec.
   Still diverges at 10 arcsec (1595.5 m).
3. ~~What happens at 1 arcsec?~~ **Answered (5.2):** FGO-G on RIC0.5 gives
   31.8 m with 10/10 converged. The OD floor at 1 arcsec is 18.1 m.
4. ~~Does Option A change the floor?~~ **Largely answered (5.7):** t* precision
   is flat across epsilon in the quadrature-valid range, so Option A is a
   correctness fix rather than an accuracy route. Untestable below epsilon = 12
   without building it.
5. **Should the Q magnitude be revisited?** The units are now right; whether 5x
   RMS is the right margin is a separate, deliberate question.
6. ~~Is the prior the binding constraint on the pre-manoeuvre arc?~~
   **Answered (5.6):** no, not for accuracy -- a correctly specified prior is
   statistically neutral. It is a convergence dial, and over-confidence is the
   only significant risk.
7. **Alternatives raised with supervisor** (not investigated): triangulating
   ground-station angles into a position region as the measurement, or
   spacecraft-to-spacecraft angles.

---

## 9. Tooling and reproducing these results

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
