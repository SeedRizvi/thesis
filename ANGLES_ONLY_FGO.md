# Angles-Only FGO — Results and Diagnostics

Status as of 2026-08-27.

**See also `FGO_VS_BLS.md`** for why FGO and BLS produce identical results on
the shipped configuration, and what separates them (arc length and dynamics
model error).

**Structure.** Part I is the results, organised by the parameter each study
varies — this is the material for the manuscript. Part II is what remains open.
Part III is the diagnostic and debugging history: how the solver problems were
found and fixed, kept for reference but out of the way.

**Every number in this document can be reproduced from
`analysis/angles_only_2026-08-25/`**, which holds the scripts that produced them
and the raw per-run results. See section 20.

---
---

# PART I — RESULTS

---

## 1. Summary

**Best configuration measured:** manoeuvre at 40% of the arc, `epsilon = 100`,
pure Gauss-Newton, angles-only, with prior.

Over a **75-seed Monte Carlo (900 runs), every cell converged 75/75** — a
failure rate <= 4% at 95% confidence — with a worst case of 18-20 iterations at
1-2 arcsec.

| noise | OD floor (no burn) | with manoeuvre estimation | ignoring the burn |
|---|---|---|---|
| 10" | 84.8 m | 110.7 m | 7983 m (**72x**) |
| 2" | 29.6 m | **24.5 m** | 6501 m (**266x**) |
| 1" | 18.3 m | **14.5 m** | 4841 m (**333x**) |

Mean position RMS. The 1 arcsec result comfortably passes the historical ~80 m
target.

**Four findings that shape the manuscript:**

1. **The manoeuvre epoch is the dominant lever** (section 4). Moving the burn
   from 13% to 40-50% of the arc halves the position error and takes convergence
   from 9/10 in 78 iterations to 10/10 in 28. Roughly half of what looked like
   method cost was scenario geometry.
2. **Below ~4 arcsec the problem is model-limited, not measurement-limited**
   (section 9). Residual dynamics model error exceeds measurement noise there,
   and any spare degrees of freedom absorb it. This is a statement about the
   dynamics model, not the estimator.
3. **`sigma_t*` scales as measurement noise over |dv|** (section 6), verified to
   ~1% on an independent configuration. Position accuracy, by contrast, is
   completely independent of burn magnitude.
4. **Range observes the dimension where model error accumulates; angles do not**
   (section 7). That is why angles-only OD becomes model-limited at a noise
   level where angles-plus-range does not.

---

## 2. Experimental setup

**Scenario.** GEO satellite, three ground stations (Rocky Point, Singapore,
Tsukuba), `dt = 60 s`, total arc 1.15 days = 99,360 s = ~1656 steps. Truth from
`orbDetHOUSE` with J2 only; the FGO propagates 2-body + J2.

**Estimators.**

- **FGO-B** — no manoeuvre parameters. 9,942 columns, 6 prior rows.
- **FGO-G** — augments the state with `(dv, t*)`, the Gaussian-impulse
  formulation. 9,946 columns, 10 prior rows.

**Configurations.** `RIC0` has `delta_v = 0` (no manoeuvre at all); `RIC0.5` has
`delta_v_ric = [0.5, 0.5, 0.5]`, i.e. |dv| = 0.866025 m/s. Epoch variants move
the burn while holding total arc length at 1.15 days, so N and the dt grid are
unchanged (`MJD_end` + `pm_duration` = 1.15):

| tag | MJD_end | pm_duration | pre-arc | pre% |
|---|---|---|---|---|
| `base` | 59349.15 | 1.000 | 12,960 s (216 steps) | 13% |
| `pre030` | 59349.30 | 0.850 | 25,920 s (432 steps) | 26% |
| `pre040` | 59349.46 | 0.690 | 39,720 s (662 steps) | **40%** |
| `pre045` | 59349.45 | 0.700 | 38,820 s (647 steps) | 39% |
| `pre0575` | 59349.575 | 0.575 | 49,620 s (827 steps) | 50% |

`t*` lands exactly on the 60 s grid in every case. `pre040` and `pre045` are
near-duplicates built independently and agree to 0.1 m, which is a free
consistency check.

**Initial state error and prior.** Drawn per ECI axis as
`N(0, 1000 m)` in position and `N(0, 1 m/s)` in velocity, so 3D norms of 1732 m
and 1.732 m/s RMS. **The same values populate P0**, which is the correct,
self-consistent choice: the prior covariance equals the distribution the
perturbation was drawn from. `dv` guess error 0.1 m/s, `t*` guess error 120 s.

**Pairing.** `build_seed` reproduces `mc_fgo.run_fgo_seed`'s RNG draw order
exactly and skips the dv/t* draws in FGO-B just as `mc_fgo` does, so **FGO-B and
FGO-G for a given seed receive identical measurements and identical x0**. All
"paired" comparisons below are per-seed differences on that basis; `*` marks
a mean difference exceeding 2 standard errors.

**Convergence** means the solver terminated on its own criteria, not on
`max_iters`.

---

## 3. Measurement noise sensitivity

### 3.1 The angles-only OD floor

`RIC0` / FGO-B — no manoeuvre anywhere, so this is the pure OD core:

| noise | converged | mean | median | max | iters |
|---|---|---|---|---|---|
| 10" | **10/10** | 91.1 m | 89.6 | 137.0 | **9** |
| 2" | **10/10** | 29.8 m | 29.6 | 34.6 | **6** |
| 1" | **10/10** | 18.1 m | 18.3 | 20.1 | **5** |

No divergences at any noise level, monotonic in noise, **5-9 iterations**. The
OD core is clean; every remaining difficulty belongs to manoeuvre estimation.

### 3.2 Noise sweep, FGO-B versus FGO-G

Paired per-seed, `RIC0_pre040` (no burn) and `pre040` (0.866 m/s burn).
Negative = FGO-G **better** than the no-manoeuvre floor. n = 75 at 1/2/10
arcsec, n = 15 elsewhere:

| noise | floor (FGO-B) | false alarm (FGO-G, no burn) | real burn (FGO-G) | false alarm vs floor | real vs floor |
|---|---|---|---|---|---|
| 1" | 17.5 | 14.0 | 14.3 | **-19.9% +/- 2.7** * | **-18.5% +/- 2.9** * |
| 2" | 28.8 | 23.1 | 24.4 | **-19.7% +/- 3.9** * | **-15.2% +/- 3.8** * |
| 3" | 36.7 | 33.3 | 35.3 | -9.3% +/- 4.6 * | -3.8% +/- 4.5 |
| **4"** | 43.5 | 43.9 | 46.4 | **+0.9% +/- 5.2** | +6.6% +/- 5.1 |
| 5" | 50.1 | 54.7 | 57.4 | +9.2% +/- 5.7 | +14.6% +/- 5.6 * |
| 6" | 56.8 | 65.6 | 68.5 | +15.6% +/- 6.1 * | +20.6% +/- 6.0 * |
| 7" | 63.5 | 76.6 | 79.5 | +20.6% +/- 6.4 * | +25.1% +/- 6.3 * |
| 10" | 84.6 | 110.0 | 112.5 | +30.0% +/- 7.1 * | +32.9% +/- 7.0 * |

Monotonic, with a sign change at **4 arcsec**. Above it, manoeuvre estimation
costs accuracy as expected. Below it, FGO-G *beats* the no-manoeuvre floor —
explained in section 9.

---

## 4. Manoeuvre epoch: the pre/post arc ratio

**This is the dominant lever in the study.** The baseline placed the burn 3.6 h
into a 27.6 h arc — only 13% of the data before it.

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

Paired per-seed against base, **all significant**: **-47% to -58%** in position
RMS. `|t*err|` improves 25-55% in the mean but does not reach significance at
n = 9-10 (2 arcsec symmetric: -29.03 +/- 16.52 s).

**The gain is almost entirely the pre-manoeuvre arc:**

| | pre-arc | post-arc |
|---|---|---|
| 2" base -> symmetric | 133.1 -> 24.1 (**-82%**) | 26.0 -> 24.3 (-7%) |
| 1" base -> symmetric | 72.4 -> 13.4 (**-81%**) | 17.8 -> 13.4 (-25%) |

**Mechanism.** The manoeuvre parameters insert a **hinge** at `t*` that
partially decouples the two halves of the trajectory. Each half must then
determine itself. Note the asymmetry, which rules out a simple arc-length
explanation: the post-arc *shortens* from 1440 to 828 steps yet its error stays
flat, while the pre-arc lengthens from 216 to 827 steps and its error falls 82%.
So it is a **threshold, not a gradient** — above roughly 800 steps an arc is
well determined and more data adds little; at 216 steps it falls off a cliff.

The physical reason is **geometric diversity, not measurement volume**. A GEO
satellite is nearly stationary relative to a ground station; the observability
of *range* from angles alone comes almost entirely from Earth's rotation
changing the parallax. In diurnal terms:

```
216 steps = 3.6 h  = 15% of a day
827 steps = 13.8 h = 57% of a day
```

A 3.6-hour arc gives three stations plenty of measurements but almost no change
in viewing geometry, and you cannot triangulate range without parallax.

**The manoeuvre parameters improve alongside the trajectory.** `|dv|err` falls
3.4x at 2 arcsec (1.99% -> 0.59% of the true burn) and 2.7x at 1 arcsec
(1.24% -> 0.45%). Convergence goes 9/10 in 78 iterations to **10/10 in 28**.

**Implication for the manuscript.** The baseline epoch is a legitimate hard
case, but reporting it alone understates the method by ~2x. A burn-position
sweep is a stronger result than either extreme.

**Untested:** pre-arc fractions beyond 50%. If "both sides need enough duration"
is right, performance should worsen symmetrically as the burn moves late.

**Confound to note:** moving the burn changes both the pre/post balance *and*
the ground-station geometry at the burn epoch. This design cannot separate them.
A control that would: hold the burn at 3.6 h but extend the arc backwards.

---

## 5. Gaussian pulse width (epsilon)

Swept at 10 seeds, `base` epoch:

```
2 arcsec                                        1 arcsec
  eps  conv  pos_rms  |t*err|  |dv|err  iters     conv  pos_rms  |t*err|  |dv|err  iters
   12 10/10     63.7    108.4   0.0833     10     9/10     50.7    111.3   0.1047     38
   20  8/10     57.0     85.8   0.0413     84     9/10     34.0     34.0   0.0376     87
   30  9/10     55.5     52.7   0.0172     78    10/10     31.8     25.0   0.0107     27
   60  9/10     54.7     51.3   0.0175     50    10/10     31.8     25.7   0.0105     20
  100 10/10     52.7     49.4   0.0173     38    10/10     31.7     25.7   0.0105     10
```

**Accuracy is flat over the quadrature-valid range (epsilon >= 30).** Widening
the pulse 3.3x moves `|t*err|` 52.7 -> 49.4 s at 2 arcsec and 25.0 -> 25.7 s at
1 arcsec: nothing. The pulse width does **not** floor `t*` precision.

**The degradation below 30 is Simpson's rule failing, not an epsilon effect.**
With `n_timesteps = 1`, RK4 samples the impulse at only three distinct times and
the applied impulse reduces exactly to Simpson's rule (section 17.2). The
sampling ratio `h/sigma = 30/epsilon` runs 0.3, 0.5, 1.0, 1.5, 2.5 as epsilon
drops, and `|dv|err` and `|t*err|` degrade *together* in that pattern, both
plateauing once the quadrature is adequate.

**Stronger argument that epsilon is not the limit:** the model mismatch *scales
with* epsilon (section 17). At epsilon = 100 the mismatch is far larger than at
30, yet `t*` precision is identical.

**Free convergence win: epsilon = 100 gives identical accuracy with 2.1x fewer
iterations at 2 arcsec (78 -> 38) and 2.7x at 1 arcsec (27 -> 10)**, and better
convergence (10/10 vs 9/10). A wider pulse gives a smoother `t*` landscape.

**Caveat:** the sweep cannot go below epsilon = 12 before RK4 breaks down, so
epsilon << 1 is untested and untestable without building section 12.

---

## 6. Manoeuvre magnitude and direction

Predicted from identifiability (section 9.3): `t*` sensitivity is proportional
to `dv`, so `sigma_t* ~ sigma_meas / |dv|`, floored by the 120 s `t*` prior.
All four configurations at the 40% epoch, 10 seeds. `pred` is calibrated **only**
on `RIC1` and then extrapolated — `pre040` is an independent check.

| config | \|dv\| | direction | noise | \|t*err\| | **pred (1/\|dv\| + prior)** | \|dv\|err | as % of \|dv\| | pos_rms |
|---|---|---|---|---|---|---|---|---|
| I02 | 0.200 | in-track | 2" | 71.8 | 82.5 | 0.0048 | 2.40% | 25.0 |
| C02 | 0.200 | cross-track | 2" | 70.1 | 82.5 | 0.0040 | 1.99% | 25.5 |
| pre040 | 0.866 | R=I=C | 2" | **35.5** | **35.0** | 0.0044 | 0.51% | 25.9 |
| RIC1 | 1.732 | R=I=C | 2" | 18.4 | 18.4 (cal) | 0.0044 | 0.25% | 26.1 |
| I02 | 0.200 | in-track | 1" | 69.3 | 63.4 | 0.0038 | 1.92% | 15.0 |
| C02 | 0.200 | cross-track | 1" | **45.9** | 63.4 | 0.0028 | 1.39% | 15.2 |
| pre040 | 0.866 | R=I=C | 1" | **19.3** | **19.1** | 0.0030 | 0.35% | 14.8 |
| RIC1 | 1.732 | R=I=C | 1" | 9.7 | 9.7 (cal) | 0.0030 | 0.18% | 14.9 |

**The law holds to ~1% on the independent point.** Calibrated on `RIC1` alone,
it predicts `pre040` at **35.0 s (measured 35.5)** at 2 arcsec and **19.1 s
(measured 19.3)** at 1 arcsec. Doubling |dv| halves the `t*` error, exactly.
At |dv| = 0.2 the 120 s prior takes over and caps the degradation.

**Position accuracy is independent of |dv|.** 25.0 / 25.5 / 25.9 / 26.1 m at
2 arcsec and 15.0 / 15.2 / 14.9 / 14.8 m at 1 arcsec, across an **8.7x range of
burn magnitude**. A small manoeuvre is not harder to *orbit-determine*, only
harder to *time*.

**The absolute delta-v error is constant, so only the relative error scales.**
`|dv|err` sits at 0.0028-0.0048 m/s across the whole range while the fraction of
the burn runs 2.40% down to 0.18%. The reason is section 9: this floor is not
measurement noise on the burn, it is absorbed dynamics model error, which does
not care how large the real manoeuvre is.

**Burn direction matters, but only where the prior does not dominate.** At
1 arcsec, cross-track (45.9 s) is 34% better timed than in-track (69.3 s) at
identical |dv| = 0.2. At 2 arcsec both sit at 70-72 s, near the prior-dominated
ceiling, so direction cannot express itself. **Measured but not explained** — a
prior guess that in-track would be *more* observable (period change compounding
over the arc) is contradicted.

---

## 7. Angles-only versus angles + range

`RIC0_pre040` and `pre040`, FGO-B vs FGO-G, range at 10 m, current Q.
n = 75 angles-only, 15 with range.

| measurements | scenario | noise | FGO-B | FGO-G | paired difference | FGO-G \|dv\|err | FGO-G \|t*err\| |
|---|---|---|---|---|---|---|---|
| angles only | no burn | 2" | 29.6 | 23.6 | **-20.4% +/- 1.6** * | 0.00324 | 104.3 |
| angles only | no burn | 1" | 18.3 | 14.5 | **-21.0% +/- 1.1** * | 0.00235 | 104.3 |
| angles + range | no burn | 2" | 12.3 | 12.1 | **-1.5% +/- 1.6 n.s.** | 0.00139 | 94.6 |
| angles + range | no burn | 1" | 10.3 | 9.6 | -7.0% +/- 1.0 * | 0.00133 | 94.3 |
| angles only | real burn | 2" | 6500.9 | 24.5 | -99.6% * | 0.00335 | 26.7 |
| angles only | real burn | 1" | 4840.7 | 14.5 | -99.7% * | 0.00248 | 14.3 |
| angles + range | real burn | 2" | 2661.2 | 11.5 | -99.6% * | 0.00142 | 6.0 |
| angles + range | real burn | 1" | 2439.4 | 9.1 | -99.6% * | 0.00137 | 6.7 |

FGO-B has no manoeuvre parameters, so the `|dv|err` and `|t*err|` columns apply
to FGO-G only. In the no-burn rows the true delta-v is exactly zero, so
`|dv|err` **is** the spurious delta-v.

**Range roughly halves the position error** (29.6 -> 12.3 m at 2 arcsec on the
floor) and **improves `t*` by 2-4.5x** on a real burn (26.7 -> 6.0 s at
2 arcsec). Note `t*` lands at ~6 s with range at *both* noise levels — no longer
angular-noise-limited, so something else floors it there.

**Statement for the manuscript.** Range measurements directly observe the
dimension in which dynamics model error accumulates, at every epoch, so that
error never builds up. Angles constrain direction but barely constrain range, so
with range removed that dimension is held **only by the dynamics model** — the
very thing that is slightly wrong. This is why angles-only OD becomes
model-limited below ~4 arcsec while angles-plus-range does not, and it is why
the effect was never seen in the earlier angles+range work. The Q units bug
(section 14.1) compounded it: pre-fix, a 1.044e-4 m/step mismatch against an
effective sigma of 0.0229 m was 0.005 sigma — invisible — where post-fix it is
0.26 sigma per step.

---

## 8. Robustness Monte Carlo (75 seeds, 900 runs)

**Configuration chosen by a 2x2 factorial** at the 40/60 epoch ({shipped
damping, pure GN} x {epsilon 30, 100}, 10 seeds, 2 arcsec). Accuracy was
**identical in all four cells** (25.9 m, `|dv|err` 0.0044, `|t*err|` 35.3-35.5)
— neither change touches the answer — but iterations differed 6x and **the two
effects multiply**:

```
  eps  damping    conv  pos_rms  it mean  it max
  100  off (GN)  10/10     25.9        8      20
  100  on        10/10     25.9       17      33
   30  off (GN)  10/10     25.9       36     167
   30  on        10/10     25.9       48     165   <- previously shipped config
```

The `it max` column is the operative one: at epsilon = 30 the mean is 36-48 but
some seed still needs 165-167 iterations, so that configuration busts a
50-iteration cap even at the improved epoch.

**MC design.** `pre040` and `RIC0_pre040` (same split, no burn, so the
false-alarm control places its non-existent manoeuvre at the same epoch)
x {FGO-B, FGO-G} x {10", 2", 1"} x 75 seeds = 900 runs, 19.4 min on 10 cores.

**Every cell converged 75/75.** By the rule of three that supports a failure
rate **<= 4% at 95% confidence**, against the <= 30% a 10-seed result can
justify.

### 8.1 Iterations

| cell | median | p95 | max | seeds > 50 |
|---|---|---|---|---|
| pre040 / FGO-G, 1" | 6 | 14 | **18** | 0 |
| pre040 / FGO-G, 2" | 6 | 15 | **20** | 0 |
| pre040 / FGO-G, 10" | 7 | 19 | 74 | **2** |
| RIC0_pre040 / FGO-B, all | 4 | 4-8 | 13 | 0 |

At 1 and 2 arcsec the worst case over 75 seeds is 18-20 iterations. The only
tail is 10 arcsec (seeds 72 and 29 at 74 and 59), and both still converge to
good answers (94.0 m, 85.7 m), so a hard cap would truncate rather than break
them. For scale, this began at 141 iterations worst-case at the baseline epoch
and 165-167 at epsilon = 30.

---

## 9. The model-limited regime below 4 arcsec

Section 3.2 shows FGO-G *beating* the no-manoeuvre floor by 17-21% below
4 arcsec, with the same effect present when **no manoeuvre exists at all**. This
section explains it, and the explanation matters more than the effect.

### 9.1 The spurious delta-v decomposes into bias and variance

The estimated `dv` was recorded as a RIC **vector**, not just a norm. If the
manoeuvre parameters absorb a deterministic model error the direction is the
same on every seed (`|mean(v)| ~ mean(|v|)`); if they fit measurement noise the
direction is random (`|mean(v)| << mean(|v|)`). On `RIC0_pre040`/FGO-G, where
the true `dv` is exactly zero:

| noise | mean\|v\| | \|mean v\| | ratio | mean vector (R, I, C) m/s |
|---|---|---|---|---|
| 1" | 0.00267 | **0.00227** | 0.85 | [-0.00185, +0.00115, +0.00065] |
| 2" | 0.00378 | **0.00246** | 0.65 | [-0.00208, +0.00125, +0.00043] |
| 5" | 0.00798 | **0.00309** | 0.39 | [-0.00271, +0.00146, -0.00022] |
| 10" | 0.01569 | **0.00471** | 0.30 | [-0.00419, +0.00168, -0.00132] |

`mean|v|` grows ~6x across the 10x noise range — it tracks the noise.
`|mean v|` grows only ~2x and its direction is stable (radial-negative,
in-track-positive). That is a **noise-independent systematic component beneath a
noise-proportional random one**. At 1 arcsec the ratio is 0.85, so the spurious
`dv` is overwhelmingly systematic; at 10 arcsec it is 0.30, mostly
noise-fitting. **The crossover at 4 arcsec is exactly where the two components
cross.**

### 9.2 The delta-v error is the same with or without a real burn

Matched measurements and noise, from the section 7 table:

```
angles only     2"    no burn 0.00324    real burn 0.00335    (+3%)
angles only     1"    no burn 0.00235    real burn 0.00248    (+6%)
angles + range  2"    no burn 0.00139    real burn 0.00142    (+2%)
angles + range  1"    no burn 0.00133    real burn 0.00137    (+3%)
```

A 0.866 m/s burn adds only 2-6% to an error that is otherwise entirely present
with no burn at all. **The delta-v error is dominated by absorbed model error,
not by measurement noise on the manoeuvre itself.** That single observation
explains three separate findings: why section 6 finds the absolute error
constant across burn magnitude, why range halves it in every row, and why FGO-G
beats FGO-B on manoeuvre-free data below 4 arcsec.

### 9.3 t* is unidentifiable when dv -> 0

`t*` sensitivity is proportional to `dv`:
`d(a_man)/dt* = dv * dg/dt*`. With `dv -> 0` the gradient vanishes and the
measurements carry **zero** information about `t*`. Estimated `t*` on
`RIC0_pre040`/FGO-G (true epoch 39,720 s):

```
 1"   mean 39708.9 s   std 106.5 s
 5"   mean 39708.9 s   std 106.4 s
10"   mean 39708.9 s   std 106.4 s
```

**Identical to 0.1 s across a 10x change in measurement noise**, with a scatter
matching the 120 s prior. `t*` sits exactly where its prior draw put it; the
measurement data is doing nothing at all. In the section 7 table the same shows
as `|t*err|` of 94-104 s in every no-burn row regardless of measurement type or
noise, against 6.0-26.7 s in the real-burn rows.

### 9.4 What this means

**This is NOT evidence that FGO-G is more accurate than FGO-B, and it must not
be presented that way.** It is an over-parameterised estimator compensating for
a deficiency in the dynamics model. The honest statement is: *below ~4 arcsec
the residual dynamics model error exceeds the measurement noise, and any spare
degrees of freedom will absorb it.* That is a claim about the **dynamics model**
(2-body + J2 against the truth propagator), not the estimator.

Two consequences:

- **Delta-v accuracy at 1-2 arcsec is limited by the dynamics model, not the
  measurements.** Improving the propagator would improve delta-v recovery;
  sharpening the angles further would not.
- **Falsifiable prediction:** a dynamics model closer to truth shrinks the
  section 3.2 effect toward zero. See open question 3.

---

## 10. What manoeuvre estimation costs and buys

At the **baseline (13%) epoch**, mean position RMS over converged seeds:

| noise | OD floor `RIC0/B` | false alarm `RIC0/G` | real manoeuvre `RIC0.5/G` | not estimating `RIC0.5/B` |
|---|---|---|---|---|
| 10" | 91.1 | 158.3 (+74%) | 160.0 (+76%) | 1705.1 (**11x** worse) |
| 2" | 29.8 | 45.6 (+53%) | 55.5 (+86%) | 1553.0 (**28x** worse) |
| 1" | 18.1 | 28.1 (+55%) | 31.8 (+76%) | 1492.7 (**47x** worse) |

At the **40% epoch** (section 8, 75 seeds):

| noise | OD floor | false alarm | real manoeuvre | not estimating |
|---|---|---|---|---|
| 10" | 84.8 | 107.5 | 110.7 | 7983.0 (**72x**) |
| 2" | 29.6 | 23.6 | **24.5** | 6500.9 (**266x**) |
| 1" | 18.3 | 14.5 | **14.5** | 4840.7 (**333x**) |

**The penalty for ignoring a burn grows as the burn moves toward the middle of
the arc** — 11-47x at 13%, 72-333x at 40%. At 13% pre-arc the fit is dominated
by the post arc and can partly absorb the discontinuity; at 40% it is pulled
between two substantial arcs and both end up wrong.

### 10.1 An unmodelled manoeuvre is a bias, not a divergence

`RIC0.5` / FGO-B converges 10/10 in 5-9 iterations at every noise level, and at
2 arcsec the spread across seeds is `std/mean = 0.1%` (min 1550.1, max 1556.1).
That is a **systematic bias**, near-identical on every seed — not instability.

The confirming signature is noise insensitivity:

```
RIC0.5/FGO-B   10" -> 1":  1705 -> 1493 m   (-12%)
RIC0/FGO-B     10" -> 1":    91 ->   18 m   (-80%)
```

Ten times better measurements buy 12%, because the error is model bias rather
than measurement variance.

---
---

# PART II — OPEN QUESTIONS AND DEFERRED WORK

---

## 11. Open questions

1. **Should the Q magnitude be revisited?** The units are now right
   (section 14.1); whether 5x the measured RMS mismatch is the right margin was
   never deliberately chosen.
2. **Why is cross-track better timed than in-track at equal |dv|?** Measured in
   section 6 (45.9 s vs 69.3 s at 1 arcsec), not explained.
3. **Is 2-body+J2 adequate below 4 arcsec?** Raised by section 9. Residual model
   error exceeds measurement noise there. Prediction: a dynamics model closer to
   truth shrinks the effect toward zero. Bears directly on whether the 1 arcsec
   results are model-limited.
4. **Does performance worsen symmetrically past a 50% pre-arc?** Section 4's
   threshold explanation predicts it should. Untested.
5. **Volume versus geometric diversity.** Halving `dt` to 30 s doubles
   measurement count at fixed arc duration. If volume drives section 4's effect
   that should help as much as doubling the arc; if parallax drives it, barely
   at all.
6. **Alternatives raised with supervisor** (not investigated): triangulating
   ground-station angles into a position region as the measurement, or
   spacecraft-to-spacecraft angles.
7. **Profiling `t*` rather than optimising it** (diagnostic only, not yet run).
   Fix `t*` on a grid, solve for states and `dv` at each — `dv` enters the
   dynamics linearly, so `t*` is the only source of manoeuvre-related
   nonlinearity. Would give a cost-versus-`t*` curve, i.e. a confidence interval
   rather than a point estimate, and would show whether the residual `t*` errors
   are optimisation failures or genuine information limits. **Discuss before
   implementing.**

---

## 12. Option A — analytic impulse integration (DEFERRED)

**Status: demoted 2026-08-26.** The epsilon sweep (section 5) shows `t*`
precision is flat across the quadrature-valid range, so Option A will not
improve `t*`. It remains worth doing as a **correctness fix** — it removes the
model mis-specification, deletes `FD_TSTAR_STEP`, and makes the manoeuvre
Jacobians exact — but it is no longer the headline plan.

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

**Cost.** `F_man_mat` is **26.1% of `create_L`** (0.235 s of 0.899 s), i.e.
9.6-18.9% of a full iteration depending on backtracking. Real, but a constant
factor — it does not change iteration counts.

**Caveats to handle.**

- **Operator splitting** neglects impulse/gravity coupling within a step.
  Estimated ~1 mm against Q's sigma. **Measure it** against a heavily
  sub-stepped reference; do not assume. Fallback is Strang splitting, which
  costs the exact-Jacobian property.
- **`FD_TSTAR_STEP = 0.01`** disappears entirely under analytic Jacobians.
- **The causal offset is a known bias**: estimated `t*` is offset from the
  physical burn time by `delay`. Document or subtract it.

Keep `orbital_dynamics` computing the full field behind an `include_man` flag —
it is required as the reference for the splitting-error measurement, and it
keeps `a_total = a_2body + a_J2 + a_man` intact in the source and the
manuscript.

**Explicitly out of scope:** replacing the Gaussian impulse with a sparse
per-step delta-v formulation. That would delete the object of study — the
Gaussian approximation of an impulsive manoeuvre is the contribution.

---
---

# PART III — DIAGNOSTICS AND DEBUGGING HISTORY

Everything below is how the solver problems were found and fixed. Kept for
reference; not needed for the manuscript results in Part I.

---

## 13. The "catastrophic failure" was the iteration budget

An earlier revision of this document claimed 2 arcsec caused a catastrophic
instability in 4 of 10 seeds. **That was wrong.** `max_iterations: 50` was
simply too small for roughly a third of 2 arcsec seeds. The same 20 runs, scored
at iteration 50 versus at convergence (pre-Q-fix):

| budget | 10" mean | 2" mean | apparent effect of sharper measurements |
|---|---|---|---|
| 50 iters | 321.1 m | 323.8 m | **+0.8%** — no benefit, 3 blow-ups |
| converged | 234.3 m | 125.7 m | **-46%** — uniform benefit |

Same code, same seeds, same data. Only the cap differs. The three "failures" at
2 arcsec converge cleanly given budget:

| seed | @50 iters | converged | iters needed |
|---|---|---|---|
| 1 | 765.04 m, `dyn 1.6e3` | 112.53 m, `dyn 1.0e1` | 141 |
| 7 | 1031.44 m, `dyn 3.1e3` | 94.05 m, `dyn 1.3e1` | 64 |
| 8 | 511.74 m, `dyn 7.0e3` | 120.54 m, `dyn 1.3e1` | 72 |

All three land in the same basin as the seeds converging inside 50 iterations,
at the *good* end of the distribution. There is no second basin and no local
minimum. The 10 arcsec arm was affected too: seed 7 reads 1089.91 m at 50
iterations and 221.56 m converged, so a previously published 246 m median was
itself depressed by an unconverged seed.

**Caveat: `cost_dyn` alone does not indicate convergence.** `create_init_state`
builds the initial trajectory with the same propagator the dynamics residual
uses, so that trajectory is dynamically self-consistent by construction
(`cost_dyn` ~ 1e-6) while being ~24 km from truth — a 1 m/s initial velocity
error integrates to ~99 km over the 27.6 h arc. Low `cost_dyn` is only
meaningful alongside low position error.

---

## 14. Defects found and fixed

### 14.1 Q was a standard deviation consumed as a variance (HIGH impact)

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

**Effect:** large accuracy gain but a markedly stiffer problem. On the 9 seeds
converging in both versions:

| | before Q fix | after Q fix | |
|---|---|---|---|
| 10 arcsec mean | 232.96 m | 160.02 m | -31% |
| 2 arcsec mean | 126.29 m | 55.53 m | -56% |

**Note:** this changes EKF and BLS results too. Those Monte Carlos have not been
re-run (see TODO.md).

### 14.2 Termination criteria unsuited to the problem

Both tests in `opt()` were replaced:

- `la.norm(delta_x * best_scale) < 1e-3` summed metres, m/s and seconds over
  9,946 variables — dimensionally incoherent, and so tight it rarely fired.
- `stalled`: 10 *consecutive* iterations each below 1e-6 relative cost. This
  terminated any well-damped variant at exactly iteration 10, before it could
  do anything — which invalidated two entire damping screens.

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

### 14.3 Line search could discard a better step

The acceptance branch overwrote `best_scale` unconditionally, so a smaller
scale passing the ratio test could replace a larger scale that had already
achieved a lower cost. Fixed by deleting the assignment; the tracking above it
was already correct.

**Effect:** fired on 4 of 20 runs but changed only the path, never the
destination (0/20 seeds changed position error), costing 1-6 extra iterations.
Kept as a correctness fix, not an improvement.

---

## 15. Solver investigation

### 15.1 The divergences, and what narrowed them

After the Q fix, on `RIC0.5`/FGO-G with the then-shipped solver:

| budget | 10" diverged | 2" diverged |
|---|---|---|
| 50 iters | 3/10 | 3/10 |
| 300 iters | 1/10 (seed 8) | 1/10 (seed 8) |

Three things narrowed this:

- **It is confined to manoeuvre estimation.** `RIC0`/FGO-B converges 10/10 in
  5-9 iterations at every noise level, and FGO-B on `RIC0.5` likewise. Only
  FGO-G is slow or unstable.
- **Removing the vestigial damping fixes most of it** (15.2).
- **Prior strength is a convergence dial** (section 16).

All of it is superseded by Part I: at the 40% epoch with epsilon = 100 and pure
GN, 900/900 runs converge.

### 15.2 Levenberg-Marquardt is inapplicable to this problem

`opt()` was never LM: `lambda_max = 5.0e-07` across 20+ runs, lambda never rose
above its initial 1e-6, and the growth branch only fires when the line search
fails across all 20 halvings, which never happens. Post-Q the median `M`
diagonal is 3.6e9, so lambda = 1e-6 is a **relative** damping of 2.8e-16 —
below machine epsilon (2.2e-16). The shipped solver is *exactly* Gauss-Newton
with backtracking.

**That turns out to be the right answer, not a bug.** Damping swept by relative
strength `tau` so the two forms are comparable (`M + tau*diag(M)` vs
`M + tau*median(diag(M))*I`), 2 arcsec, maxit 50:

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
configuration converges back onto the starting point. Even `tau = 1e-12` — one
part in a trillion, relative — freezes the solve, implying the informative
directions have relative curvature below 1e-12, i.e. `cond(M) >~ 1e12`.

**Why LM fails.** In the eigenbasis of `M` the GN step along direction *i* is
`g_i/mu_i`; the LM step is `g_i/(mu_i + tau*d_i)`. Damping leaves steep
directions alone and suppresses shallow ones by `mu_i/(tau*d_i)`. LM's premise
is that low-curvature directions are where the quadratic model is least
trustworthy. **That premise is false here**: angles constrain direction but
barely constrain range, so range-like directions are intrinsically shallow;
`t*` sits at the bottom of the curvature spectrum (2.53e2 against velocity's
9.49e6); and 15.4 states it outright — a 13% cost change corresponds to an 85%
error change, so the directions that reduce *error* are exactly those that
barely reduce *cost*. LM damps out precisely the directions that must move.

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
line search is the correct instrument. LM applies a direction-changing remedy to
a length problem.

**Measured effect of removing the damping entirely** (10 seeds, baseline epoch):

| | shipped: div / mean* / iters | pure GN: div / mean* / iters |
|---|---|---|
| 10" @50 | 3/10 · 145.5 m · 40 | 3/10 · **131.6 m** · 40 |
| 2" @50 | 3/10 · 52.8 m · 35 | **1/10** · 55.5 m · **24** |
| 10" @300 | 1/10 · 160.0 m · 82 | 1/10 · 159.4 m · 74 |
| 2" @300 | 1/10 · 55.5 m · 78 | **0/10** · **53.5 m** · **40** |

*over converged seeds only. Seed 8 goes from 2827.9 m diverged to 35.1 m
converged. Not uniformly better: at 10 arcsec / 50 iterations GN fixes seed 10
but breaks seed 2 (198.4 m -> 4362.6 m); seed 2 converges fine under GN at 300
iterations, so it is slower, not broken.

### 15.3 The t* error budget dominates the residual

`t*` enters the cost only through dynamics residuals within +/-3*epsilon of the
manoeuvre — roughly 3-4 steps, i.e. **~20 of 19,898 residual rows (0.1%)**. But
it displaces the entire post-manoeuvre arc, **87% of the trajectory at the
baseline epoch**. One variable determined by 0.1% of the cost sets most of the
error.

```
corr(|t*_err|, pos_rms) = +0.54 .. +0.70, consistent across both arms
                          and both iteration budgets

mean |t*_err| = 50.4 s  ->  0.866 m/s x 50.4 s = 44 m predicted displacement
observed position RMS                          = 53.5 m
```

`t*` is **unbiased but imprecise**: mean signed error only +8.0 s while the
scatter is ~60 s.

**Why `t*` is hard, physically.** A 50 s timing error on a 0.866 m/s burn
displaces the orbit by 43 m. At GEO range that is **0.22 arcsec** — nine times
*below* the 2 arcsec measurement noise floor. Angles are being asked to resolve
a sub-noise signal.

### 15.4 The objective is a long flat valley

For seed 1 at 2 arcsec, between iteration 50 and convergence:

```
cost      11519  ->  ~10000     (-13%)
pos RMS     765  ->     112     (-85%)
```

At iteration 50 the whitened residual RMS is already 0.76 sigma — the
measurements are essentially satisfied while the trajectory is 765 m wrong.
This explains why Gauss-Newton crawls, why cost-based termination is a poor
proxy, and why results differ slightly between machines (BLAS/SuperLU rounding).

---

## 16. The prior study

Two knobs, deliberately separated. `sigma_scale` moves the sampled perturbation
**and** the prior together (the self-consistent scenario axis). `p0_scale` moves
the prior **only**, leaving the sampled error at sigma = 1000 m / 1.0 m/s per
axis — a deliberate mis-specification. Seeds are exactly paired: numpy scales
the same standard normals, so seed 3 at `sigma_scale` 0.5 gets precisely half
the error it gets at 1.0. 240 runs, 2 arcsec, baseline epoch.

### 16.1 Better initial knowledge helps the pre-arc only

`sigma_scale` 1.0 -> 0.25 (four times better initial knowledge):

| config | overall | pre | post |
|---|---|---|---|
| RIC0/FGO-B | -0.1% n.s. | -0.7% n.s. | +0.1% n.s. |
| RIC0/FGO-G | -6.6% n.s. | -11.6% n.s. | -0.1% n.s. |
| RIC0.5/FGO-G | **-10.9%** * | **-15.2%** * | **+0.00 +/- 0.07 m** n.s. |

The hinge prediction is confirmed, and the post-arc result is the sharpest
number in the study: **+0.00 +/- 0.07 m**. Four times better initial knowledge
changes the post-manoeuvre arc by nothing at all, while moving the pre-arc 15%.
`RIC0/FGO-B`, which has no hinge, is completely insensitive.

### 16.2 A correctly specified prior is statistically neutral

`p0_scale` against the correct value, paired:

```
RIC0/FGO-G      p0=0.25   +58.0%  *      p0=2    -2.4%  n.s.
                p0=0.5    +11.3%  n.s.   p0=4    -2.9%  n.s.
RIC0.5/FGO-G    p0=0.25   +42.3%  *      p0=4    -5.0%  n.s.
                p0=0.5     +9.5%  n.s.   p0=1e6  -5.3%  n.s.
```

**Only the overconfident arm is significant.** Every setting at `p0 >= 1`,
including removing the prior outright, sits within ~2 SE of correct. There is no
evidence the prior helps or hurts accuracy when honestly specified — expected,
since it claims sigma ~ 1732 m on x0 while the measurements determine x0 to
~40 m. It is roughly 40x looser than the data.

**Caution on reading the `p0 < 1` arm.** It samples the error at 1.0 m/s and
then tells the estimator to trust it to 0.25 m/s. Its degradation is arithmetic,
not a discovery. What it *does* give is a sensitivity figure: **a
four-times-optimistic prior costs +42-58%.** Real covariances from TLEs or a
prior OD are often optimistic, so err conservative.

### 16.3 What the prior is actually for

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
iterations — because with `dv -> 0` the `t*` gradient vanishes (9.3), so `t*` is
unidentifiable and the solver wanders. `RIC0.5/FGO-G`, where a real burn gives
`t*` something to lock onto, still converges 9/10 with no prior at all.

**Prior strength is a convergence-versus-accuracy dial.** Tighter converges
faster and more reliably at the cost of bias. It is a genuine trade, not a free
win, and should be set deliberately rather than tuned until runs converge.

---

## 17. The impulse model mis-specification

The truth propagator applies an **instantaneous** delta-v at `t*`; the FGO
models it as a Gaussian of width epsilon **centred** on `t*`, so half the
impulse is applied before the manoeuvre occurs.

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

### 17.1 t* sits exactly on the dt grid

`t_star_true = 12960.0 s` = 216 x 60 exactly, because `mc_fgo.propagate_truth`
takes it as the last epoch of the pre-manoeuvre propagation. A symmetric pulse
centred on a grid node puts half its impulse a full step early:

```
0.5 * |dv| * dt = 0.5 * 0.866025 * 60 = 25.981 m
```

which is precisely the "narrowing alone plateaus at ~26 m" result. **The plateau
is a grid-boundary artefact, not a fundamental limit of narrow pulses.**

With the pulse centred at `t* + delay`, the residual error is just the timing
delay, `|dv| * delay`, confirmed to ~5%:

| epsilon | delay | measured | 0.866 x delay |
|---|---|---|---|
| 10 | 30 | 26.05 m | 25.98 |
| 5 | 15 | 13.06 m | 12.99 |
| 3 | 9 | 7.87 m | 7.79 |
| 1 | 3 | 2.67 m | 2.60 |
| 0.5 | 1.5 | 1.37 m | 1.30 |

### 17.2 Why RK4 cannot resolve a narrow pulse

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
true increment, and it is what section 5's sub-30 degradation is measuring.

---

## 18. Hypotheses tested and rejected

Recorded so they are not retried.

| hypothesis | result |
|---|---|
| The prior causes the 2" failures | No. Failures occur with and without it. |
| Failure correlates with initial-guess quality | No. corr = -0.110 (dv), -0.161 (t*), -0.052 (pos RMS at 10"). |
| Sub-stepping fixes the manoeuvre-step error | No. 0.54 m of 36.67 m (17). |
| It's a local minimum / second basin | No. All seeds reach the same basin given budget (13). |
| lambda grows and freezes t* | No. lambda never grows at all (15.2). |
| Any LM damping helps | **No — settled.** Every tau >= 1e-12 returns the initial guess (15.2). |
| `lambda*diag(M)` damping helps | Rejected twice, but BOTH screens were invalid: the first ran under the broken stall counter pre-Q-fix; the second compared forms at equal lambda0, which post-Q is a 3.6e9x difference in relative damping. The valid test is the tau sweep. |
| Nielsen gain-ratio lambda rule | Worse than shipped at matched damping form (2" s3: 828.3 vs 69.2 m). A line search almost always accepts, so the rejection branch never fires and no gain-ratio rule can grow lambda while a line search sits in front of it. |
| Removing damping entirely (pure GN) | **Helps** (15.2). |
| Normal equations lose all precision | No. `||(M+lam*I)dx - L'y|| / ||L'y|| = 9.2e-13`. But LSMR needs >20,000 iterations without converging, implying cond(M) >~ 1e9 — badly conditioned, not catastrophically. |
| Pulse width epsilon floors t* precision | **No** (5). Flat over the quadrature-valid range. |
| Sparse per-step delta-v reparametrisation | **Out of scope** — would delete the object of study (12). |

---

## 19. Predictions scorecard

| prediction | outcome |
|---|---|
| OD core monotonic in noise, no divergences | **Confirmed** (3.1). |
| FGO-G on RIC0 only slightly worse than FGO-B | **Partly.** +53-74% — same order, but more than "slight". |
| FGO-B on RIC0.5 will not diverge | **Confirmed**, strongly: std/mean = 0.1% (10.1). |
| FGO-B on RIC0.5 stays in the hundreds of metres | **Wrong.** 1.5-1.7 km. |
| Q fix hurts FGO-B on RIC0.5 disproportionately | **Supported**, not proven — trend only. |
| Strengthening the prior improves the pre-arc, not the post-arc | **Confirmed** (16.1). Post-arc moves +0.00 +/- 0.07 m under a 4x change. |
| Pulse width epsilon floors t* precision | **Wrong** (5). |
| Moving the burn later helps the pre-arc | **Confirmed** (4), -47% to -58%, all paired-significant. |
| epsilon and pure GN both help convergence and compose | **Confirmed** (8). 48 -> 8 mean iterations together; accuracy untouched. |
| FGO-G costs accuracy relative to FGO-B | **Wrong at low noise** (3.2). At 1-2 arcsec FGO-G is 17-21% BETTER, even with no manoeuvre present. |
| That low-noise gain is absorbed dynamics model error | **Confirmed** (9). Crossover at 4 arcsec, where the systematic and noise-driven components of the spurious dv cross. |
| sigma_t* scales as sigma_meas / \|dv\| | **Confirmed** (6), to ~1% on an independent config. |
| In-track burns are better timed than cross-track | **Wrong** (6). At 1 arcsec cross-track is 34% BETTER at equal \|dv\|. Unexplained. |
| Range removes the low-noise FGO-G advantage | **Confirmed** (7). -20% collapses to -1.5% n.s. at 2 arcsec. |
| `p0_scale = 1` is optimal for accuracy | **Wrong** (16.2). Everything at `p0 >= 1` is statistically indistinguishable; only over-confidence is significant. An earlier reading of a "monotonic improvement as the prior weakens" was noise. |

---

## 20. Tooling and reproducing these results

Everything is in **`analysis/angles_only_2026-08-25/`** (see its README for a
file-by-file description).

**The current harness:**

- `mc_harness.py` — configurable across epoch x epsilon x damping x mode x
  noise x seeds, with `--range`. Produced Part I sections 4, 6, 7, 8 and 9.

      python analysis/angles_only_2026-08-25/mc_harness.py --tag mytest \
             --epochs pre040 --modes FGO-B FGO-G --noises 2.0 1.0 \
             --seeds 75 --eps 100 --damping off --workers 10

- `tier0_diag.py` — supplies `build_seed`, which every harness imports. It
  replicates `mc_fgo.run_fgo_seed`'s RNG draw order exactly, which is what makes
  these seed numbers comparable with the Monte Carlo drivers, and what makes
  FGO-B/FGO-G paired.
- `truth_cache.py` — loads truth from `out/*.csv` rather than re-propagating,
  and sidesteps the fixed `/tmp` path that makes `mc_fgo.propagate_truth` race
  under parallel use.

**Earlier harnesses:** `sweep_real.py` (calls the real `opt()`; produced the
Part III regressions), `od_study.py` (isolation study), `phase4_prior.py`
(section 16), `sweep_eps_epoch.py` (sections 4 and 5), `damping_sweep.py`
(15.2).

**Raw results:** `mc_mc75.json` (the 900-run MC), `mc_cross.json` (noise
crossover), `mc_dvscale.json` (section 6), `mc_range.json` (section 7),
`epoch_sweep.json`, `eps_sweep.json`, `phase4_main.json`, `sweep300.json` (the
pre-fix baseline), `real_step1/2/3.json` (regressions after each fix).

**Caveat recorded in the README:** `seed_sweep.py`, `seed_sweep_rounded.py` and
`sweep300.py` **reimplement** `opt()` instead of calling it. They are superseded
and will not reflect solver changes made after 2026-08-25.

**Not carried over:** `check_jacobian` (verifies `create_L` against a finite
difference of `create_y`, exploiting `y(x + d) ~= y(x) - L*d`) should be added
to the repo permanently (see TODO.md) — it is essential before section 12
changes how the impulse enters both functions. Baseline on current code:

```
short arc, 1300 cols exhaustive : max rel discrepancy 1.07e-08
one_rev,   9946 cols exhaustive : max rel discrepancy 1.09e-08,  0 cols > 1e-3
```

**Note on repo state:** `epsilon = 100` and pure Gauss-Newton are used by the
harness but are **not** applied to `Orbit_FGO.py` or the configs. The repo's
default behaviour is still epsilon = 30 with the vestigial damping.
