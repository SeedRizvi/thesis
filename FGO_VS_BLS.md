# Why FGO and BLS Perform Identically — and What Separates Them

Status as of 2026-08-28. All numbers measured, 25 seeds unless stated.

Companion to `ANGLES_ONLY_FGO.md`. That document covers angles-only accuracy and
the solver; this one covers a separate question raised by the 50-seed
preliminary Monte Carlo: **why do two structurally different estimators produce
identical results, and what makes them differ?**

---

## 1. The problem

On the shipped configuration (1.15 day arc, 2-body+J2 truth, angles-only,
2 arcsec) FGO and BLS agree to within measurement precision in every
manoeuvre-estimating (`-G`) cell of the 50-seed MC:

```
config           FGO-G      BLS-G      dv_err FGO   dv_err BLS   t* FGO   t* BLS
deltaC0.2        24.3 m     24.3 m       0.0033       0.0033      71.4 s   71.4 s
deltaI0.2        24.1 m     24.2 m       0.0039       0.0039     116.3    116.2
deltaRIC0.5      24.5 m     24.5 m       0.0034       0.0034      31.3     31.4
deltaRIC1        24.6 m     24.7 m       0.0035       0.0035      15.9     16.0
```

Agreement to 3-4 significant figures in position, delta-v AND t*. That is a
problem for a manuscript whose aim is to show FGO's unique value for manoeuvre
estimation: on these results there is no argument for adopting it.

---

## 2. What "bending" is

The FGO does not estimate 6 numbers. It estimates **every state on the
trajectory** as a free variable (`x_0 ... x_1655`, i.e. 9,940 unknowns at
1.15 days). What stops them wandering is the dynamics factors -- for each
consecutive pair, a residual

```
( x_{k+1} - f(x_k) ) / sigma_Q
```

where `f` is the propagator. That penalises any departure from "the next state
is the propagated previous state", weighted by the process noise.

- **sigma_Q tiny** -> the penalty is overwhelming, `x_{k+1} = f(x_k)` to within
  microns, and the whole trajectory is a deterministic propagation of `x_0`.
  The 9,940 variables encode nothing beyond the 6 in `x_0`: a **redundant
  parameterisation of exactly what BLS solves**.
- **worth paying** -> each step drifts slightly off the propagated path in
  whatever direction the measurements favour. Over thousands of steps this
  accumulates into a trajectory no single initial condition could produce under
  the model. That is *bending*, and physically it is the FGO absorbing
  acceleration it cannot model.

**Measuring it (`rigid_dev`).** Take the converged FGO solution, propagate
rigidly from its OWN `x_0` and its OWN `(dv, t*)`, and RMS the difference
against its estimated states. That is the part of its answer that is not a
deterministic propagation.

**Why separation tracks it exactly.** BLS's entire solution space *is* rigid
trajectories -- every candidate it can represent is a deterministic propagation
from some `(x_0, dv, t*)`. So:

- `rigid_dev ~ 0` -> the FGO's optimum lies INSIDE BLS's search space. Both find
  the same point. Identical results, by construction.
- `rigid_dev` large -> the FGO's optimum lies OUTSIDE what BLS can express. BLS
  must settle for the best rigid approximation and falls short by however much
  of the solution it cannot represent.

**`rigid_dev` is not correlated with the FGO's advantage -- it IS the advantage,
measured in metres.**

---

## 3. Evidence

`RIC0.5`, FGO-G, 2-body+J2 truth, angles-only 2 arcsec:

| arc | N | rigid_dev | pos_rms | BLS/FGO |
|---|---|---|---|---|
| 0.15 d (short) | 216 | 0.001 m | 484.7 m | 1.00x |
| 1.15 d | 1656 | 1.17 m | 24.0 m | 1.00x |
| 3.0 d | 4320 | 773 m | 25.3 m | **4.54x** |

The separation appears exactly when the bending does. At 1.15 days the part of
the FGO's answer BLS cannot represent is ~1 metre against a 24 m error --
nothing to distinguish.

**The `-B` mode is the control.** With no manoeuvre parameters the FGO is forced
to bend (nothing else can explain the burn) and it beats BLS even on the
baseline: `rigid_dev` 6467 m, BLS/FGO 1.24x. Same Q, same arc, same code -- the
only change is removing the cheap explanation.

**Why `-G` suppresses it.** `(dv, t*)` are constrained only by a weak prior and
are therefore nearly free, while bending is charged at 1/sigma_Q^2. Per step,
Q allows sigma_pos = 5.222e-04 m while a 2 arcsec measurement at GEO constrains
position to ~407 m -- the dynamics factors are **7.8e5 times stiffer than the
measurements**. Given both options the optimiser takes the cheap one and leaves
the trajectory rigid. It *chooses* rigidity; it is not forced into it.

---

## 4. The 2x2 (25 seeds, 800 runs)

Two independent levers, each roughly 4x alone, compounding to 8.4x.
`RIC0.5`, `-G`, position RMS:

| | 2-body+J2 | +SRP/moon/sun |
|---|---|---|
| **1.15 d** | 24.04 / 24.06 = **1.00x** n.s. | 253.1 / 992.4 = **3.92x** * |
| **3.0 d** | 25.33 / 115.03 = **4.54x** * | 133.7 / 1122.6 = **8.40x** * |

(FGO / BLS. `*` = paired difference exceeds 2 standard errors. The baseline cell
is `+0.1% +/- 0.1`, i.e. a clean null.)

`rigid_dev` across the same four cells: 1.17 -> 5286 -> 773 -> 18063 m.

### Manoeuvre estimation

| scenario | FGO dv | BLS dv | ratio | FGO t* | BLS t* | ratio |
|---|---|---|---|---|---|---|
| 1.15 d, 2body+J2 | 0.00370 | 0.00370 | 1.0x | 29.2 s | 29.3 s | 1.0x |
| 1.15 d, +lunisolar | 0.16325 | 0.21592 | 1.3x | 84.0 s | 2211.1 s | 26.3x |
| **3.0 d, 2body+J2** | **0.00206** | **0.00739** | **3.6x** | **14.1 s** | **221.6 s** | **15.7x** |
| 3.0 d, +lunisolar | 0.16539 | 0.32027 | 1.9x | 59.1 s | 45.1 s | 0.8x |

**The 3-day / 2-body+J2 cell is the strongest result in the study.** FGO gets
delta-v to 0.00206 m/s (0.24% of the 0.866 m/s burn, 3.6x better than BLS) and
t* to 14.1 s against BLS's 221.6 s -- a **15.7x timing advantage** -- while
*keeping* absolute accuracy excellent (25 m position).

The luni-solar arms separate the estimators more dramatically but degrade
delta-v by ~80x in absolute terms (0.0021 -> 0.165 m/s), which weakens the
manoeuvre-estimation claim even as it strengthens the FGO-vs-BLS one.

**Anomaly:** in `3.0 d + lunisolar` BLS times the burn *better* (45.1 vs 59.1 s,
0.8x). Persists at 25 seeds, so it is not sampling noise. Unexplained.

---

## 5. The relation to N is a threshold, not a power law

The incentive argument: spreading a delta-v across n steps costs each step
`(dv/n)/sigma_v`, so the total penalty is `dv^2/(n sigma_v^2)` -- **cost ~ 1/n** --
while the benefit accrues at every measurement epoch, **benefit ~ n**. Hence
**benefit/cost ~ n^2**.

That predicts the direction correctly and identifies N as the controlling
variable. It does **not** predict the magnitude:

```
N  216 -> 1656  (x7.67)   rigid_dev x 632   implies N^3.17   (n^2 predicts x59)
N 1656 -> 4320  (x2.61)   rigid_dev x1188   implies N^7.38   (n^2 predicts x6.8)
```

The implied exponent is not constant, so no single power law fits. Both steps
produce a factor of ~600-1200 regardless of how much N actually changed. That is
the signature of a **threshold**: below it the optimiser stays essentially rigid
(1 mm of deviation over the whole short arc), above it bends as far as the
Q-versus-measurement trade-off allows (751 m). **n^2 tells you which side of the
threshold you are on; it does not tell you how much bending you get.** Present
it as a regime change with a crossover between roughly 1 and 3 days, not as a
scaling law -- a fitted exponent would be fitting the transition.

**Do not confuse this with 1/sqrt(N) measurement averaging.** A fixed-parameter
estimator averaging N measurements has error ~ sigma/sqrt(N). Going 1656 ->
4320 (x2.61) predicts 24.0 -> 14.9 m for both. Measured:

```
  BLS-G RIC0.5   24.06 -> 115.03 m   (4.8x WORSE)
  FGO-G RIC0.5   24.04 ->  25.33 m   (flat)
```

1/sqrt(N) averaging only helps when the model is right. BLS is **model-limited**:
a longer arc gives it more accumulated dynamics error to absorb into six rigid
parameters, so more data actively hurts. FGO's process noise soaks that up.
*More data only helps if your model can hold it.*

---

## 6. Why this was hidden: the Q history

Pre-fix, the Q std-dev/variance bug (`ANGLES_ONLY_FGO.md` 14.1) gave an
effective sigma of sqrt(5.222e-04) = **0.0229 m**, 43.8x looser than the correct
5.222e-04 m. Bending cost goes as 1/sigma^2, so deviation used to be ~1900x
cheaper -- which is why FGO used to beat BLS.

The measured Q for a luni-solar truth model is **0.0300 m** -- within 30% of that
buggy value. The units bug and the missing forces had been cancelling: the old
results used roughly the right process noise for a realistic dynamics model,
in a scenario that did not contain one. Fixing one without the other collapsed
the comparison.

So restoring the separation is not tuning back to a flattering number. It is
making the scenario honestly match the Q we were already effectively using.

**Measured per-step mismatch (truth vs FGO 2-body+J2, 1.15 d arc):**

```
variant                    |pos| m    vs base    q_pos_R (5xRMS)
baseline (2body+J2)      1.348e-04       1.0x     5.2229e-04
+SRP                     1.632e-04       1.2x     6.3362e-04
+moon                    8.583e-03      63.7x     2.4719e-02
+sun                     4.546e-03      33.7x     1.5420e-02
+SRP+moon+sun            1.060e-02      78.6x     3.0046e-02
+SRP+moon+sun+grav8      1.059e-02      78.6x     2.9983e-02
```

Ordering at GEO for this satellite (0.01 m^2/kg): **Moon > Sun >> SRP**. SRP adds
20% alone and nothing once luni-solar is on. Gravity degree 8 adds nothing
measurable. The RIC anisotropy also changes character: 1 : 0.70 : 0.42
(radial-dominated, J2/integration error) becomes 1 : 0.996 : 1.058 (isotropic).

---

## 7. Recommendation

**Longer arc is the better primary lever.** It separates FGO from BLS 4.5x,
*improves* both estimators' manoeuvre estimates rather than wrecking them,
requires no change to the force model, and is trivially defensible -- GEO SSA
tracking arcs are routinely multi-day.

**Luni-solar is the stronger secondary argument**: more physically honest, and it
demonstrates robustness to model error, but it costs ~80x in absolute delta-v
accuracy.

Presenting the 2x2 is stronger than either alone: it shows the advantage is
systematic in the two independent directions the mechanism predicts, and that
they compound.

**Framing for the manuscript.** The honest statement is not "FGO is more accurate
than BLS". It is: *FGO's solution space strictly contains BLS's, and the extra
capacity is worth exactly the amount of model error the manoeuvre parameters
cannot absorb.* When the dynamics model is near-perfect and the arc short, the
two are the same estimator and should be reported as such.

---

## 8. Open questions

1. **Why does BLS time the burn better in `3.0 d + lunisolar`** (45.1 vs 59.1 s)?
   Persists at 25 seeds. Contradicts every other cell.
2. **Where exactly is the arc-length threshold?** Only 216 / 1656 / 4320 sampled.
   A sweep would locate the crossover and make the regime-change claim concrete.
3. **Is 5x the RMS mismatch the right Q margin?** Never deliberately chosen
   (see TODO.md). Now load-bearing, since Q sets the bending cost directly.
4. **Does the prior remain accuracy-neutral at the looser Q?**
   `ANGLES_ONLY_FGO.md` 16 measured that at sigma_Q = 5.2e-04; the finding may
   not survive at 3.0e-02.
5. **Short arc**: FGO uses 0.00% freedom there even in `-B`. Confirmed, but the
   216-step arc is also parallax-starved (`ANGLES_ONLY_FGO.md` 4), so the two
   effects are confounded in that config.

---

## 9. Reproducing

Scripts in the session scratchpad (not yet promoted to the repo):
`screen2x2.py` (the 800-run 2x2), `force_cal.py` (section 6 table),
`pinned.py` / `pinned_short.py` (the `rigid_dev` diagnostic).

The `rigid_dev` measurement is worth promoting to the repo permanently -- it is
the single number that predicts whether FGO and BLS will differ, and it costs
one extra propagation per solve:

```python
rigid = np.zeros_like(fgo.states); rigid[0] = fgo.states[0]
for i in range(1, fgo.N):
    rigid[i] = fgo.prop_one_timestep(rigid[i-1], (i-1)*fgo.dt)
rigid_dev = rms(norm(fgo.states[:, :3] - rigid[:, :3], axis=1))
```

**Note:** `mc_fgo.propagate_truth` had a bug that silently corrupted truth for any
config not written in YAML flow style (`initial_state: [...]`). Fixed 2026-08-28
by editing the parsed dict instead of regexing raw text. Any result generated
with programmatically-written configs before that fix is invalid.
