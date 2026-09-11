- Add an epsilon sweep section to interim_fgo_vs_bls.md. Add an --eps-sweep mode
  to report_data/interim_suite/interim_suite.py alongside --arc-sweep, then render
  it as the section after "Baseline". Suggested eps = 20,25,30,45,60,100,150,200,300
  x {RIC0, RIC0.5, I0.2} x {FGO, BLS, EKF}, -G only (-B is invariant to epsilon:
  it enters only via the manoeuvre term).
  A pre-GMST probe found that small epsilon costs convergence rather than accuracy
  (position RMS flat from eps 5 to 300; eps=10 hit the iteration cap on 3/5 seeds,
  eps=15 on 1/5, eps>=20 converged on every seed). That probe's raw data was
  scratchpad-only and is gone, and it predates the GMST fix which changed
  convergence substantially -- so re-measure rather than cite it.
- Update project structure in README: it still lists only a handful of files and
  predates mc_runner.py, the mc_* drivers, Orbit_BLS/EKF and report_data/.
- SETTLED (2026-09-11, supervisor): Q recalibrated per scenario stays as-is,
  including for the third-body cases where it is sized to forces the estimators do
  not model. Rationale: process noise is inherent to the FGO formulation and BLS
  omits it by construction, so the difference is a property of the estimators and
  a supporting argument for the discussion rather than an unfair advantage.
  Evidence retained for the write-up: holding Q at the 2-body+J2 value collapses
  the lunisolar advantage from 4.63x to 1.09x (-G) and 15.26x to 2.02x (-B), with
  rigid_dev falling 3405 -> 398 m; see the "baseline Q" section of
  interim_fgo_vs_bls.md. A Q sweep over 1x-20x the per-step RMS mismatch showed the
  shipped 5x is conservative (loosening it increases the FGO's margin) and that the
  clean-arc result is nearly Q-insensitive (13.89 -> 13.28 m over a 20x range).
  The per-step mismatch is strongly correlated (rho_1 ~ 0.999, tau ~ 225 steps),
  so a white-noise Q is an approximation; dynamic model compensation with a shared
  acceleration parameterisation is the principled alternative, deliberately scoped
  out and worth a line in the discussion.
