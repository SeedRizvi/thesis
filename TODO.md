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
- Update project structure in README. Also mention where the manuscript results are
  in the repo.
- Revisit the Q magnitude, separately from the units fix. The values are now
  correctly treated as standard deviations, but whether 5x the measured RMS
  mismatch is the right margin was never deliberately chosen. Tightening Q was
  a large accuracy win (-56% at 2 arcsec). It was also thought to make the problem
  markedly stiffer, but that was measured pre-GMST: across the 1920 committed
  solver runs the iteration count is now p50=5, p99=30, p99.9=47, max=83.
- Delete or clearly mark the pre-GMST result files in report_data/ (report_mc_*.csv,
  report_mc_w_range_*.csv, results.md, results_200mc.md). Every number in them was
  produced with the broken measurement geometry and none can be cited. The current
  numbers live in interim_fgo_vs_bls.md and report_data/interim_suite/.
