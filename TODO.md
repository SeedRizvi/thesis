- Update project structure in README. Also mention where the manuscript results are in the repo.
- Re-run the EKF and BLS Monte Carlos. The Q std-dev/variance fix changed
  Orbit_EKF.compute_Q as well as Orbit_FGO.compute_S_Q_inv, and BLS inherits
  from the FGO class, so every published EKF/BLS number has moved. Only the FGO
  angles-only sweep has been re-measured so far (see ANGLES_ONLY_FGO.md 1).
- Make opt() a proper Levenberg-Marquardt. Measured: lambda never grows in any
  run (lambda_max = 5e-7 over 20+ runs) because the growth branch only fires
  when the line search fails across all 20 halvings, which never happens. It is
  Gauss-Newton with backtracking, and the damping is vestigial. NOTE: the t*
  column's curvature is 2.53e2 against velocity's 9.49e6, so lambda*I would
  freeze t* first -- curvature-scaled damping must land WITH the lambda-update
  fix, not after it. Three variants have already been measured and rejected
  (diag damping, reactive lambda, textbook Nielsen) but all were confounded with
  the then-broken stall counter; see ANGLES_ONLY_FGO.md 7 before retrying.
  Target: no divergences at max_iters = 50.
- Promote the parallel sweep harness from scratchpad into the repo. Runs
  10 seeds x {10", 2"} x {50, 300 iters} in ~8 min on 10 cores by calling the
  real opt(); validated to reproduce the serial baseline exactly. Every number
  in ANGLES_ONLY_FGO.md came from it, so it needs to be version-controlled for
  the results to be reproducible.
- Re-add check_jacobian to the repo permanently. Verifies create_L against a
  finite difference of create_y. Essential before Option A, which changes how
  the impulse enters both. The reference implementation this was ported from
  had an equivalent (reference_fgo_.py:315 test_Jacobian) that was lost.
- Implement Option A (analytic impulse integration) -- see ANGLES_ONLY_FGO.md 6
  for the formulas, Jacobians, cost measurement and caveats. Deferred until the
  solver converges reliably, otherwise the model improvement cannot be separated
  from convergence luck.
- De-duplicate the process-noise covariance. Orbit_EKF.compute_Q and
  Orbit_FGO.compute_S_Q_inv are two copies of the same rotate-and-square logic.
  The std-dev/variance bug had to be fixed in both; the next one will too.
- Remove the deprecated/unused substepping leftovers. NOTE: sub-stepping was previously
  thought to fix the manoeuvre-step dynamics error. It does not -- see ANGLES_ONLY_FGO.md.
  The 36.7 m error is model mismatch (Gaussian vs instantaneous delta-v), not integration
  error; sub-stepping removes only 0.54 m of it.
- Investigate t* estimation bias in EKF-G. Mean signed error is -20.6s (old P0) / -29.3s
  (config P0) over 8 seeds on deltaRIC0.5, with a -157s outlier. The 120s t* prior is not
  the binding constraint, so this looks separate from the P0 work.
- Remove redundant config .get() fallbacks in the drivers. load_config_parameters already
  defaults every key, but callers re-fetch with disagreeing values (dv: 0.1 in mc_*, 0.5 in
  pipelines; t*: 60.0 at ekf_pipeline.py:176 vs 120.0 at :194). Dead code today, but
  misleading and would diverge if anything ever bypasses the loader.
- Consider whether truth's t* should stay on the dt grid. mc_fgo.propagate_truth
  takes t_star_true as the last epoch of the pre-manoeuvre arc, so it is always an
  exact multiple of dt (12960.0 = 216 x 60). That is what produces the 26 m
  boundary artefact (see ANGLES_ONLY_FGO.md 4.4). A real burn does not respect
  the measurement grid, so an off-grid t* is arguably the more honest scenario --
  and it would also be a cheap direct test of the boundary diagnosis.
- Fix the fixed temp path in mc_fgo.propagate_truth. It writes and then deletes
  /tmp/mc_fgo_post_{tag}.yml, so two processes on the same config race and one
  dies with FileNotFoundError. Harmless today, but it blocks parallelising the
  Monte Carlo across configs. tempfile.mkstemp() or a PID suffix.
- Revisit the Q magnitude, separately from the units fix. The values are now
  correctly treated as standard deviations, but whether 5x the measured RMS
  mismatch is the right margin was never deliberately chosen. Tightening Q was
  a large accuracy win (-56% at 2 arcsec) but made the problem markedly stiffer
  (1/10 seeds now diverges even at 300 iterations).
