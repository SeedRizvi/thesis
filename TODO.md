- Finish the epsilon sweep section for interim_fgo_vs_bls.md. Runner is
  scratchpad eps_suite2.py (15 seeds, -G only, baseline config: 2body+J2,
  1.15 d, 40% epoch, 2 arcsec, dt=60). Sweep eps = 20,25,30,45,60,100,150,200,300
  x {RIC0, RIC0.5, I0.2} x {FGO, BLS, EKF} = 1215 runs, ~30 min on 8 workers.
  Insert as the section after "Baseline" using scratchpad mkeps.py.
  eps <= 15 is deliberately excluded: a 120-run probe (eps_probe.json) shows
  eps=10 caps the 300-iteration limit on 3/5 seeds in both configs and eps=15
  on 1/5, while eps>=20 converges on every seed and eps>=45 is flat at 5
  iterations. Position RMS is constant (26.5-29.3 m) across eps 5 to 300, so
  epsilon costs convergence, not accuracy -- the probe result is arguably the
  more useful finding and is worth reporting alongside the sweep.
  NOTE: -B mode is invariant to epsilon (it enters only via the manoeuvre term),
  so only -G needs sweeping.
- Promote the rigid_dev diagnostic and the 2x2 harness into the repo. rigid_dev
  is the single number that predicts whether FGO and BLS will differ (see
  FGO_VS_BLS.md 2) and costs one extra propagation per solve. screen2x2.py
  produced the 800-run 2x2 and is still scratchpad-only.
- Update project structure in README. Also mention where the manuscript results are in the repo.
- Raise max_iterations from 50 to 200-300 in the configs before the real MC.
  Measured on the 50-seed preliminary: 8/50 short-arc FGO-G seeds hit the cap and
  read 1827-5696 m; given budget every one converges on its own criteria in
  65-132 iterations and lands at 112-489 m, matching BLS exactly. maxit 300 and
  1000 give identical answers, so nothing needs more than ~132. one_rev runtime
  tails (max/p50 of 5-8x on deltaRIC0.5 and deltaRIC1) suggest 50 is close to
  binding there too. Costs nothing since termination is on CONV_REL_PRED.
- Re-run the EKF and BLS Monte Carlos. The Q std-dev/variance fix changed
  Orbit_EKF.compute_Q as well as Orbit_FGO.compute_S_Q_inv, and BLS inherits
  from the FGO class, so every published EKF/BLS number has moved. Only the FGO
  angles-only sweep has been re-measured so far (see ANGLES_ONLY_FGO.md 1).
- SETTLED: do NOT make opt() a Levenberg-Marquardt. The damping was removed
  from both Orbit_FGO and Orbit_BLS on 2026-08-28. LM damps the weakly-observed
  directions that must move, and the prior is already the principled
  regulariser -- see ANGLES_ONLY_FGO.md 15.2 and the 18 hypothesis table before
  re-proposing it.
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
- SETTLED (2026-08-28), recorded so it is not re-investigated: EKF-G |dv| error
  blows up with range at eps=30 (1.031 m/s vs a true 0.866, i.e. worse than
  guessing zero) but is fine angles-only at eps=100 (0.057). Diagnosed: the
  filter is statistically consistent (NIS/dof 0.99-1.03 in all cells, so the
  covariance is NOT optimistic and the Q fix did not make it over-confident).
  dv does not jump at the burn -- it drifts for the whole post-burn arc via the
  cross-covariance P[0:6,6:], which carries real and mostly USEFUL information.
  Process noise on the dv/t* block changes nothing (<4% at any level), so it is
  not a confidence lock-in. Decommissioning dv/t* after the burn window fixes
  the range case (-62%) but is significantly WORSE at eps=100 angles-only
  (+365%), which is the shipped configuration. No fix applied; this is a
  range-plus-narrow-pulse artefact, not a defect. Raw data:
  scratchpad ekf_phase2.json / ekf_phase3.json.
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
