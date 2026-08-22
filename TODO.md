- Update project structure in README. Also mention where the manuscript results are in the repo.
- Remove the deprecated/unused substepping leftovers.
- Investigate t* estimation bias in EKF-G. Mean signed error is -20.6s (old P0) / -29.3s
  (config P0) over 8 seeds on deltaRIC0.5, with a -157s outlier. The 120s t* prior is not
  the binding constraint, so this looks separate from the P0 work.
- Remove redundant config .get() fallbacks in the drivers. load_config_parameters already
  defaults every key, but callers re-fetch with disagreeing values (dv: 0.1 in mc_*, 0.5 in
  pipelines; t*: 60.0 at ekf_pipeline.py:176 vs 120.0 at :194). Dead code today, but
  misleading and would diverge if anything ever bypasses the loader.
- Revisit the LM lambda shrink rate in Orbit_FGO.opt(). The gate on the decrease is gone,
  but the rate is still /2 while the growth branch is *10. Symmetric *10 //10 is the
  standard LM form and on short-arc angles-only it was decisive (seed 1: 13008 m at /2 vs
  1892 m at /10). Not adopted yet because short-arc is the config built to favour BLS, so
  tuning a global solver constant on it risks overfitting. Decide from one_rev evidence.
- Consider a relative, dimensionally-consistent step-convergence test in Orbit_FGO.opt().
  The existing `la.norm(delta_x * best_scale) < 1e-3` sums metres, m/s and seconds over
  ~1300-9946 variables, so it is both dimensionally incoherent and extremely tight (tens
  of microns per component). It errs conservative so it is not a correctness risk, but it
  rarely fires, which leaves the stagnation counter as the effective convergence criterion.
