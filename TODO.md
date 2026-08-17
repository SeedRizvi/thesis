- Update project structure in README. Also mention where the manuscript results are in the repo.
- Remove the deprecated/unused substepping leftovers.
- Investigate t* estimation bias in EKF-G. Mean signed error is -20.6s (old P0) / -29.3s
  (config P0) over 8 seeds on deltaRIC0.5, with a -157s outlier. The 120s t* prior is not
  the binding constraint, so this looks separate from the P0 work.
- Remove redundant config .get() fallbacks in the drivers. load_config_parameters already
  defaults every key, but callers re-fetch with disagreeing values (dv: 0.1 in mc_*, 0.5 in
  pipelines; t*: 60.0 at ekf_pipeline.py:176 vs 120.0 at :194). Dead code today, but
  misleading and would diverge if anything ever bypasses the loader.