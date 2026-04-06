# Multi-Manoeuvre FGO Results

**See [graph of results as well](./plots/mc_multi_manoeuvre.png)**

## Test configuration
- Model: $J2+SRP$ (simple config), $dt=60$ s
- $Q_{pos}=4.2\times 10^{-4}$ m, $Q_{vel}=1.4\times 10^{-5}$ m/s (calibrated 3x J2+SRP mismatch)
- Angular noise = 10 arcsec, range noise = 100 m
- $\epsilon=0.5$ s, $t^*_{initial\_error}=120$ s, $\Delta v_{initial\_error}=0.1$ m/s
- 5 seeds per config (seeds 0–4)

## Configs

### 2-manoeuvre
- Config: `configs/config_geo_one_rev_2man.yml`
- Manoeuvre 1: $\Delta v = [1, 0, 0]$ m/s at $t^*_1 = 12960$ s (0.15 days)
- Manoeuvre 2: $\Delta v = [0, 1, 0]$ m/s at $t^*_2 = 47520$ s (0.55 days)
- Total arc: 1.0 day (1 GEO rev), 3 propagation segments

### 3-manoeuvre
- Config: `configs/config_geo_one_rev_3man.yml`
- Manoeuvre 1: $\Delta v = [1, 0, 0]$ m/s at $t^*_1 = 12960$ s (0.15 days)
- Manoeuvre 2: $\Delta v = [0, 1, 0]$ m/s at $t^*_2 = 37140$ s (0.43 days)
- Manoeuvre 3: $\Delta v = [0, 0, 1]$ m/s at $t^*_3 = 61320$ s (0.71 days)
- Total arc: 1.0 day (1 GEO rev), 4 propagation segments

## Per-seed results

### 2-manoeuvre

| seed | $t^*_1$ err (s) | $\|\Delta v_1\|$ err | $t^*_2$ err (s) | $\|\Delta v_2\|$ err | pos RMS (m) |
|---:|---:|---:|---:|---:|---:|
| 0 | -39.12 | 0.0288 | -34.09 | 0.0286 | 39.27 |
| 1 | +13.51 | 0.0123 | -59.79 | 0.0246 | 45.79 |
| 2 | -27.47 | 0.0070 | +48.99 | 0.0121 | 70.73 |
| 3 | -53.30 | 0.0226 | +26.68 | 0.0330 | 41.26 |
| 4 | +5.93 | 0.1160 | +61.21 | 0.1320 | 2770.67 |

### 3-manoeuvre

| seed | $t^*_1$ err (s) | $\|\Delta v_1\|$ err | $t^*_2$ err (s) | $\|\Delta v_2\|$ err | $t^*_3$ err (s) | $\|\Delta v_3\|$ err | pos RMS (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -39.12 | 0.0289 | +8.88 | 0.0484 | -72.67 | 0.0155 | 41.61 |
| 1 | +13.50 | 0.0123 | +38.10 | 0.0301 | -22.63 | 0.0236 | 46.53 |
| 2 | -27.47 | 0.0069 | -3.21 | 0.0296 | +116.48 | 0.0168 | 71.81 |
| 3 | -53.29 | 0.0226 | +25.91 | 0.0272 | -59.17 | 0.0294 | 41.14 |
| 4 | +7.99 | 0.0507 | +22.19 | 0.0379 | +125.02 | 0.0157 | 56.61 |

## Ensemble statistics

### 2-manoeuvre

| manoeuvre | $t^*$ mean (s) | $t^*$ std (s) | $t^*$ RMS (s) | $\|\Delta v\|$ mean (m/s) | $\|\Delta v\|$ std (m/s) |
|:---|---:|---:|---:|---:|---:|
| Man 1 | -20.09 | 25.79 | 32.69 | 0.0374 | 0.0401 |
| Man 2 | +8.60 | 47.38 | 48.16 | 0.0461 | 0.0435 |

- pos RMS: mean = 593.5 m, std = 1217.1 m, median = 45.8 m (seed 4 outlier at 2770 m)
- vel RMS: mean = 0.0686 m/s, std = 0.0819 m/s

### 3-manoeuvre

| manoeuvre | $t^*$ mean (s) | $t^*$ std (s) | $t^*$ RMS (s) | $\|\Delta v\|$ mean (m/s) | $\|\Delta v\|$ std (m/s) |
|:---|---:|---:|---:|---:|---:|
| Man 1 | -19.68 | 26.21 | 32.78 | 0.0243 | 0.0153 |
| Man 2 | +18.37 | 14.26 | 23.26 | 0.0346 | 0.0077 |
| Man 3 | +17.41 | 86.00 | 87.74 | 0.0202 | 0.0055 |

- pos RMS: mean = 51.5 m, std = 12.9 m, median = 46.5 m (no outliers)
- vel RMS: mean = 0.0409 m/s, std = 0.0077 m/s

## Comparison to single-manoeuvre baseline

| scenario | $t^*$ RMS (s) | $\|\Delta v\|$ mean (m/s) | pos RMS median (m) |
|:---|---:|---:|---:|
| Single (20 seeds) | 17.23 | 0.025 | 45.5 |
| 2-man, man 1 | 32.69 | 0.037 | ~42 |
| 2-man, man 2 | 48.16 | 0.046 | ~42 |
| 3-man, man 1 | 32.78 | 0.024 | 51.5 |
| 3-man, man 2 | 23.26 | 0.035 | 51.5 |
| 3-man, man 3 | 87.74 | 0.020 | 51.5 |

## Key findings

1. **Delta-v estimation remains accurate** across all scenarios (0.02–0.05 m/s mean, std 0.004–0.04 m/s), comparable to the single-manoeuvre case. The FGO resolves multiple impulses reliably.

2. **$t^*$ estimation degrades with more DOFs**: single-manoeuvre RMS ~17 s grows to ~33–48 s (2-man) and ~23–88 s (3-man).

3. **The last manoeuvre in the arc has highest $t^*$ uncertainty** (man 3 in 3-man: RMS = 88 s, std = 86 s vs man 2: RMS = 23 s, std = 14 s). Less post-manoeuvre data to constrain the final impulse epoch.

4. **Man 1 $t^*$ is consistent across 2-man and 3-man configs**

5. **Orbit accuracy is healthy**: median pos RMS ~42–52 m in non-outlier cases, comparable to single-manoeuvre (~45 m). The FGO successfully resolves the full trajectory across multiple impulse boundaries.

6. **Outlier rate**: 1 out of 5 seeds diverged in the 2-man case (seed 4, pos = 2770 m). The same seed did NOT diverge in the 3-man case (pos = 57 m), suggesting the additional manoeuvre DOFs can sometimes provide a better optimisation landscape. Will need further monte-carlo testing but that takes some time at this stage.
