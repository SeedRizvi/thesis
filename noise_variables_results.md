# Noise Analysis at different stages of development
- Using a Simplfied Model which is $J2$ only.

## Delta-V estimation ()
Biggest contributors from most to least are:
1. Process noise velocity (Q_vel)
2. Measurement Angular noise
3. Measurement Range noise

$Q_{pos}$ - No impact

| $Q_{pos}$ (m) | \|$\Delta v_{error}$\| (m/s) |
|------:|-------------------:|
| 1.0 | 0.1109 |
| 10.0 | 0.1110 |
| 50.0 | 0.1110 |
| 100.0 | 0.1108 |
| 500.0 | 0.1063 |

$Q_{vel}$ - DOMINANT bottleneck

| $Q_{vel}$ (m/s) | \|$\Delta v_{error}$\| (m/s) |
|------:|-------------------:|
| 0.0001 | 0.0454 |
| 0.001 | 0.1108 |
| 0.005 | 0.1893 |
| 0.01 | 0.2451 |
| 0.1 | 0.4101 |

Measurement range noise - mild impact

| $\sigma_{range}$ (m) | \|$\Delta v_{error}$\| (m/s) |
|------:|-------------------:|
| 1.0 | 0.0860 |
| 10.0 | 0.0778 |
| 50.0 | 0.0913 |
| 100.0 | 0.1108 |
| 500.0 | 0.1048 |

Measurement angle noise - moderate impact [1, 5, 10, 20, 36 arcsec]

| $\sigma_{angle}$ (deg) | \|$\Delta v_{error}$\| (m/s) |
|------:|-------------------:|
| 0.00028 | 0.0707 |
| 0.00139 | 0.0849 |
| 0.00278 | 0.1108 |
| 0.00556 | 0.1230 |
| 0.01 | 0.1255 |

initial position state error - zero impact

| $\sigma_{pos,0}$ (m) | \|$\Delta v_{error}$\| (m/s) |
|------:|-------------------:|
| 10.0 | 0.1108 |
| 100.0 | 0.1108 |
| 500.0 | 0.1108 |
| 1000.0 | 0.1108 |
| 5000.0 | 0.1108 |

## Manoeuvre Epoch estimation
- Simplified model ($J2$ only)
- $\Delta v$ = [1.0, 1.0, 1.0] m/s
- $t^*_{initial\_error}$ = 120 s
- $\Delta v_{initial\_error}$ = 0.1 m/s

### Calibrated Q Results
- Uses 3x larger $Q$ covariances than calculated from dynamics mismatch
- Scaled from testing:
  - $Q_{vel} \propto dt$
  - $Q_{pos} \propto dt^2$

| $dt$ (s) | $Q_{vel}$ | $Q_{pos}$ | $t^*$ error (s) | \|$\Delta v_{error}$\| (m/s) |
|-------:|------:|------:|------------:|-------------------:|
| 5 | 0.0000011 | 0.0000029 | - | - |
| 10 | 0.0000023 | 0.0000114 | 1.84 | 0.0256 |
| 15 | 0.0000034 | 0.0000257 | - | - |
| 30 | 0.0000069 | 0.000103 | 4.08 | 0.0332 |
| 60 | 0.0000137 | 0.000413 | -11.39 | 0.0230 |

For reference, the paper (Zhang et al.): $t^*$ RMSE = 2.86s at $dt=10s$ with batch least-squares.

### Conservative Q results
- Uses 10x larger $Q$ covariances than calculated from dynamics mismatch

| $dt$ (s) | $Q_{vel}$ | $Q_{pos}$ | $t^*$ error (s) | \|$\Delta v_{error}$\| (m/s) |
|-------:|------:|------:|------------:|-------------------:|
| 10 | 0.000046 | 0.00138 | -4.36 | 0.1169 |
| 60 | 0.000046 | 0.00138 | -19.14 | 0.0339 |
| 60 | 0.000046 | 100.0 | -20.21 | 0.0318 |
| 30 | 0.000046 | 100.0 | 24.79 | 0.0468 |

## Monte Carlo t* distribution ($J2+SRP$, $dt=60$ s)
- Model: $J2+SRP$ (simple config)
- $Q_{pos}=4.2\times 10^{-4}$, $Q_{vel}=1.4\times 10^{-5}$ (3x J2+SRP mismatch)
- $\Delta v$ = [1.0, 1.0, 1.0] m/s, angular noise = 10 arcsec, range noise = 100 m
- $t^*_{initial\_error}$ = 120 s, $\Delta v_{initial\_error}$ = 0.1 m/s
- Independent RNG seed per trial (measurements, $x_0$, $\Delta v$/$t^*$ initial guesses)

### 20-seed distribution

| metric | mean | std-dev | RMS | min | max | median |
|:------|------:|------:|------:|------:|------:|------:|
| $t^*$ error (s) | -1.52 | 17.16 | 17.23 | -26.50 | +60.41 | -1.95 |
| \|$\Delta v_{error}$\| (m/s) | 0.058 | 0.135 | 0.147 | 0.005 | 0.642 | 0.025 |
| pos RMS (m) | 545.4 | 2171.7 | 2239.2 | 36.2 | 10011.6 | 45.5 |
| vel RMS (m/s) | 0.067 | 0.147 | 0.161 | 0.011 | 0.702 | 0.047 |

- Roughly unbiased in $t^*$ (mean $\approx$ median $\approx -2$ s) with $\sigma \approx 17$ s at $dt=60$.
- One extreme outlier (seed 19) diverged to pos RMS = 10 km; median stats reflect typical behaviour.

### Angular-noise sensitivity (5 seeds/level, same $Q$)

| $\sigma_{angle}$ (arcsec) | $t^*$ RMS (s) | $t^*$ mean (s) | $t^*$ std-dev (s) | \|$\Delta v_{error}$\| (m/s) | pos RMS (m) | vel RMS (m/s) |
|------:|------:|------:|------:|------:|------:|------:|
| 1.0 | 11.62 | -8.72 | 7.69 | 0.022 | 14.4 | 0.0446 |
| 5.0 | 14.07 | -9.36 | 10.51 | 0.023 | 37.4 | 0.0395 |
| 10.0 | 13.93 | -8.89 | 10.72 | 0.025 | 49.3 | 0.0403 |
| 36.0 | 13.82 | -8.47 | 10.92 | 0.029 | 68.0 | 0.0408 |

- **$t^*$ error is seed-locked, not noise-level-locked**: per-seed $t^*$ values stay within ~2 s of themselves across the full 1"-36" range (e.g. seed 3 gives $-22$ to $-27$ s at all noise levels).
- **Position RMS scales linearly with angular noise** (14 m at 1" $\to$ 68 m at 36"), confirming the measurement channel is healthy.
- **Conclusion**: at $dt=60$ the $t^*$ precision floor is set by dynamics/sub-stepping coupling (Gaussian pulse window vs $dt$ grid), not measurement noise. Matches the earlier $dt$-scaling observation ($t^*$ error $\propto dt$).
