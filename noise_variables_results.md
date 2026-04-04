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





