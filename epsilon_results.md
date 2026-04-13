# Epsilon Sensitivity Study

## Test configuration
- Model: $J2+SRP$ (simple config), $dt=60$ s
- $Q_{pos}=4.2\times 10^{-4}$ m, $Q_{vel}=1.4\times 10^{-5}$ m/s (calibrated 3x J2+SRP mismatch)
- Angular noise = 10 arcsec, range noise = 100 m
- $\Delta v$ = [1.0, 1.0, 1.0] m/s (single manoeuvre)
- $t^*_{initial\_error}$ = 120 s, $\Delta v_{initial\_error}$ = 0.1 m/s
- 10 seeds per epsilon level (seeds 0–9)
- Config: `configs/config_geo_one_rev_deltaXYZ1_simple.yml`

## Epsilon values tested

| $\epsilon$ (s) | $3\sigma$ window (s) | window as % of $dt$ |
|---:|---:|---:|
| 0.5 | 1.5 | 2.5% |
| 1.0 | 3.0 | 5.0% |
| 2.0 | 6.0 | 10.0% |
| 5.0 | 15.0 | 25.0% |
| 10.0 | 30.0 | 50.0% |
| 20.0 | 60.0 | 100.0% |
| 30.0 | 90.0 | 150.0% |
| 50.0 | 150.0 | 250.0% |
| 100.0 | 300.0 | 500.0% |
| 200.0 | 600.0 | 1000.0% |
| 500.0 | 1500.0 | 2500.0% |

## Aggregate results

| $\epsilon$ (s) | $t^*$ RMS (s) | $t^*$ mean (s) | $t^*$ std-dev (s) | $\|\Delta v\|$ mean (m/s) | $\|\Delta v\|$ std-dev (m/s) | pos RMS (m) | runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 13.12 | -5.89 | 11.72 | 0.0264 | 0.0130 | 49.1 | 21.7 |
| 1.0 | 13.12 | -5.90 | 11.72 | 0.0267 | 0.0133 | 49.1 | 24.0 |
| 2.0 | 13.13 | -5.95 | 11.70 | 0.0280 | 0.0114 | 49.1 | 18.4 |
| 5.0 | 13.09 | -6.24 | 11.51 | 0.0265 | 0.0130 | 49.1 | 14.0 |
| 10.0 | 12.97 | -6.14 | 11.43 | 0.0266 | 0.0130 | 49.1 | 14.1 |
| 20.0 | 12.44 | -5.73 | 11.04 | 0.0266 | 0.0132 | 49.1 | 13.8 |
| 30.0 | 12.10 | -5.58 | 10.73 | 0.0265 | 0.0131 | 49.1 | 14.0 |
| 50.0 | 11.83 | -5.78 | 10.32 | 0.0265 | 0.0132 | 49.1 | 13.7 |
| 100.0 | 11.76 | -6.45 | 9.84 | 0.0278 | 0.0132 | 49.2 | 14.3 |
| 200.0 | 11.82 | -6.27 | 10.02 | 0.0412 | 0.0134 | 49.5 | 14.2 |
| 500.0 | 12.12 | -2.58 | 11.84 | 0.1418 | 0.0139 | 52.2 | 12.8 |

## Per-seed t* errors (s)

| seed | $\epsilon$=0.5 | $\epsilon$=1.0 | $\epsilon$=2.0 | $\epsilon$=5.0 | $\epsilon$=10.0 | $\epsilon$=20.0 | $\epsilon$=30.0 | $\epsilon$=50.0 | $\epsilon$=100.0 | $\epsilon$=200.0 | $\epsilon$=500.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -14.62 | -14.62 | -14.62 | -14.61 | -14.08 | -12.73 | -11.97 | -11.16 | -10.88 | -10.97 | -5.70 |
| 1 | -0.92 | -1.25 | -1.66 | -2.37 | -2.74 | -3.04 | -3.29 | -4.40 | -6.60 | -5.97 | +2.33 |
| 2 | -6.19 | -6.19 | -6.28 | -6.09 | -5.91 | -5.80 | -5.73 | -5.73 | -6.17 | -7.42 | -10.94 |
| 3 | -26.50 | -26.50 | -26.50 | -26.50 | -26.44 | -25.91 | -25.20 | -24.11 | -22.19 | -20.13 | -15.70 |
| 4 | +3.79 | +3.80 | +3.87 | +4.33 | +4.67 | +4.93 | +5.21 | +6.31 | +8.41 | +9.82 | +15.97 |
| 5 | +13.54 | +13.54 | +13.54 | +13.54 | +13.56 | +13.50 | +13.19 | +11.87 | +8.63 | +7.48 | +10.22 |
| 6 | +3.90 | +3.91 | +3.65 | +0.51 | +0.12 | +0.13 | +0.04 | -0.12 | -0.33 | +0.48 | +2.23 |
| 7 | +0.30 | +0.54 | +0.76 | +0.99 | +1.18 | +1.32 | +0.98 | -0.45 | -2.60 | -0.62 | +9.31 |
| 8 | -20.16 | -20.16 | -20.16 | -20.16 | -19.90 | -18.24 | -17.46 | -17.28 | -17.90 | -18.57 | -11.75 |
| 9 | -12.10 | -12.10 | -12.10 | -12.08 | -11.82 | -11.44 | -11.59 | -12.77 | -14.81 | -16.79 | -21.80 |

## Key findings

1. **Epsilon has minimal effect on estimation accuracy at $dt=60$**: $t^*$ RMS varies ~10% (11.76–13.13 s) across a 200x range of $\epsilon$ (0.5 to 100). $\|\Delta v\|$ error and position RMS are constant to the reported precision.

2. **Per-seed $t^*$ values are locked across $\epsilon$**: e.g. seed 3 gives $-26.5$ to $-22.2$ s across the full range.

3. **Gradual improvement at large $\epsilon$**: $t^*$ RMS decreases from 13.1 s ($\epsilon$=0.5) $\to$ 12.1 s ($\epsilon$=30) $\to$ 11.8 s ($\epsilon$=100).

4. **Per-seed convergence at extreme $\epsilon$**: at $\epsilon$=100 ($3\sigma$ = 300 s = 5 dt steps), individual seeds begin drifting more noticeably (e.g. seed 1: $-0.9 \to -6.6$ s, seed 5: $+13.5 \to +8.6$ s).

5. **Runtime scales inversely with $\epsilon$ below $\epsilon \approx 5$**: at $\epsilon$=0.5 and 1.0, sub-stepping is expensive (22–24 s/trial). At $\epsilon \geq 5$ the pulse is wide enough to avoid heavy sub-stepping (14 s/trial, ~40% faster).
   
6. **Interpretation**: at $dt=60$ the timestep resolution is the main constraint on $t^*$ precision. Whether the Gaussian pulse is 1.5 s or 300 s wide, the optimiser cannot resolve $t^*$ more finely than the $dt$ grid allows. This is consistent with the angular-noise sensitivity finding (see `noise_variables_results.md`) where $t^*$ was also insensitive to measurement quality at this $dt$. 

7. **Per-seed drift**: $\epsilon = 5$ is a sweet spot at $dt=60$ with the same accuracy as $\epsilon = 0.5$ but 35% faster runtime. For larger $\epsilon$ values the Gaussian pulse spans many dt steps and we see per-seed estimates begin to drift apart.

8. **$\Delta v$ degradation at extreme $\epsilon$**: at $\epsilon = 200$ the $\|\Delta v\|$ error rises to 0.041 m/s (~1.56x the $\epsilon = 0.5$ baseline), and at $\epsilon = 500$ ($3\sigma$ = 1500 s = 25 dt steps) it jumps to **0.142 m/s** (~5.37x $\epsilon = 0.5$ baseline). The Gaussian pulse becomes too wide and the impulse magnitude is poorly constrained. The optimiser trades $\Delta v$ amplitude for pulse width. $t^*$ RMS remains similar (12.1 s) but per-seed values become less stable. This sets a practical upper bound: $\epsilon \lessapprox 100$ to maintain $\Delta v$ estimate accuracy.
