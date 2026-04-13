# Sensitivity Sweep Results - Per-Component RIC Q Model

- Config: `config_geo_one_rev_deltaRIC1.yml`  
- Ground Stations: Rocky Point + Singapore + Tsukuba 
- Model: $J2$ only,
- Settings: $\Delta v_{RIC} = [1,1,1]$ m/s, $\varepsilon = 0.5$ s, $t^*_{initerror} = 120$ s, $\Delta v_{initerror} = 0.1$ m/s
- Trials: randomiser seeds [0, 4] (5 total) for each test.  
- Base Q (calibrated at $dt=60$ s, 3x dynamics mismatch):  
$\quad Q_{pos} = [3.13\times10^{-4},\ 2.20\times10^{-4},\ 1.37\times10^{-4}]$ m (R, I, C)  
$\quad Q_{vel} = [1.04\times10^{-4},\ 7.26\times10^{-5},\ 4.56\times10^{-5}]$ m/s (R, I, C)  
- Baseline noise: angular 10 arcsec, range 15 m.

---

## Time-step ($dt$) Testing

- Swept variable: $dt \in \{5, 10, 30, 60, 120\}$ s
- Q scaled with dt: $Q_{pos} \propto dt^2$, $Q_{vel} \propto dt$ (confirmed exact by calibration: vel~$dt^{1.00}$, pos~$dt^{2.00}$)
- Baseline: $Q_{pos} = [3.13\times10^{-4},\ 2.20\times10^{-4},\ 1.37\times10^{-4}]$ m, $Q_{vel} = [1.04\times10^{-4},\ 7.26\times10^{-5},\ 4.56\times10^{-5}]$ m/s, angular noise 10 arcsec, range noise 15 m

| $dt$ (s) | $N$ | pos RMS mean (m) | pos RMS std (m) | vel RMS mean (m/s) | vel RMS std (m/s) | $\|\Delta v_{err}\|$ (m/s) | $t^*$ RMS (s) | $t^*$ mean (s) |
|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 5 | 19008 | 16.9 | 1.2 | 0.01409 | 0.00100 | 0.0363 | 3.32 | +2.62 |
| 10 | 9504 | 23.1 | 1.9 | 0.02205 | 0.00420 | 0.0502 | 3.81 | −2.73 |
| 30 | 3168 | 1109 | 2151 | 0.1190 | 0.1769 | 0.1500 | 107.67 | −51.61 |
| 60 | 1584 | 389 | 688 | 0.04638 | 0.04430 | 0.0454 | 27.32 | +14.29 |
| 120 | 792 | 60.3 | 10.9 | 0.02995 | 0.01790 | 0.0411 | 8.44 | +6.71 |

Per-seed pos RMS (m):

| $dt$ (s) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|-------:|-------:|-------:|-------:|-------:|-------:|
| 5 | 17.4 | 14.7 | 18.1 | 16.6 | 17.5 |
| 10 | 21.1 | 25.7 | 22.3 | 21.6 | 24.9 |
| 30 | 31.7 | 43.0 | **5411** | 27.9 | 31.9 |
| 60 | **1765** | 62.5 | 45.5 | 34.8 | 37.6 |
| 120 | 57.3 | 77.8 | 66.8 | 46.5 | 53.2 |

**Observations:**
- $dt=5$ s gives the best overall performance: pos 17 m, $t^*$ 3.3 s RMS, stable across seeds.
- $dt=10$ s also robust: pos 23 m, $t^*$ 3.8 s RMS. Closely matches Zhang et al. ($t^*$ RMSE = 2.86 s at $dt=10$ s).
- $dt=30$ and $dt=60$ have occasional single-seed divergence (seeds 2 and 0 respectively) dominating the mean. Typical seeds converge to 28–62 m.
- $dt=120$ s is stable but coarser; $t^*$ resolution degrades to ~8 s as the Gaussian pulse ($\varepsilon=0.5$ s) is irresolvable within 120 s steps.
- Non-monotonic behaviour is explained by sub-stepping resolution: at $dt \gg \varepsilon$, $t^*$ estimation becomes unreliable, leading to sporadic divergence rather than systematic bias.

---

## $Q_{pos}$ Testing

- Swept variable: $Q_{pos}$ multiplier; 1x reference = $[3.13\times10^{-4},\ 2.20\times10^{-4},\ 1.37\times10^{-4}]$ m (R, I, C) (output of `calibrate_q_ric.py`)
- Baseline (fixed): $dt=60$ s, $Q_{vel} = [1.04\times10^{-4},\ 7.26\times10^{-5},\ 4.56\times10^{-5}]$ m/s, angular noise 10 arcsec, range noise 15 m

| $Q_{pos}$ mult | pos RMS mean (m) | pos RMS std (m) | $\|\Delta v_{err}\|$ (m/s) | $t^*$ RMS (s) | $t^*$ mean (s) |
|-------:|-------:|-------:|-------:|-------:|-------:|
| 0.01x | 3940 | 5453 | 0.1059 | 85.6 | +47.2 |
| 0.03x | 2309 | 5063 | 0.0666 | 81.1 | +34.8 |
| 0.1x | 1000 | 2135 | 0.0780 | 54.5 | +22.8 |
| 0.3x | 508 | 874 | 0.0720 | 27.3 | +13.7 |
| 1x *(calibrated)* | 389 | 769 | 0.0454 | 27.3 | +14.3 |
| **3x** | **43.6** | **11.3** | 0.1373 | **7.0** | −0.89 |
| 10x | 44.2 | 11.0 | 0.2229 | 7.0 | −0.86 |
| 30x | 43.8 | 11.2 | 0.0448 | 7.0 | −1.09 |
| 100x | 43.5 | 11.5 | 0.0513 | 7.0 | −1.07 |

- Above 3x the solution saturates at the measurement-noise floor - $Q_{pos}$ has no further effect.
- Below calibrated value (< 1x) divergence dominates the mean due to over-reliance on dynamics.
- **Conclusion**: $Q_{pos}$ must exceed ~3x the calibrated value. Beyond that it is irrelevant.

---

## $Q_{vel}$ Testing

- Swept variable: $Q_{vel}$ multiplier; 1x reference = $[1.04\times10^{-5},\ 7.26\times10^{-6},\ 4.56\times10^{-6}]$ m/s (R, I, C) (output of `calibrate_q_ric.py`)
- Baseline (fixed): $dt=60$ s, $Q_{pos} = [3.13\times10^{-4},\ 2.20\times10^{-4},\ 1.37\times10^{-4}]$ m, angular noise 10 arcsec, range noise 15 m

| $Q_{vel}$ mult | pos RMS mean (m) | pos RMS std (m) | $\|\Delta v_{err}\|$ (m/s) | $t^*$ RMS (s) | $t^*$ mean (s) |
|-------:|-------:|-------:|-------:|-------:|-------:|
| 0.1x | 148 | 193 | 0.0096 | 4.3 | +2.8 |
| **0.3x** | **27.9** | **7.1** | **0.0107** | 6.1 | +1.3 |
| 1x *(calibrated)* | 35.6 | 9.6 | 0.0158 | 6.6 | +0.6 |
| 3x | 37.3 | 9.9 | 0.0218 | 6.7 | −0.25 |
| 10x | 389 | 769 | 0.0454 | 27.3 | +14.3 |
| 30x | 126 | 157 | 0.0890 | 26.7 | +15.2 |
| 100x | 65.7 | 15.7 | 0.2604 | 8.7 | +4.7 |
| 300x | 1420 | 2824 | 0.3798 | 115.1 | +79.9 |
| 1000x | 1243 | 2599 | 0.4655 | 133.1 | +92.3 |

- $Q_{vel}$ is the **dominant tuning parameter**.
- Sweet spot at 0.3-1x calibrated: pos 28–36 m, $\|\Delta v_{err}\| \approx 0.01$–0.02 m/s.
- The calibrated value (1x) performs well. Above 10x divergence begins; at 300x and beyond the dynamics are completely unconstrained.
- **Conclusion**: use $Q_{vel}$ near the calibrated value (0.3–1x) for best results.

---

## Measurement Angular Noise

- Swept variable: $\sigma_{angle} \in \{1, 2, 5, 10, 20, 36, 72\}$ arcsec
- Baseline: $dt=60$ s, $Q_{pos} = [3.13\times10^{-4},\ 2.20\times10^{-4},\ 1.37\times10^{-4}]$ m, $Q_{vel} = [1.04\times10^{-4},\ 7.26\times10^{-5},\ 4.56\times10^{-5}]$ m/s, range noise 15 m

| $\sigma_{angle}$ (arcsec) | pos RMS mean (m) | pos RMS std (m) | $\|\Delta v_{err}\|$ (m/s) | $t^*$ RMS (s) |
|-------:|-------:|-------:|-------:|-------:|
| 1 | 32.75 | 27.1 | 0.0291 | 3.95 |
| 2 | 110.31 | 177.1 | 0.3261 | 214.2 |
| 5 | 47.75 | 15.6 | 0.0345 | 3.78 |
| 10 *(base)* | 43.54 | 11.4 | 0.0835 | 295.6 |
| 20 | 370.53 | 727.3 | 0.1351 | 26.3 |
| 36 | 57.07 | 28.5 | 0.3954 | 3.76 |
| 72 | 68.69 | 54.0 | 0.0375 | 3.69 |

- **I forgot to gather $t^*$ std-dev data, and running these tests takes some time, so I have not done it at this time.**
- Results are highly variable with 5 seeds - large std indicates occasional divergence. Need more seeds but this is quite time-consuming for now.
- No clean monotonic trend; non-monotonicity (e.g. 36" better than 20"); randomness at play.
- $t^*$ spikes at 2" and 10" are single-seed outlier divergences, not physical.
- **Conclusion**: angular noise has hard-to-assess impact at GEO range (~42 000 km).Range noise dominates position accuracy. More seeds needed for reliable statistics.

---

## Measurement Range Noise

- Swept variable: $\sigma_{range} \in \{1, 5, 10, 25, 50, 100, 200, 500\}$ m
- Baseline: $dt=60$ s, $Q_{pos}$ and $Q_{vel}$ at 1x calibrated, angular noise 10 arcsec

| $\sigma_{range}$ (m) | pos RMS mean (m) | pos RMS std (m) | $\|\Delta v_{err}\|$ (m/s) | $t^*$ RMS (s) | $t^*$ mean (s) |
|-------:|-------:|-------:|-------:|-------:|-------:|
| 1 | 7.45 | 3.22 | 0.2022 | 81.2 | +36.9 |
| 5 | 20.69 | 3.73 | 0.0493 | 3.04 | −0.77 |
| **10** | **32.61** | 7.89 | **0.0311** | **4.92** | −0.74 |
| 15 *(base)* | 37.25 | 9.91 | 0.0218 | 6.65 | -0.27 |
| 25 | 510 | 998 | 0.0665 | 29.2 | +11.2 |
| 50 | 5483 | 6434 | 0.0803 | 86.0 | +53.8 |
| 100 | 4912 | 6504 | 0.4280 | 138.5 | +83.6 |
| 200 | 11385 | 10226 | 0.1647 | 117.6 | +87.0 |
| 500 | 38155 | 42749 | 0.2164 | 123.8 | +84.5 |

- Range noise is the **dominant factor** for position accuracy.
- Clean breakpoint between 10–25 m: below 10 m well-converged; at >=25 m divergence dominates.
- At 1 m: pos is best but $t^*$ and $\|\Delta v_{err}\|$ degrade — over-constraining ranging creates tension with the Gaussian manoeuvre model.
- Sweet spot: **5–15 m** range noise for good pos and manoeuvre estimation simultaneously.
- $t^*$ error tracks pos error (both collapse at high range noise).
- **Conclusion**: range noise is the primary practical accuracy bottleneck; target sensor $\sigma_{range} \leq 15$ m.
