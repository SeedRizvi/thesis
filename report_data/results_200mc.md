# Results Tables - 200 Monte Carlo Trials

---

## 4.3.1 No-manoeuvre baseline (`RIC0`)

**Table 16:** Orbit determination accuracy for the no-manoeuvre case (`RIC0`), 200 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 53.20        | 0.0775          | 200/200 |
| BLS-B  | 484.59       | 0.0363          | 0/200 |
| FGO-B  | 25.92        | 0.0085          | 200/200 |
| FGO-G  | 27.06        | 0.0088          | 199/200 |
| BLS-G  | 314.47       | 0.0233          | 5/200 |
| EKF-G  | 55.07        | 0.0824          | 200/200 |

**Table 17:** Manoeuvre parameter estimates for `RIC0` (no true manoeuvre). Values shown are mean errors across 200 trials; near-zero $\Delta\mathbf{v}$ confirms no false positive.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0005               | 0.0001               | −0.0003              | 0.0224                               | 516.39        |
| BLS-G  | −0.0029              | −0.0013              | −0.0270              | 0.0511                               | 15866.50      |
| EKF-G  | 0.0096               | −0.0026              | 0.0002               | 0.0721                               | 126.58        |

---

## 4.3.2 Standard manoeuvre cases (`RIC1`, `RIC0.5`)

**Table 18:** Orbit determination accuracy for `RIC1` (1.0 m/s per-axis manoeuvre), 200 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 230.05       | 0.2209          | 0/200 |
| BLS-B  | 6368.77      | 0.6548          | 0/200 |
| FGO-B  | 141.36       | 0.1156          | 0/200 |
| FGO-G  | 26.10        | 0.0230          | 200/200 |
| BLS-G  | 306.48       | 0.0434          | 0/200 |
| EKF-G  | 59.53        | 0.1053          | 194/200 |

**Table 19:** Manoeuvre parameter estimation for `RIC1`. Mean errors across 200 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0118               | 0.0036               | 0.0043               | 0.0236                               | 4.30          |
| BLS-G  | −0.0095              | −0.0070              | 0.0135               | 0.0179                               | 31.05         |
| EKF-G  | −0.1736              | −0.2598              | −0.1444              | 0.8752                               | 179.40        |

**Table 20:** Orbit determination accuracy for `RIC0.5` (0.5 m/s per-axis manoeuvre), 200 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 123.95       | 0.1301          | 0/200 |
| BLS-B  | 2991.98      | 0.3176          | 0/200 |
| FGO-B  | 74.14        | 0.0583          | 200/200 |
| FGO-G  | 26.12        | 0.0138          | 200/200 |
| BLS-G  | 305.94       | 0.0329          | 0/200 |
| EKF-G  | 56.42        | 0.0887          | 200/200 |

**Table 21:** Manoeuvre parameter estimation for `RIC0.5`. Mean errors across 200 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0056               | 0.0011               | 0.0015               | 0.0206                               | 8.70          |
| BLS-G  | −0.0026              | −0.0001              | 0.0206               | 0.0208                               | 61.53         |
| EKF-G  | −0.0496              | −0.0834              | −0.0011              | 0.5916                               | 91.52         |

---

## 4.3.3 Challenging single-axis manoeuvre cases (`I0.2`, `C0.2`)

**Table 22:** Orbit determination accuracy for `I0.2` (0.2 m/s in-track manoeuvre), 200 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 63.57        | 0.0832          | 200/200 |
| BLS-B  | 5573.88      | 0.4389          | 0/200 |
| FGO-B  | 31.30        | 0.0170          | 200/200 |
| FGO-G  | 27.37        | 0.0112          | 200/200 |
| BLS-G  | 485.52       | 0.0370          | 0/200 |
| EKF-G  | 55.14        | 0.0829          | 200/200 |

**Table 23:** Manoeuvre parameter estimation for `I0.2`. Mean errors across 200 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | −0.0014              | 0.0007               | −0.0006              | 0.0199                               | 207.28        |
| BLS-G  | −0.0063              | −0.0048              | 0.0136               | 0.0214                               | 960.84        |
| EKF-G  | 0.0095               | 0.0013               | 0.0027               | 0.0871                               | 118.91        |

**Table 24:** Orbit determination accuracy for `C0.2` (0.2 m/s cross-track manoeuvre), 200 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 59.57        | 0.0822          | 200/200 |
| BLS-B  | 706.50       | 0.0754          | 0/200 |
| FGO-B  | 29.34        | 0.0160          | 200/200 |
| FGO-G  | 28.55        | 0.0107          | 198/200 |
| BLS-G  | 338.10       | 0.0359          | 0/200 |
| EKF-G  | 55.11        | 0.0828          | 200/200 |

**Table 25:** Manoeuvre parameter estimation for `C0.2`. Mean errors across 200 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0001               | −0.0002              | 0.0005               | 0.0199                               | 150.92        |
| BLS-G  | −0.0077              | −0.0062              | −0.0039              | 0.0279                               | 2087.99       |
| EKF-G  | 0.0127               | −0.0033              | −0.0005              | 0.0907                               | 116.40        |

---

## 4.3.4 Short arc manoeuvre case (`RIC0.5_short`, 3.6 hours)
* Tested over a 3.6 hour arc (1.2 hours pre-manoeuvre + 2.4 hours post-manoeuvre), rather than standard 27.6 hour arc used in ALL other cases.

**Table 26:** Orbit determination accuracy for `RIC0.5_short` (0.5 m/s per-axis manoeuvre, 3.6 h arc), 200 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 278.70       | 0.3541          | 0/200 |
| BLS-B  | 1499.18      | 0.5640          | 0/200 |
| FGO-B  | 241.02       | 0.1909          | 0/200 |
| BLS-G  | 22.96        | 0.0305          | 200/200 |
| FGO-G  | 34.09        | 0.0331          | 200/200 |
| EKF-G  | 87.56        | 0.2497          | 168/200 |

**Table 27:** Manoeuvre parameter estimation for `RIC0.5_short`. Mean errors across 200 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0058               | 0.0026               | 0.0025               | 0.0253                               | 7.62          |
| BLS-G  | 0.0024               | 0.0025               | 0.0032               | 0.0119                               | 5.14          |
| EKF-G  | −0.0393              | −0.0367              | −0.0306              | 0.5447                               | 82.90         |
