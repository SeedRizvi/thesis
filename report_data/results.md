# Results Tables - Updated from Sections 4.3.1, 4.3.2, 4.3.3 in Report

---

## 4.3.1 No-manoeuvre baseline (`RIC0`)

**Table 16:** Orbit determination accuracy for the no-manoeuvre case (`RIC0`), 20 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 52.59        | 0.0750          | 20/20 |
| BLS-B  | 485.03       | 0.0364          | 0/20 |
| FGO-B  | 25.90        | 0.0083          | 20/20 |
| FGO-G  | 26.55        | 0.0086          | 20/20 |
| BLS-G  | 282.00       | 0.0210          | 1/20 |
| EKF-G  | 54.70        | 0.0803          | 20/20 |

**Table 17:** Manoeuvre parameter estimates for `RIC0` (no true manoeuvre). Values shown are mean errors across 20 trials; near-zero $\Delta\mathbf{v}$ confirms no false positive.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0038               | 0.0064               | −0.0024              | 0.0215                               | 390.41        |
| BLS-G  | −0.0046              | −0.0033              | 0.0080               | 0.0122                               | 15617.40      |
| EKF-G  | −0.0104              | 0.0128               | −0.0049              | 0.0767                               | 135.49        |

---

## 4.3.2 Standard manoeuvre cases (`RIC1`, `RIC0.5`)

**Table 18:** Orbit determination accuracy for `RIC1` (1.0 m/s per-axis manoeuvre), 20 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 229.90       | 0.2245          | 0/20  |
| BLS-B  | 6368.30      | 0.6548          | 0/20 |
| FGO-B  | 141.06       | 0.1154          | 0/20  |
| FGO-G  | 26.09        | 0.0221          | 20/20 |
| BLS-G  | 307.05       | 0.0432          | 0/20 |
| EKF-G  | 62.54        | 0.1012          | 19/20 |

**Table 19:** Manoeuvre parameter estimation for `RIC1`. Mean errors across 20 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0135               | 0.0078               | 0.0038               | 0.0260                               | 5.81          |
| BLS-G  | −0.0096              | −0.0070              | 0.0134               | 0.0179                               | 30.58         |
| EKF-G  | −0.6301              | −0.6140              | −0.5017              | 1.2423                               | 76.17         |

**Table 20:** Orbit determination accuracy for `RIC0.5` (0.5 m/s per-axis manoeuvre), 20 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 121.97       | 0.1283          | 0/20  |
| BLS-B  | 2991.62      | 0.3176          | 0/20 |
| FGO-B  | 73.89        | 0.0580          | 20/20 |
| FGO-G  | 26.11        | 0.0132          | 20/20 |
| BLS-G  | 306.27       | 0.0328          | 0/20 |
| EKF-G  | 56.92        | 0.0861          | 20/20 |

**Table 21:** Manoeuvre parameter estimation for `RIC0.5`. Mean errors across 20 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0077               | 0.0056               | 0.0013               | 0.0215                               | 11.51         |
| BLS-G  | −0.0026              | −0.0001              | 0.0206               | 0.0208                               | 60.51         |
| EKF-G  | −0.1559              | −0.0774              | 0.0176               | 0.7147                               | 83.23         |

---

## 4.3.3 Challenging single-axis manoeuvre cases (`I0.2`, `C0.2`)

**Table 22:** Orbit determination accuracy for `I0.2` (0.2 m/s in-track manoeuvre), 20 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 62.33        | 0.0808          | 20/20 |
| BLS-B  | 5574.48      | 0.4390          | 0/20 |
| FGO-B  | 31.03        | 0.0167          | 20/20 |
| FGO-G  | 27.31        | 0.0113          | 20/20 |
| BLS-G  | 545.63       | 0.0409          | 0/20 |
| EKF-G  | 54.70        | 0.0811          | 20/20 |

**Table 23:** Manoeuvre parameter estimation for `I0.2`. Mean errors across 20 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0006               | 0.0054               | −0.0008              | 0.0188                               | 237.84        |
| BLS-G  | −0.0092              | −0.0051              | 0.0123               | 0.0215                               | 879.81        |
| EKF-G  | −0.0090              | 0.0033               | 0.0018               | 0.0772                               | 127.83        |

**Table 24:** Orbit determination accuracy for `C0.2` (0.2 m/s cross-track manoeuvre), 20 Monte Carlo trials.

| Method | Pos. MRMS (m) | Vel. MRMS (m/s) | Conv. |
|--------|--------------|-----------------|-------|
| EKF-B  | 58.37        | 0.0798          | 20/20 |
| BLS-B  | 706.82       | 0.0754          | 0/20 |
| FGO-B  | 29.35        | 0.0158          | 20/20 |
| FGO-G  | 30.41        | 0.0113          | 19/20 |
| BLS-G  | 318.61       | 0.0336          | 0/20 |
| EKF-G  | 54.77        | 0.0812          | 20/20 |

**Table 25:** Manoeuvre parameter estimation for `C0.2`. Mean errors across 20 trials.

| Method | $\Delta v_R$ err (m/s) | $\Delta v_I$ err (m/s) | $\Delta v_C$ err (m/s) | $\|\|\Delta\mathbf{v}\|\|$ err (m/s) | $t^*$ RMS (s) |
|--------|----------------------|----------------------|----------------------|--------------------------------------|---------------|
| FGO-G  | 0.0000               | 0.0044               | 0.0009               | 0.0184                               | 173.10        |
| BLS-G  | −0.0072              | −0.0043              | −0.0106              | 0.0167                               | 1623.62       |
| EKF-G  | 0.0003               | 0.0102               | −0.0029              | 0.0906                               | 121.33        |
