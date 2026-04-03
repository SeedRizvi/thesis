Biggest contributors from most to least are:
1. Process noise velocity (Q_vel)
2. Measurement Angular noise
3. Measuurement Range noise

Process noise position (Q_pos) - No impact
Process noise position (m):

| Value | \|dv_error\| (m/s) |
|------:|-------------------:|
| 1.0 | 0.1109 |
| 10.0 | 0.1110 |
| 50.0 | 0.1110 |
| 100.0 | 0.1108 |
| 500.0 | 0.1063 |

Process noise velocity (Q_vel) - DOMINANT bottleneck
Process noise velocity (m/s):

| Value | \|dv_error\| (m/s) |
|------:|-------------------:|
| 0.0001 | 0.0454 |
| 0.001 | 0.1108 |
| 0.005 | 0.1893 |
| 0.01 | 0.2451 |
| 0.1 | 0.4101 |

Range noise - mild impact
Range noise (m):

| Value | \|dv_error\| (m/s) |
|------:|-------------------:|
| 1.0 | 0.0860 |
| 10.0 | 0.0778 |
| 50.0 | 0.0913 |
| 100.0 | 0.1108 |
| 500.0 | 0.1048 |

Angular noise - moderate impact
Angular noise (deg) [1, 5, 10, 20, 36 arcsec]:

| Value | \|dv_error\| (m/s) |
|------:|-------------------:|
| 0.00028 | 0.0707 |
| 0.00139 | 0.0849 |
| 0.00278 | 0.1108 |
| 0.00556 | 0.1230 |
| 0.01 | 0.1255 |

Initial state error - zero impact
Initial position error (m):

| Value | \|dv_error\| (m/s) |
|------:|-------------------:|
| 10.0 | 0.1108 |
| 100.0 | 0.1108 |
| 500.0 | 0.1108 |
| 1000.0 | 0.1108 |
| 5000.0 | 0.1108 |
