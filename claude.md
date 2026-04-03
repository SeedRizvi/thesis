# Thesis Project Documentation

## Project Overview
This thesis project implements Factor Graph Optimisation (FGO) for satellite orbit determination using ground-based measurements (azimuth, elevation, and range). The system combines high-fidelity orbit propagation from orbDetHOUSE with a custom FGO implementation that supports both angular-only and angular+range measurements.

## Project Structure

```
~/thesis/
├── fgo_pipeline.py           # Main FGO pipeline script
├── Orbit_FGO.py              # Core FGO implementation
├── propagator.py             # orbDetHOUSE Python wrapper
├── setup_orbdethouse.py      # Setup script for dependencies
├── configs/                  # Configuration files
├── out/                      # Propagation output CSVs
├── plots/                    # Generated plots
├── auxdata/                  # Gravity models & ephemeris data
└── orbDetHOUSE/              # C++ orbit propagator library
```

## Core Components

### fgo_pipeline.py
**Purpose**: Main integration script that orchestrates the complete FGO workflow

**Key Functions**:
- `load_propagator_output(csv_path)`: Loads orbit propagation results from CSV
- `simulate_measurements(states, times, ground_stations, ...)`: Simulates ground station measurements with noise
  - Computes azimuth/elevation/range from satellite to ground stations
  - Adds configurable measurement noise
  - Supports both angular-only and angular+range modes
- `compute_measurements_full(r_sat_eci, station_llh, t, ...)`: Computes measurements using coordinate transformations (ECI → ECEF → ENU)
- `load_config_parameters(config_path)`: Loads FGO parameters and ground station network from YAML config
- `run_fgo_with_propagator(config_path, ...)`: Complete pipeline execution
  1. Runs orbit propagation via propagator.py
  2. Loads propagation results
  3. Simulates measurements from ground stations
  4. Sets up process/measurement noise models
  5. Generates initial state with errors
  6. Runs FGO optimisation
  7. Computes and reports final errors
- `plot_fgo_results(results, save_path)`: Generates comprehensive visualisation with 7 subplots:
  - 3D trajectory (with correct aspect ratio)
  - Position/velocity component errors
  - Total position/velocity errors
  - Error distribution histogram
  - Summary statistics

**Command-line Arguments**:
- `--config PATH`: Path to config file (default: `configs/config_geo_realistic.yml`)
- `--no-range`: Disable range measurements (range enabled by default)
- `--max-iters N`: Maximum optimisation iterations (overrides config)
- `--quiet`: Suppress verbose output

**Note**: All FGO parameters (measurement noise, range noise, process noise, initial errors) are configured via the YAML config file under the `fgo_parameters` section. Only the above runtime options are available via command-line.

**Output**:
- Saves plots to `./plots/fgo_results_full.png` or `./plots/fgo_results_angular.png`
- Saves numerical results to `./out/fgo_results_full.npz` or `./out/fgo_results_angular.npz`

### Orbit_FGO.py
**Purpose**: Core Factor Graph Optimisation implementation for satellite orbit determination

**Main Class**: `SatelliteOrbitFGO`
- Implements sparse factor graph optimisation for satellite states
- Supports J2 perturbations in dynamics model
- Uses Levenberg-Marquardt-style optimisation with line search
- Auto-detects measurement type (2 or 3 measurements per station)

**Key Methods**:
- `__init__(meas, R, Q, ground_stations, dt, x0, use_range)`: Initialises FGO problem
  - Auto-detects whether data contains range measurements
  - Sets up measurement/process noise covariance matrices
  - Initialises state trajectory using dynamics propagation
- `prop_one_timestep(state)`: Propagates state using 2-body + J2 dynamics
- `compute_measurements(r_sat_eci, station_llh, t)`: Forward measurement model (ECI → Az/El/Range)
- `H_mat(state, station_idx, t)`: Computes measurement Jacobian via finite differences
- `F_mat(state)`: Computes dynamics Jacobian via finite differences
- `create_L()`: Builds sparse Jacobian matrix for entire factor graph
- `create_y()`: Builds residual vector (measurements + dynamics)
- `opt(max_iters, verbose)`: Runs optimisation loop with adaptive regularisation

**Features**:
- Sparse matrix implementation for efficiency (handles hundreds of timesteps)
- Adaptive regularisation (λ adjustment)
- Line search for step size selection
- Handles angle wrapping for azimuth measurements
- Configurable measurement types (angular-only or angular+range)

### propagator.py
**Purpose**: Python wrapper for orbDetHOUSE C++ orbit propagator

**Main Class**: `OrbitPropagator`
- `__init__(orbdethouse_path)`: Initialises wrapper
  - Adds orbDetHOUSE library path to Python path
  - Imports compiled C++ wrapper (`orbit_propagator_wrapper.so`)
  - Verifies `auxdata/` directory exists
  - Creates `out/` and `plots/` directories
- `propagate(config_file, output_file)`: Runs orbit propagation
  - Loads YAML configuration
  - Executes C++ propagator
  - Saves results to CSV in `out/` directory
  - Returns absolute path to output file
- `propagate_from_state(config_file, delta_v, output_file)`: Two-stage propagation
  - Runs initial propagation
  - Applies delta-v manoeuvre to final state
  - Continues propagation from new state
  - Generates comparison plots

**Key Function**: `plot_orbit_3d(csv_file, output_file, title)`
- Plots 3D orbital trajectory from CSV data
- Marks start (green) and end (red) points
- Saves to `plots/` directory

**Output**: CSV files in `out/` with columns: `tSec, x, y, z, vx, vy, vz` (ECI frame)

## Directory Structure

### configs/
Contains YAML configuration files for orbit propagation and FGO

**Structure**:
```yaml
scenario_parameters:
  time_step: 60
  MJD_start: ...
  MJD_end: ...

initial_orbital_parameters:
  initial_state: [x, y, z, vx, vy, vz]  # ECI coordinates

propagator_truth_settings:
  earth_gravity_model_order: 20
  third_body_attraction: true
  solar_radiation_pressure: true
  # ... other force model settings

ground_stations:
  - name: "New York"
    latitude: 40.7128
    longitude: -74.0060
    altitude: 0.0
  # ... more stations

fgo_parameters:
  use_range: true
  measurement_noise_deg: 0.00278
  range_noise_m: 100.0
  process_noise_position: 100.0
  process_noise_velocity: 0.01
  initial_position_error: 1000.0
  initial_velocity_error: 1.0
  max_iterations: 50
```

### out/
Output directory for CSV files generated by orbit propagation
- Format: `tSec, x, y, z, vx, vy, vz` (ECI frame, SI units)
- Also stores `.npz` files with FGO results

### plots/
Output directory for generated visualisation plots
- FGO result plots: `fgo_results_full.png`, `fgo_results_angular.png`
- Orbit trajectory plots from propagator

### auxdata/
Required auxiliary data files for orbDetHOUSE (copied from orbDetHOUSE during setup)
- `GGM03S.txt`: Gravity model coefficients
- `linux_p1550p2650.440`: JPL planetary ephemeris
- `cod21587.erp`: Earth rotation parameters

### orbDetHOUSE/
External C++ orbit propagator library
- **Type**: Black-box dependency
- **Contains**:
  - `wsllib/orbit_propagator_wrapper.so`: Compiled Python bindings
  - Internal configuration and auxiliary files
- **Note**: Requires compilation before use (see README.md)

## Complete Workflow

1. **Setup** (one-time):
   ```bash
   python setup_orbdethouse.py
   ```
   - Copies `auxdata/` from orbDetHOUSE to project root
   - Verifies compiled wrapper exists

2. **Run FGO Pipeline**:
   ```bash
   python fgo_pipeline.py --config configs/your_config.yml
   ```

3. **Pipeline Execution**:
   - Load configuration parameters and ground station network
   - Run orbit propagation (via propagator.py → orbDetHOUSE)
   - Load propagated "truth" trajectory from CSV
   - Simulate noisy measurements from ground stations
   - Generate initial state estimate with errors
   - Run FGO to optimise state trajectory
   - Compute position/velocity errors
   - Generate visualisation plots
   - Save results to `out/` and `plots/`

## Key Features

- **Measurement Flexibility**: Supports both angular-only (Az/El) and angular+range measurements
- **High-Fidelity Dynamics**: Uses orbDetHOUSE for truth propagation with configurable force models
- **Realistic Measurement Model**: Includes coordinate transformations (ECI → ECEF → ENU) and Earth rotation
- **Sparse Optimisation**: Efficient factor graph implementation using scipy.sparse
- **Configurable Noise Models**: Separate control of measurement and process noise
- **Comprehensive Visualisation**: 7-subplot analysis including trajectory, errors, and statistics
- **Command-line Interface**: Full control via arguments or config file

## Gaussian Impulse Approximation for Manoeuvre Estimation

### Overview
Implements the Gaussian Impulse Approximation (Zhang et al., IEEE TAES 2025) to estimate impulsive manoeuvre delta-v within the FGO. Instead of splitting propagation at the manoeuvre boundary, the impulse is modelled as a smooth Gaussian function, making dynamics differentiable and allowing the FGO to estimate delta-v components as optimisation parameters.

### Key Equations
- **Gaussian impulse** (Eq. 17): `g(t, t*, ε) = (1/(ε√2π)) * exp(-(t-t*)²/(2ε²))`
- **Modified dynamics** (Eq. 21): `ẋ = f(t,x) + B * Σ Δv_j * g(t, t_j*, ε)` where B=[0₃;I₃]
- The Gaussian acceleration only enters velocity derivatives

### Implementation Details
- **Module-level functions**: `gaussian_impulse()`, `gaussian_impulse_dt_star()` in `Orbit_FGO.py`
- **Constructor**: Accepts `manoeuvres` list (each with `delta_v` and `t_star`) and `epsilon` parameter
- **Time-aware dynamics**: `orbital_dynamics(state, t=None)` adds Gaussian acceleration when `t` is provided
- **Adaptive sub-stepping**: Near manoeuvre epochs (within 3σ of t*), `prop_one_timestep` subdivides the 60s step into smaller steps (`epsilon/5`) to resolve the narrow Gaussian pulse
- **Augmented Jacobian**: `create_L` includes 3 extra columns per manoeuvre for delta-v parameters, computed via finite differences in `F_man_mat`
- **Augmented state vector**: `add_delta`, `update_state`, `create_y` handle the extended vector `[states | man_params]`
- **Backward compatibility**: When `manoeuvres=None`, all code paths are identical to the original (no Gaussian terms, no extra columns)

### Current Status
- **Delta-v estimation**: Working. t* is fixed (known), 3 delta-v components estimated per manoeuvre.
- **t* estimation**: Not yet implemented. The derivative `gaussian_impulse_dt_star` is available but unused. This is the next major feature.

### Pipeline Integration
- `fgo_pipeline.py` captures `first_csv_path` from propagation to determine `t_star = df_pre['tSec'].iloc[-1]`
- Initial delta-v guess: `dv_true + N(0, dv_initial_error)`
- `use_gaussian_estimation` flag (CLI: `--no-gaussian`) toggles between Gaussian estimation and legacy split-propagation
- Results include `dv_true`, `dv_estimated`, `dv_error` for post-processing

### Config Parameters (under `manoeuvre_parameters`)
```yaml
manoeuvre_parameters:
  delta_v: [0.0, 0.0, 1.0]     # True delta-v [X, Y, Z] (m/s)
  pm_duration: 0.85             # Post-manoeuvre propagation duration (days)
  epsilon: 0.5                  # Gaussian shaping parameter (seconds)
  dv_initial_error: 0.1         # Initial delta-v guess error std dev (m/s)
```

## Issues Faced and Resolutions

### 1. False Positive Delta-V in X/Y Components
**Symptom**: When applying a Z-only manoeuvre (e.g., [0,0,1] m/s), the estimator consistently reported non-zero X and Y delta-v estimates (~0.2 m/s), while Z was estimated accurately.

**Investigation**: Projected the false positive into the RTN (Radial-Tangential-Normal) orbital frame. Found the error was consistently **along-track** regardless of orbital position — it appeared in ECI-X or ECI-Y depending on where in the orbit the manoeuvre occurred (because the along-track direction rotates with the orbit).

**Root cause**: **Weak observability of in-plane delta-v**. The process noise Q_vel was too generous (0.01 m/s), allowing the optimizer to absorb delta-v errors by distributing small per-state velocity corrections across many post-manoeuvre timesteps. Cross-track (Z) delta-v was accurately estimated because a plane change creates a distinctive pattern across all subsequent measurements that process noise corrections cannot mimic.

**Resolution**: Set Q_vel to match actual dynamics model uncertainty. A sensitivity sweep (`sweep_sensitivity.py`) confirmed Q_vel was the dominant bottleneck — reducing it from 0.01 to 0.001 cut the delta-v error from ~0.24 to ~0.11 m/s.

### 2. Determining Appropriate Q_vel
**Method**: Measured per-step dynamics model error by propagating truth states through the FGO dynamics (two-body+J2) and comparing to the high-fidelity truth trajectory. This isolates the pure model error with no measurement or estimation effects.

**Results**:
- Full-fidelity truth vs J2-only FGO: mean 0.000347 m/s, max 0.000465 m/s per step
- Simplified truth (J2+SRP) vs J2-only FGO: ~2.7e-6 m/s per step (from SRP: 4.56e-8 m/s² × 60s)

**Recommendation**: Q_vel = 0.001 m/s provides ~3× margin over measured full-fidelity mismatch. This is the current default for testing.

### 3. Delta-V Error Floor
The delta-v estimation error (~0.11 m/s with Q_vel=0.001) is a **fixed floor independent of manoeuvre magnitude**. A 50 m/s manoeuvre has the same absolute error as a 1 m/s manoeuvre. This floor is set by measurement noise (10 arcsec angles, 100m range), ground station geometry, and observation cadence — not by the manoeuvre estimation itself.

## Sensitivity Analysis Results
From `sweep_sensitivity.py` (varying one parameter at a time from baseline):

| Parameter | Impact on delta-v error | Notes |
|-----------|------------------------|-------|
| Q_vel (process noise velocity) | **DOMINANT** — 0.0001→0.045, 0.01→0.245 | Must match dynamics model accuracy |
| Angular noise | Moderate — 1 arcsec→0.07, 36 arcsec→0.13 | 10 arcsec is realistic for testing |
| Range noise | Mild — relatively flat across 1-500m | Helps orbit accuracy more than delta-v |
| Q_pos (process noise position) | Negligible — flat at ~0.11 across 1-500m | Delta-v is a velocity quantity |
| Initial state error | Zero impact — flat at ~0.11 across 10-5000m | Optimizer fully corrects this |

## Diagnostic Tools (Not For Production Use)

### Delta-V Prior Regularisation (`dv_prior_sigma`)
Adds a Gaussian prior penalty `(1/σ) * (dv_prior - dv_current)` to the cost function, discouraging delta-v estimates far from the initial guess. Implemented as extra rows in `create_L` and `create_y`.

**Finding**: Reduced error from ~0.27 to ~0.20 m/s but hit a floor. The "right" σ depends on knowing how far the initial guess is from truth — not available operationally. **Requires prior knowledge of the manoeuvre to tune effectively.**

### Adaptive Q Matrix Tightening (`q_tightening_factor`, `q_tightening_window`)
Multiplies `S_Q_inv` by a factor for timesteps within a window of t*, forcing tighter dynamics constraints near the manoeuvre.

**Finding**: The window size dominates (more constrained timesteps = better), the factor barely matters beyond a threshold. With window=20, error dropped to ~0.10 m/s. Confirmed that the number of unconstrained post-manoeuvre states is what allows the optimizer to absorb delta-v errors.

**Both tools require knowledge of the manoeuvre epoch and characteristics to optimise, so they are not suitable for operational use.** They were valuable for diagnosing the observability issue and confirming that Q_vel was the true lever.

## Notes

- All state vectors are in ECI (Earth-Centred Inertial) frame
- Positions in metres, velocities in m/s, angles in radians
- Ground station positions specified in geodetic coordinates (lat/lon/alt)
- Range measurements enabled by default (use `--no-range` to disable)
- Typical performance: ~100m position RMS with range, ~7km without range
- Supervisor direction: simplified dynamics (J2 + SRP), no third-body perturbations needed
