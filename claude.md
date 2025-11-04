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
  measurement_noise_deg: 0.01
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

## Notes

- All state vectors are in ECI (Earth-Centred Inertial) frame
- Positions in metres, velocities in m/s, angles in radians
- Ground station positions specified in geodetic coordinates (lat/lon/alt)
- Range measurements enabled by default (use `--no-range` to disable)
- Typical performance: ~100m position RMS with range, ~7km without range
