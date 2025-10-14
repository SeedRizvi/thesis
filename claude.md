# Thesis Project Documentation

## Project Overview
This is a thesis project focused on orbit propagation and simulation.

## Project Structure

### Key Components

#### orbDetHOUSE Directory
- **Purpose**: Orbit propagator used to simulate orbits and generate data for test cases
- **Type**: External library/tool that acts as a black box
- **Location**: `./orbDetHOUSE/`
- **Contains**:
  - Compiled orbit propagator wrapper (`wsllib/orbit_propagator_wrapper.so`)
  - Configuration files (`yamals/config_orb.yml`, `yamals/config_orb_copy.yml`)

#### propagator.py
- **Purpose**: Python wrapper script to interface with orbDetHOUSE
- **Location**: `./propagator.py` (root directory)
- **Key Functions**:
  - `setup_env(orbdethouse_path)`: Configures the environment for orbDetHOUSE by:
    - Changing to the orbDetHOUSE directory to handle relative path dependencies
    - Adding the library path to sys.path
    - Importing the orbit_propagator_wrapper module
  - `run_propagation(orbit_propagator_wrapper)`: Executes orbit propagation using config files
  - `main(orbdethouse_path)`: Main wrapper function that manages directory changes and cleanup
- **Usage**: Treats orbDetHOUSE as a black box, handles all path dependencies automatically
- **Output**: Generates `prop_results_py.csv` with trajectory data (tSec, x, y, z, vx, vy, vz)

## Workflow
1. The propagator.py script changes directory to orbDetHOUSE to handle relative paths
2. Imports the compiled orbit propagator wrapper from wsllib/
3. Runs orbit propagation using YAML configuration files
4. Saves results to CSV format
5. Restores original directory after completion

## Notes
- orbDetHOUSE requires being run from its own directory due to relative path dependencies
- The project uses a compiled C++/C library (`.so` file) wrapped with Python bindings
- Configuration for orbit propagation is managed through YAML files in orbDetHOUSE
