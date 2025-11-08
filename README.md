# Thesis
My FGO-based Impulsive Manoeuvre Estimation thesis project.

## Credits
- [orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator) for providing the propagator used for obtaining orbit data.

- [C. Taylor and J. Gross](https://github.com/cntaylor/factorGraph2DsatelliteExample), 2D FGO satellite orbit model, which was adapted into 3D for this project in [Orbit_FGO.py](Orbit_FGO.py).

## Project Structure

```
~/thesis/
├── auxdata/              # Copied from orbDetHOUSE (required)
├── configs/              # Configuration files for orbit propagation
├── out/                  # Propagation outputs
├── plots/                # Generated plots
├── orbDetHOUSE/          # C++ propagator dependency
├── propagator.py         # Orbit propagator wrapper
├── fgo_pipeline.py       # Factor Graph Optimisation pipeline
├── Orbit_FGO.py          # FGO implementation
├── setup_orbdethouse.py  # Setup script
└── README.md
```

## Installation Instructions

**Note**: These installation instructions are for WSL/Linux machines only.

### Initial Setup (Common for Both Options)

**1. Clone this repository**
```bash
git clone https://github.com/SeedRizvi/thesis.git
cd thesis
```

**2. Clone orbDetHOUSE**

[orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator) is required but omitted from this repo due to size.

```bash
git clone -b python_wrapper_propagator https://github.com/YangDrYang/orbDetHOUSE.git
```

**3. Compile orbDetHOUSE**
```bash
cd orbDetHOUSE/
make -f makefile_py_wsl clean
make -f makefile_py_wsl
python3 pyscripts/test_orbit_propagator_wrapper_wsl.py  # Test installation
cd ..
```

---

### Option 1: Automated Setup (Recommended)

**4. Run Setup Script**
```bash
python setup_orbdethouse.py
```

This script will automatically:
- Copy `auxdata/` from orbDetHOUSE to project root
- Verify the compiled Python wrapper exists
- Confirm setup completion

**5. Install Python Dependencies**
```bash
pip install "numpy<2" pandas matplotlib pyyaml scipy
```

**6. Run Factor Graph Optimisation**
```bash
python fgo_pipeline.py
```

---

### Option 2: Manual Installation

**4. Copy Required Data Files**
```bash
cp -r orbDetHOUSE/auxdata/ ./auxdata/
```

**5. Verify Compiled Wrapper**
```bash
ls orbDetHOUSE/wsllib/orbit_propagator_wrapper.so
```

If the file doesn't exist, return to step 3.

**6. Install Python Dependencies**
```bash
pip install "numpy<2" pandas matplotlib pyyaml scipy
```

**7. Run Factor Graph Optimisation**
```bash
python fgo_pipeline.py
```

## Dependencies

- [orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator) - C++ orbit propagator
- Python 3.8+
- numpy<2 (for matplotlib compatibility)
- pandas
- matplotlib
- pyyaml
- pybind11 (for orbDetHOUSE compilation)

## Usage

Run the Factor Graph Optimisation pipeline with default settings:
```bash
python fgo_pipeline.py
```

### Command-Line Options

All FGO parameters (noise levels, errors, etc.) are configured via the config file, see `configs/config_geo_realistic.yml` as an example. 

The following CLI options are available:

- `--config PATH` - Path to configuration file (default: `configs/config_geo_realistic.yml`)
- `--no-range` - Disable range measurements, use angular-only
- `--max-iters N` - Override maximum optimisation iterations from config
- `--quiet` - Suppress verbose output

**Examples:**
```bash
# Use a custom configuration file
python fgo_pipeline.py --config configs/your_config_file.yml

# Disable range measurements
python fgo_pipeline.py --no-range

# Override max iterations and run quietly
python fgo_pipeline.py --max-iters 100 --quiet
```

<!-- ## Known Issues

### Segmentation Faults with Delta-V Maneuvers

**Issue:** Running delta-v maneuvers (e.g., `python3 fgo_pipeline.py --delta_v 0 0 50`) causes segmentation faults when creating multiple propagator instances (which is required to avoid discontinuities).

**Root Cause:** The orbDetHOUSE C++ wrapper lacks proper resource cleanup (no destructor to free JPL ephemeris memory) and maintains corrupted global state after the first propagation.

**Solution:** The `propagator.py` implementation now runs each propagation in a separate Python process, for fresh module loading and preventing state corruption. This is now the default propagator used by `fgo_pipeline.py`. -->