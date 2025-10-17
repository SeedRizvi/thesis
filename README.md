# Thesis
My amazing thesis project where I propagate orbits and do data fusion and tracking stuff!

## Project Structure

```
~/thesis/
├── auxdata/              # Copied from orbDetHOUSE (required)
├── configs/              # Your configuration files (refer to orbDetHOUSE)
├── out/                  # Propagation outputs
├── orbDetHOUSE/          # Propagator dependency
├── propagator.py         # Orbit propagator wrapper
├── main.py               # Where all the magic happens
└── README.md
```

## Quick Setup

### 1. Clone orbDetHOUSE
[orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator) is required but omitted from this repo due to size.

```bash
cd ~/thesis/
git clone -b python_wrapper_propagator https://github.com/YangDrYang/orbDetHOUSE.git
```

### 2. Compile orbDetHOUSE
```bash
cd orbDetHOUSE/
make -f makefile_py_wsl clean
make -f makefile_py_wsl
python3 pyscripts/test_orbit_propagator_wrapper_wsl.py  # Test installation
cd ..
```

### 3. Setup Dependencies
```bash
python setup_orbdethouse.py
```

### 4. Install Python Dependencies
```bash
pip install "numpy<2" pandas matplotlib pyyaml
```

## Dependencies

- [orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator) - C++ orbit propagator
- Python 3.8+
- numpy<2 (for matplotlib compatibility)
- pandas
- matplotlib
- pyyaml
- pybind11 (for orbDetHOUSE compilation)

## Notes

- orbDetHOUSE outputs use concatenated naming: `out/out_prop{filename}`
- NumPy must be <2.0 for matplotlib compatibility
- auxdata contains ephemeris and gravity model data required by C++