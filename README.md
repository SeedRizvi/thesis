# Thesis
My amazing thesis project. Currently requiries the installation of 

## Usage
[orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator) is required but ommitted from this repo due to its large size. Ensure it exists in the following format for correct use.

```
~/thesis/
├── dsgp4/
├── sgp4/
├── orbDetHOUSE/
├── orbtdethouse_wrapper.py
├── README.md
└── working_orbdethouse_wrapper.py
```

### Setup
```
cd orbDetHOUSE/
make -f makefile_py_wsl clean
make -f makefile_py_wsl
cd ..
```

## Dependencies
- [orbDetHOUSE](https://github.com/YangDrYang/orbDetHOUSE/tree/python_wrapper_propagator)
- matplotlib
- numpy
- pandas

#### Not currently in use, but here just in case
- [dSGP4](https://github.com/esa/dSGP4/)
- [SGP4](https://github.com/aholinch/sgp4)
- pytorch

## Limitations
- dSGP4 does not support orbital periods > 225 minutes, i.e. GEO (out of my control)
