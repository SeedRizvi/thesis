import sys
import os
import time
import yaml


def setup_env(orbdethouse_path):
    """
    Setup function to configure the environment for orbDetHOUSE. 

    Args:
        orbdethouse_path (str): Path to the orbDetHOUSE directory

    Returns:
        tuple: (original_dir, orbit_propagator_wrapper) or (None, None) if setup fails
    """
    # Store original directory to restore later
    original_dir = os.getcwd()

    try:
        # Change to orbDetHOUSE directory to handle relative path dependencies
        os.chdir(orbdethouse_path)

        # Add the directory containing the .so file to the system path
        lib_path = os.path.abspath("wsllib")
        sys.path.insert(0, lib_path)

        # Check if the .so file exists in the lib_path
        so_file = os.path.join(lib_path, "orbit_propagator_wrapper.so")
        if not os.path.exists(so_file):
            print(f"orbit_propagator_wrapper.so not found at {so_file}")
            return None, None

        try:
            import orbit_propagator_wrapper # type: ignore
            print("Module imported successfully")
            return original_dir, orbit_propagator_wrapper
        except ModuleNotFoundError as e:
            print(f"Import Error: {e}")
            return None, None

    except Exception as e:
        print(f"Setup error: {e}")
        return None, None

def run_propagation(orbit_propagator_wrapper):
    time.sleep(0.01)
    propagator = orbit_propagator_wrapper.OrbitPropagatorWrapper("yamls/config_orb.yml")
    results = propagator.propagateOrbit()
    # 93600,-5447560.8920494,-4956657.29193027,-2305805.59242059,3491.01298853073,-945.945974206399,-6211.50713325284,
    last_state = results[-1]
    # Extract elements 1:7 from the last state (x, y, z, vx, vy, vz)
    state_values = last_state[1:8].tolist()

    # Read the existing YAML file
    with open('yamls/config_orb_copy.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Update only the initial_state in the initial_orbtial_parameters section
    config['initial_orbtial_parameters']['initial_state'] = state_values

    # Write the modified config back to the file
    with open('yamls/config_orb_copy.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    propagator2 = orbit_propagator_wrapper.OrbitPropagatorWrapper("yamls/config_orb_copy.yml")
    results2 = propagator2.propagateOrbit()

    # Save the results
    headerTraj = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
    resultsFileName = "prop_results_py.csv"
    propagator.saveResults(results, headerTraj, resultsFileName)
    resultsFileName = "prop_results_py2.csv"
    propagator.saveResults(results2, headerTraj, resultsFileName)

    print("Orbit propagation completed successfully")


def main(orbdethouse_path):
    """
    Wrapper function to use orbDetHOUSE as a black box by changing to its directory
    to handle relative path dependencies.

    Args:
        orbdethouse_path (str): Path to the orbDetHOUSE directory
    """
    # Setup environment
    original_dir, orbit_propagator_wrapper = setup_env(orbdethouse_path)

    if original_dir is None or orbit_propagator_wrapper is None:
        print("orbtDetHOUSE setup failed...")
        return

    try:
        run_propagation(orbit_propagator_wrapper)

    finally:
        # Always restore the original directory
        os.chdir(original_dir)
        print(f"Restored original directory: {os.getcwd()}")


if __name__ == "__main__":
    # Default orbDetHOUSE path if not provided
    default_orbdethouse_path = os.path.abspath("orbDetHOUSE")
    main(default_orbdethouse_path)