import sys
import os


def main(orbdethouse_path):
    """
    Wrapper function to use orbDetHOUSE as a black box by changing to its directory
    to handle relative path dependencies.

    Args:
        orbdethouse_path (str): Path to the orbDetHOUSE directory
    """
    # Store original directory to restore later
    original_dir = os.getcwd()

    try:
        # Change to orbDetHOUSE directory to handle relative path dependencies
        os.chdir(orbdethouse_path)
        print(f"Changed to directory: {os.getcwd()}")

        # Add the directory containing the .so file to the system path
        lib_path = os.path.abspath("wsllib")
        print(f"Library path: {lib_path}")
        sys.path.insert(0, lib_path)

        # Print the sys.path for debugging
        print("sys.path:", sys.path)

        # Check if the .so file exists in the lib_path
        so_file = os.path.join(lib_path, "orbit_propagator_wrapper.so")
        if os.path.exists(so_file):
            print(f"Found orbit_propagator_wrapper.so at {so_file}")
        else:
            print(f"orbit_propagator_wrapper.so not found at {so_file}")
            return

        try:
            import orbit_propagator_wrapper
            print("Module imported successfully")
        except ModuleNotFoundError as e:
            print(f"Error: {e}")
            return

        # Assuming you have a class named OrbitPropagator in your C++ code
        propagator = orbit_propagator_wrapper.OrbitPropagatorWrapper("yamls/config_orb.yml")
        results = propagator.propagateOrbit()

        # Define the headers and results file name
        headerTraj = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
        resultsFileName = "prop_results_py.csv"
        # Save the results
        propagator.saveResults(results, headerTraj, resultsFileName)

        print("Orbit propagation completed successfully")

    finally:
        # Always restore the original directory
        os.chdir(original_dir)
        print(f"Restored original directory: {os.getcwd()}")


if __name__ == "__main__":
    # Default orbDetHOUSE path if not provided
    default_orbdethouse_path = os.path.abspath("orbDetHOUSE")
    main(default_orbdethouse_path)