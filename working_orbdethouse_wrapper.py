#!/usr/bin/env python3.10
"""
orbDetHOUSE wrapper for running orbit propagation from thesis directory.
Place this file in ~/thesis/

Usage:
    python3.10 orbit_propagator.py                    # Use default config
    python3.10 orbit_propagator.py my_config.yml      # Use custom config
"""

import sys
import os
from pathlib import Path

# Set up paths
THESIS_DIR = Path(__file__).parent
ORBDETHOUSE_DIR = THESIS_DIR / "orbDetHOUSE"
WSLLIB_DIR = ORBDETHOUSE_DIR / "wsllib"

# Add to Python path
sys.path.insert(0, str(WSLLIB_DIR))

# Import the module
try:
    import orbit_propagator_wrapper
except ImportError as e:
    print(f"Error: Failed to import orbit_propagator_wrapper: {e}")
    print(f"Make sure you've compiled the wrapper:")
    print(f"  cd {ORBDETHOUSE_DIR}")
    print(f"  make -f makefile_py_wsl clean")
    print(f"  make -f makefile_py_wsl")
    sys.exit(1)


def propagate_orbit(config_file=None, output_file=None):
    """
    Run orbit propagation using orbDetHOUSE.
    
    Parameters:
    -----------
    config_file : str, optional
        Path to configuration YAML file. If None, uses default config.
    output_file : str, optional
        Name for output CSV file. If None, uses default name.
    
    Returns:
    --------
    results : propagation results object
    """
    
    # Default values
    if config_file is None:
        config_file = "yamls/config_orb.yml"
    if output_file is None:
        output_file = "prop_results_py.csv"
    
    # Convert to Path for easier handling
    config_path = Path(config_file)
    
    # Change to orbDetHOUSE directory (needed for relative paths in config)
    original_cwd = os.getcwd()
    
    try:
        # If config is in thesis dir (absolute or relative to thesis)
        if config_path.is_absolute():
            config_to_use = str(config_path)
        elif (THESIS_DIR / config_path).exists():
            config_to_use = str(THESIS_DIR / config_path)
        else:
            # Config is relative to orbDetHOUSE directory
            os.chdir(ORBDETHOUSE_DIR)
            config_to_use = str(config_path)
        
        # If we haven't changed directory yet, do it now for aux files
        if os.getcwd() == original_cwd:
            os.chdir(ORBDETHOUSE_DIR)
        
        print(f"Configuration file: {config_to_use}")
        print(f"Output file: {output_file}")
        print("-" * 50)
        
        # Initialize propagator
        print("Initializing propagator...")
        propagator = orbit_propagator_wrapper.OrbitPropagatorWrapper(config_to_use)
        
        # Run propagation
        print("Running orbit propagation...")
        results = propagator.propagateOrbit()
        
        # Define headers for output file
        headerTraj = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
        
        # Save results
        print(f"Saving results to: {output_file}")
        propagator.saveResults(results, headerTraj, output_file)
        
        print("✓ Propagation completed successfully!")
        
        # Show file location
        output_path = Path.cwd() / output_file
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"\nOutput saved to: {output_path}")
            print(f"File size: {file_size:,} bytes")
            
            # Show first few lines
            print("\nFirst 5 lines of output:")
            with open(output_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.strip()}")
                    else:
                        print("  ...")
                        break
        
        return results
        
    except Exception as e:
        print(f"Error during propagation: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)


def main():
    """
    Main function that handles command-line arguments.
    """
    
    print("\n" + "="*60)
    print("orbDetHOUSE Orbit Propagator")
    print("="*60 + "\n")
    
    # Parse command-line arguments
    config_file = None
    output_file = None
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"Using custom config: {config_file}")
    else:
        print("Using default config: yamls/config_orb.yml")
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        print(f"Output file: {output_file}")
    else:
        output_file = "prop_results_py.csv"
        print(f"Using default output: {output_file}")
    
    print()
    
    # Run propagation
    results = propagate_orbit(config_file, output_file)
    
    if results is not None:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print("✓ Orbit propagation completed")
        print("✓ Results saved to CSV file")
        print("\nYou can now:")
        print("  1. View the results in the CSV file")
        print("  2. Plot the trajectory using the analysis scripts")
        print("  3. Run with different configs for comparison")
    
    return results


if __name__ == "__main__":
    main()