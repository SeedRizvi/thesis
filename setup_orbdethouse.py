#!/usr/bin/env python3
"""
Setup script for clean orbDetHOUSE integration.
Copies required data files to project root.
"""

import os
import shutil

def setup_orbdethouse_dependencies(orbdethouse_path="orbDetHOUSE"):
    """Copy auxdata directory from orbDetHOUSE to current directory."""
    
    orbdethouse_abs = os.path.abspath(orbdethouse_path)
    source_auxdata = os.path.join(orbdethouse_abs, "auxdata")
    dest_auxdata = "auxdata"
    
    if not os.path.exists(source_auxdata):
        print(f"ERROR: auxdata not found at {source_auxdata}")
        print(f"Make sure orbDetHOUSE is properly set up.")
        return False
    
    if os.path.exists(dest_auxdata):
        response = input(f"{dest_auxdata}/ already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping auxdata copy.")
            return True
        shutil.rmtree(dest_auxdata)
    
    print(f"Copying {source_auxdata} -> {dest_auxdata}/")
    shutil.copytree(source_auxdata, dest_auxdata)
    print("✓ auxdata copied successfully")
    
    # Check for .so file
    so_path = os.path.join(orbdethouse_abs, "wsllib", "orbit_propagator_wrapper.so")
    if not os.path.exists(so_path):
        print("\nWARNING: orbit_propagator_wrapper.so not found!")
        print(f"Run the following commands:")
        print(f"  cd {orbdethouse_path}")
        print(f"  make -f makefile_py_wsl clean")
        print(f"  make -f makefile_py_wsl")
        print(f"  cd ..")
        return False
    
    print("✓ orbit_propagator_wrapper.so found")
    print("\n" + "="*60)
    print("Setup complete! You can now use:")
    print("  from propagator import OrbitPropagator, plot_orbit_3d")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = setup_orbdethouse_dependencies()
    exit(0 if success else 1)