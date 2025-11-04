import os
import sys
import numpy as np
import subprocess

def check_orbdethouse():
    orbdethouse_path = os.path.abspath("orbDetHOUSE")
    if not os.path.exists(orbdethouse_path):
        print("ERROR: orbDetHOUSE not found!")
        print(f"Expected at: {orbdethouse_path}")
        print("\nPlease clone and compile orbDetHOUSE:")
        print("  git clone -b python_wrapper_propagator https://github.com/YangDrYang/orbDetHOUSE.git")
        print("  cd orbDetHOUSE")
        print("  make -f makefile_py_wsl")
        return False
    return True

def run_propagator():
    print("\n" + "="*70)
    print("STEP 1: Propagating GEO Orbit with orbDetHOUSE")
    print("="*70)
    
    try:
        from propagator import OrbitPropagator
    except ImportError:
        print("ERROR: propagator.py not found!")
        print("Please ensure propagator.py is in the current directory")
        return None
    
    prop = OrbitPropagator("orbDetHOUSE")
    csv_path = prop.propagate("./configs/config_geo_test.yml", output_file="geo_truth.csv")
    
    print(f"✓ Propagation complete: {csv_path}")
    return csv_path

def generate_measurements(truth_csv):
    print("\n" + "="*70)
    print("STEP 2: Generating Measurements from Ground Stations")
    print("="*70)
    
    result = subprocess.run([
        sys.executable, 
        "generate_fgo_data.py", 
        truth_csv, 
        "geo_example.npz"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        return False
    
    print("✓ Measurements generated: geo_example.npz")
    return True

def run_fgo():
    print("\n" + "="*70)
    print("STEP 3: Running Factor Graph Optimization")
    print("="*70)
    print("")
    
    result = subprocess.run([
        sys.executable,
        "Orbit_FGO.py"
    ])
    
    if result.returncode != 0:
        print("ERROR: FGO failed with return code", result.returncode)
        return False
    
    print("\n✓ FGO optimization complete")
    return True

def show_summary():
    print("\n" + "="*70)
    print("GEO FGO TEST SUMMARY")
    print("="*70)
    
    if not os.path.exists("geo_example.npz"):
        print("ERROR: Measurement file not found")
        return
    
    data = np.load("geo_example.npz", allow_pickle=True)
    truth = data['truth']
    
    n_timesteps = len(truth)
    dt = float(data['dt'])
    time_span = (n_timesteps - 1) * dt
    
    mean_radius = np.mean(np.linalg.norm(truth[:, :3], axis=1))
    mean_velocity = np.mean(np.linalg.norm(truth[:, 3:], axis=1))
    
    GE = 3.986004418e14
    period = 2 * np.pi * np.sqrt((mean_radius**3) / GE)
    
    print(f"\nOrbit Characteristics:")
    print(f"  Mean radius: {mean_radius/1000:.1f} km")
    print(f"  Mean velocity: {mean_velocity:.1f} m/s")
    print(f"  Orbital period: {period/3600:.2f} hours")
    print(f"  Time span: {time_span/3600:.2f} hours")
    print(f"  Timesteps: {n_timesteps}")
    
    if abs(mean_radius - 42164000) < 1000000:
        print(f"  ✓ Orbit type: GEO (geostationary)")
    else:
        print(f"  ⚠ WARNING: Not a GEO orbit!")
    
    if os.path.exists("fg_geo_example_res.npz"):
        results = np.load("fg_geo_example_res.npz")
        fg_res = results['fg_res']
        truth = results['truth']
        
        errors = fg_res - truth
        pos_errors = np.linalg.norm(errors[:, :3], axis=1)
        vel_errors = np.linalg.norm(errors[:, 3:], axis=1)
        
        print(f"\nFGO Results:")
        print(f"  Position RMS: {np.sqrt(np.mean(pos_errors**2)):.2f} m")
        print(f"  Position Max: {np.max(pos_errors):.2f} m")
        print(f"  Velocity RMS: {np.sqrt(np.mean(vel_errors**2)):.4f} m/s")
        print(f"  Velocity Max: {np.max(vel_errors):.4f} m/s")
        
        if np.sqrt(np.mean(pos_errors**2)) < 100:
            print(f"  ✓ Excellent performance for GEO!")
        elif np.sqrt(np.mean(pos_errors**2)) < 1000:
            print(f"  ✓ Good performance for GEO")
        else:
            print(f"  ⚠ Performance below expectations")
    
    print("\nGenerated Files:")
    if os.path.exists("out/geo_truth.csv"):
        print("  ✓ out/geo_truth.csv (truth trajectory)")
    if os.path.exists("geo_example.npz"):
        print("  ✓ geo_example.npz (measurements)")
    if os.path.exists("fg_geo_example_res.npz"):
        print("  ✓ fg_geo_example_res.npz (FGO results)")
    if os.path.exists("geo_example_full_fgo_results.png"):
        print("  ✓ geo_example_full_fgo_results.png (plots)")

def main():
    print("="*70)
    print("GEO SATELLITE FGO TEST WORKFLOW")
    print("="*70)
    print("\nThis script will:")
    print("  1. Propagate a GEO orbit using orbDetHOUSE")
    print("  2. Generate angle measurements from 3 ground stations")
    print("  3. Run Factor Graph Optimization")
    print("  4. Display results")
    
    print("\n" + "="*70)
    print("IMPORTANT: Dynamics Model Matching")
    print("="*70)
    print("For FGO to work, the truth propagator (orbDetHOUSE) must use")
    print("the SAME dynamics model as the FGO propagator:")
    print("  - 2-body gravity")
    print("  - J2 perturbation (degree 2, order 2)")
    print("  - NO third-body, SRP, tides, or relativity")
    print("\nThe config_geo_test.yml has been set up with simplified dynamics.")
    print("="*70)
    
    if not check_orbdethouse():
        return
    
    truth_csv = run_propagator()
    if truth_csv is None:
        return
    
    if not generate_measurements(truth_csv):
        return
    
    if not run_fgo():
        return
    
    show_summary()
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
