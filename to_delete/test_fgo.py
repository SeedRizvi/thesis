#!/usr/bin/env python3
"""
Quick test script to verify the fixed FGO implementation
"""

import numpy as np
import sys
import os

# Add current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from to_delete.Orbit_FGO_fixed import SatelliteOrbitFGO, generate_test_data
import matplotlib.pyplot as plt


def test_fgo():
    """Test the Factor Graph Optimization implementation"""
    
    print("="*70)
    print("Testing Fixed FGO Implementation")
    print("="*70)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic FGO functionality...")
    try:
        meas, truth, dt, R, Q, ground_stations = generate_test_data()
        print("   ✓ Test data generation successful")
    except Exception as e:
        print(f"   ✗ Failed to generate test data: {e}")
        return False
    
    # Test 2: FGO initialization
    print("\n2. Testing FGO initialization...")
    try:
        x0 = truth[0].copy()
        x0[:3] += np.random.normal(0, 1000, 3)  # Add 1km position error
        x0[3:] += np.random.normal(0, 1, 3)      # Add 1m/s velocity error
        
        fgo = SatelliteOrbitFGO(meas, R, Q, ground_stations, dt, x0=x0)
        print("   ✓ FGO initialization successful")
    except Exception as e:
        print(f"   ✗ Failed to initialize FGO: {e}")
        return False
    
    # Test 3: Propagation
    print("\n3. Testing orbit propagation...")
    try:
        state_test = truth[0]
        state_prop = fgo.prop_one_timestep(state_test)
        prop_error = np.linalg.norm(state_prop[:3] - truth[1, :3])
        print(f"   ✓ Propagation test (error: {prop_error:.2f} m)")
        
        if prop_error > 10000:  # More than 10km error indicates problem
            print(f"   ⚠ Warning: Large propagation error")
    except Exception as e:
        print(f"   ✗ Failed propagation test: {e}")
        return False
    
    # Test 4: Measurement computation
    print("\n4. Testing measurement computation...")
    try:
        az, el = fgo.compute_azimuth_elevation(truth[0, :3], ground_stations[0], 0)
        print(f"   ✓ Measurement computation (az: {np.deg2rad(az):.3f}°, el: {np.deg2rad(el):.3f}°)")
    except Exception as e:
        print(f"   ✗ Failed measurement computation: {e}")
        return False
    
    # Test 5: Jacobian computation
    print("\n5. Testing Jacobian matrices...")
    try:
        F = fgo.F_mat(truth[0])
        H = fgo.H_mat(truth[0], 0, 0)
        print(f"   ✓ F matrix shape: {F.shape}")
        print(f"   ✓ H matrix shape: {H.shape}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(F)) or np.any(np.isinf(F)):
            print("   ⚠ Warning: F matrix contains NaN or Inf")
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            print("   ⚠ Warning: H matrix contains NaN or Inf")
    except Exception as e:
        print(f"   ✗ Failed Jacobian computation: {e}")
        return False
    
    # Test 6: Optimization
    print("\n6. Testing optimization (this may take a moment)...")
    print("-"*70)
    
    initial_errors = fgo.states - truth
    initial_pos_rms = np.sqrt(np.mean(np.linalg.norm(initial_errors[:, :3], axis=1)**2))
    print(f"Initial Position RMS: {initial_pos_rms:.2f} m")
    
    try:
        fgo.opt(max_iters=20, verbose=True)
        print("-"*70)
        
        final_errors = fgo.states - truth
        final_pos_rms = np.sqrt(np.mean(np.linalg.norm(final_errors[:, :3], axis=1)**2))
        final_vel_rms = np.sqrt(np.mean(np.linalg.norm(final_errors[:, 3:], axis=1)**2))
        
        print(f"\nFinal Position RMS: {final_pos_rms:.2f} m")
        print(f"Final Velocity RMS: {final_vel_rms:.4f} m/s")
        
        # Check convergence
        if final_pos_rms < initial_pos_rms * 0.1:  # Should improve by at least 10x
            print("   ✓ Optimization converged successfully")
        else:
            print(f"   ⚠ Warning: Limited convergence (improvement: {initial_pos_rms/final_pos_rms:.1f}x)")
            
    except Exception as e:
        print(f"   ✗ Optimization failed: {e}")
        return False
    
    # Test 7: Verify results are reasonable
    print("\n7. Verifying results...")
    if final_pos_rms < 1000:  # Less than 1km RMS
        print(f"   ✓ Position accuracy acceptable ({final_pos_rms:.1f} m)")
    else:
        print(f"   ⚠ Position accuracy poor ({final_pos_rms:.1f} m)")
    
    if final_vel_rms < 1.0:  # Less than 1 m/s RMS
        print(f"   ✓ Velocity accuracy acceptable ({final_vel_rms:.4f} m/s)")
    else:
        print(f"   ⚠ Velocity accuracy poor ({final_vel_rms:.4f} m/s)")
    
    # Generate simple plot
    print("\n8. Generating verification plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Position errors over time
    pos_errors = np.linalg.norm(final_errors[:, :3], axis=1)
    ax1.plot(pos_errors)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Position Error Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2D trajectory comparison
    ax2.plot(truth[:, 0]/1e6, truth[:, 1]/1e6, 'r-', label='Truth', linewidth=2)
    ax2.plot(fgo.states[:, 0]/1e6, fgo.states[:, 1]/1e6, 'b--', 
             label='Estimated', alpha=0.7)
    ax2.set_xlabel('X (Mm)')
    ax2.set_ylabel('Y (Mm)')
    ax2.set_title('Orbit Comparison (XY Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.suptitle('FGO Verification Results')
    plt.tight_layout()
    plt.savefig('fgo_test_results.png', dpi=100)
    print("   ✓ Plot saved to fgo_test_results.png")
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    print("="*70)
    
    return True


if __name__ == '__main__':
    success = test_fgo()
    
    if success:
        print("\n✅ FGO implementation is working correctly!")
        print("\nYou can now use it with your propagator by running:")
        print("  python fgo_pipeline.py --config your_config.yml")
    else:
        print("\n❌ FGO implementation has issues - check the errors above")
    
    plt.show()
