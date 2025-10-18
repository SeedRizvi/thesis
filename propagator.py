import sys
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class OrbitPropagator:
    """
    Clean wrapper for orbDetHOUSE using direct .so import.
    
    Requirements:
    - auxdata/ directory in root (copy from orbDetHOUSE/auxdata/)
    
    C++ behavior:
    - Saves output files to out/out_prop{filename} (concatenates prefix with filename)
    - Example: filename "results.csv" → saved as "out/out_propresults.csv"
    """
    def __init__(self, orbdethouse_path="orbDetHOUSE"):
        self.orbdethouse_path = os.path.abspath(orbdethouse_path)
        so_path = os.path.join(self.orbdethouse_path, "wsllib")
        
        if so_path not in sys.path:
            sys.path.insert(0, so_path)
        
        try:
            import orbit_propagator_wrapper # type: ignore
            self.wrapper = orbit_propagator_wrapper
        except ImportError as e:
            raise ImportError(
                f"Failed to import orbit_propagator_wrapper from {so_path}. "
                f"Run 'make -f makefile_py_wsl' in orbDetHOUSE first. Error: {e}"
            )
        
        if not os.path.exists("auxdata"):
            raise FileNotFoundError(
                "auxdata/ directory not found in current directory. "
                f"Copy it from {self.orbdethouse_path}/auxdata/ to your project root."
            )
        
        os.makedirs("out", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
    
    def propagate(self, config_file, output_file="results.csv"):
        """
        Run orbit propagation.
        
        Args:
            config_file: Path to YAML config (absolute or relative to current dir)
            output_file: Output filename (C++ will prepend out_prop to it)
            
        Returns:
            Absolute path to output CSV
        """
        config_abs = os.path.abspath(config_file)
        
        if not os.path.exists(config_abs):
            raise FileNotFoundError(f"Config not found: {config_abs}")
        
        propagator = self.wrapper.OrbitPropagatorWrapper(config_abs)
        results = propagator.propagateOrbit()
        
        header = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
        
        output_filename = os.path.basename(output_file) if isinstance(output_file, str) else output_file
        
        propagator.saveResults(results, header, output_filename)
        
        output_path = os.path.abspath(os.path.join("out", f"out_prop{output_filename}"))
        
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"Propagation completed but output not found at: {output_path}\n"
                f"Results length: {len(results)}"
            )
        
        return output_path
    
    def propagate_from_state(self, config_file, delta_v, output_file="results_continued.csv"):
        """
        Run propagation, apply delta-v to final state, then propagate again.
        Generates 3 plots: first propagation, second propagation, and combined.
        
        Args:
            config_file: Template config to use
            delta_v: Velocity vector [dvx, dvy, dvz] in m/s to apply to final state
            output_file: Output filename for second propagation
            
        Returns:
            Tuple of (first_csv_path, second_csv_path, combined_csv_path)
        """
        config_abs = os.path.abspath(config_file)
        
        if not os.path.exists(config_abs):
            raise FileNotFoundError(f"Config not found: {config_abs}")
        
        propagator = self.wrapper.OrbitPropagatorWrapper(config_abs)
        results = propagator.propagateOrbit()
        
        last_state = results[-1]
        state_values = last_state[1:8].tolist()
        
        state_values[3] += delta_v[0]
        state_values[4] += delta_v[1]
        state_values[5] += delta_v[2]
        
        with open(config_abs, 'r') as f:
            config = yaml.safe_load(f)
        
        config['initial_orbtial_parameters']['initial_state'] = state_values
        config['scenario_parameters']['MJD_start'] = config['scenario_parameters']['MJD_end']
        config['scenario_parameters']['MJD_end'] = config['scenario_parameters']['MJD_start'] + 0.5
        
        temp_config = config_abs.replace('.yml', '_temp.yml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        propagator2 = self.wrapper.OrbitPropagatorWrapper(temp_config)
        results2 = propagator2.propagateOrbit()
        
        header = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
        
        output_filename = os.path.basename(output_file)
        first_filename = output_filename.replace('.csv', '_pre_delta.csv')
        second_filename = output_filename.replace('.csv', '_post_delta.csv')
        combined_filename = output_filename.replace('.csv', '_combined.csv')
        
        propagator.saveResults(results, header, first_filename)
        propagator2.saveResults(results2, header, second_filename)
        
        first_output = os.path.abspath(os.path.join("out", f"{first_filename}"))
        second_output = os.path.abspath(os.path.join("out", f"{second_filename}"))
        combined_output = os.path.abspath(os.path.join("out", f"{combined_filename}"))
        
        if not os.path.exists(second_output):
            raise RuntimeError(f"Propagation completed but output not found at: {second_output}")
        
        df1 = pd.read_csv(first_output)
        df2 = pd.read_csv(second_output)
        
        time_offset = df1['tSec'].iloc[-1]
        df2_shifted = df2.copy()
        df2_shifted['tSec'] = df2_shifted['tSec'] + time_offset
        
        df_combined = pd.concat([df1, df2_shifted.iloc[1:]], ignore_index=True)
        df_combined.to_csv(combined_output, index=False)
        
        plot_orbit_3d(first_output, 
                     output_file="plots/pre_manouevre.png",
                     title="First Propagation (Before Delta-V)")
        plot_orbit_3d(second_output, 
                     output_file="plots/post_manouevre.png",
                     title="Second Propagation (After Delta-V)")
        plot_orbit_3d(combined_output, 
                     output_file="plots/combined.png",
                     title="Combined Propagation (t0 → t1 → t2)")
        
        return first_output, second_output, combined_output


def plot_orbit_3d(csv_file, output_file=None, title="Orbital Trajectory"):
    """
    Plot 3D orbital trajectory from propagation results.

    Args:
        csv_file: Path to CSV file with orbit data
        output_file: Path to save plot. If None, displays plot
        title: Plot title
    """
    df = pd.read_csv(csv_file)
    
    x_km = df['x'].values / 1000.0
    y_km = df['y'].values / 1000.0
    z_km = df['z'].values / 1000.0
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x_km, y_km, z_km, linewidth=0.8)
    ax.scatter(x_km[0], y_km[0], z_km[0], 
               color='green', s=50, label='Start')
    ax.scatter(x_km[-1], y_km[-1], z_km[-1], 
               color='red', s=50, label='End')
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    max_range = max([abs(x_km).max(), abs(y_km).max(), abs(z_km).max()])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    prop = OrbitPropagator("orbDetHOUSE")
    
    # csv1 = prop.propagate("configs/config_orb_long.yml", output_file="results1.csv")
    # plot_orbit_3d(csv1, output_file="plots/orbit1.png")

    # csv2 = prop.propagate("configs/config_orb_short.yml", output_file="results2.csv")
    # plot_orbit_3d(csv2, output_file="plots/orbit2.png")
    
    delta_v = [100.0, 50.0, -300.0] # m/s
    csv1, csv2, csv_combined = prop.propagate_from_state("configs/config_orb.yml", 
                                                          delta_v=delta_v,
                                                          output_file="results.csv")