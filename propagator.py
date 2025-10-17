import sys
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt


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
        
        # C++ saves to out/out_prop + filename (concatenated, not directory)
        output_path = os.path.abspath(os.path.join("out", f"out_prop{output_filename}"))
        
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"Propagation completed but output not found at: {output_path}\n"
                f"Results length: {len(results)}"
            )
        
        return output_path
    
    def propagate_from_state(self, config_file, output_file="results_continued.csv"):
        """
        Run propagation, then start a new propagation from the final state.
        
        Args:
            config_file: Template config to use
            output_file: Output filename (C++ will prepend out_prop to it)
            
        Returns:
            Tuple of (first_csv_path, continued_csv_path)
        """
        config_abs = os.path.abspath(config_file)
        
        if not os.path.exists(config_abs):
            raise FileNotFoundError(f"Config not found: {config_abs}")
        
        propagator = self.wrapper.OrbitPropagatorWrapper(config_abs)
        results = propagator.propagateOrbit()
        
        last_state = results[-1]
        state_values = last_state[1:8].tolist()
        
        with open(config_abs, 'r') as f:
            config = yaml.safe_load(f)
        
        config['initial_orbtial_parameters']['initial_state'] = state_values
        
        temp_config = config_abs.replace('.yml', '_temp.yml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        propagator2 = self.wrapper.OrbitPropagatorWrapper(temp_config)
        results2 = propagator2.propagateOrbit()
        
        header = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
        
        output_filename = os.path.basename(output_file)
        first_filename = output_filename.replace('.csv', '_first.csv')
        
        propagator.saveResults(results, header, first_filename)
        propagator2.saveResults(results2, header, output_filename)
        
        # C++ saves to out/out_prop + filename (concatenated)
        first_output = os.path.abspath(os.path.join("out", f"out_prop{first_filename}"))
        output_path = os.path.abspath(os.path.join("out", f"out_prop{output_filename}"))
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Propagation completed but output not found at: {output_path}")
        
        return first_output, output_path


def plot_orbit_3d(csv_file, output_file=None, title="Orbital Trajectory"):
    """
    Plot 3D orbital trajectory from propagation results.

    Args:
        csv_file: Path to CSV file with orbit data
        output_file: Path to save plot. If None, displays plot
        title: Plot title
    """
    df = pd.read_csv(csv_file)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(df['x'].values, df['y'].values, df['z'].values, linewidth=0.8)
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], 
               color='green', s=50, label='Start')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], 
               color='red', s=50, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    max_range = max(df[['x', 'y', 'z']].abs().max())
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    prop = OrbitPropagator("orbDetHOUSE")
    
    csv1 = prop.propagate("configs/config_orb.yml", output_file="results1.csv")
    plot_orbit_3d(csv1, output_file="orbit1.png")
    
    csv1, csv2 = prop.propagate_from_state("configs/config_orb.yml", 
                                           output_file="results2.csv")
    plot_orbit_3d(csv2, output_file="orbit2.png")