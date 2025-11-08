import sys
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import json


class OrbitPropagator:
    """
    Split-script wrapper for orbDetHOUSE that avoids segmentation faults.

    This version runs each propagation as a completely separate Python process
    to ensure clean module loading each time.

    Requirements:
    - auxdata/ directory in root (copy from orbDetHOUSE/auxdata/)

    C++ behavior:
    - Saves output files to out/out_prop{filename} (concatenates prefix with filename)
    - Example: filename "results.csv" → saved as "out/out_propresults.csv"
    """
    def __init__(self, orbdethouse_path="orbDetHOUSE"):
        self.orbdethouse_path = os.path.abspath(orbdethouse_path)

        if not os.path.exists("auxdata"):
            raise FileNotFoundError(
                "auxdata/ directory not found in current directory. "
                f"Copy it from {self.orbdethouse_path}/auxdata/ to your project root."
            )

        os.makedirs("out", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def propagate(self, config_file, output_file="results.csv"):
        """
        Run orbit propagation in a completely separate Python process.

        Args:
            config_file: Path to YAML config (absolute or relative to current dir)
            output_file: Output filename (C++ will prepend out_prop to it)

        Returns:
            Absolute path to output CSV
        """
        config_abs = os.path.abspath(config_file)

        if not os.path.exists(config_abs):
            raise FileNotFoundError(f"Config not found: {config_abs}")

        # Create standalone script for single propagation
        script = f"""
import sys
import os
sys.path.insert(0, '{os.path.join(self.orbdethouse_path, "wsllib")}')
import orbit_propagator_wrapper

config_file = '{config_abs}'
output_file = '{output_file}'

propagator = orbit_propagator_wrapper.OrbitPropagatorWrapper(config_file)
results = propagator.propagateOrbit()

header = ["tSec", "x", "y", "z", "vx", "vy", "vz"]
output_filename = os.path.basename(output_file)
propagator.saveResults(results, header, output_filename)

print("SUCCESS")
"""

        # Run script in completely separate process
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            if result.returncode == -11:
                raise RuntimeError(f"Segmentation fault during propagation")
            else:
                raise RuntimeError(f"Propagation failed: {result.stderr}")

        output_path = os.path.abspath(os.path.join("out", f"{os.path.basename(output_file)}"))

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not found at: {output_path}")

        return output_path

    def propagate_from_state(self, config_file, delta_v, output_file="results_continued.csv", duration=0.5):
        """
        Run propagation with delta-v maneuver using completely separate processes.

        Each propagation runs in its own Python process, ensuring no module state
        conflicts can occur.

        Args:
            config_file: Template config to use
            delta_v: Velocity vector [dvx, dvy, dvz] in m/s to apply to final state
            output_file: Output filename for second propagation
            duration: Duration in days for second propagation

        Returns:
            Tuple of (first_csv_path, second_csv_path, combined_csv_path)
        """
        config_abs = os.path.abspath(config_file)

        if not os.path.exists(config_abs):
            raise FileNotFoundError(f"Config not found: {config_abs}")

        output_filename = os.path.basename(output_file)
        first_filename = output_filename.replace('.csv', '_pre_delta.csv')
        second_filename = output_filename.replace('.csv', '_post_delta.csv')
        combined_filename = output_filename.replace('.csv', '_combined.csv')

        # STEP 1: First propagation in separate process
        print("\t-------------------------------------------------")
        print("\tRunning first propagation in separate process...")
        first_output = self.propagate(config_abs, first_filename)

        # Read results to get final state
        df1 = pd.read_csv(first_output)
        last_state = df1.iloc[-1]

        # Apply delta-v to final state
        new_state = [
            float(last_state['x']),
            float(last_state['y']),
            float(last_state['z']),
            float(last_state['vx']) + delta_v[0],
            float(last_state['vy']) + delta_v[1],
            float(last_state['vz']) + delta_v[2]
        ]

        # Load and modify config for second propagation
        with open(config_abs, 'r') as f:
            config = yaml.safe_load(f)

        config['initial_orbtial_parameters']['initial_state'] = new_state
        config['scenario_parameters']['MJD_start'] = config['scenario_parameters']['MJD_end']
        config['scenario_parameters']['MJD_end'] = config['scenario_parameters']['MJD_start'] + duration

        # Write temporary config
        temp_config = config_abs.replace('.yml', '_temp_split.yml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # STEP 2: Second propagation in new separate process
        print("\tRunning second propagation in new separate process...")
        second_output = self.propagate(temp_config, second_filename)

        # STEP 3: Combine results
        df2 = pd.read_csv(second_output)

        time_offset = df1['tSec'].iloc[-1]
        df2_shifted = df2.copy()
        df2_shifted['tSec'] = df2_shifted['tSec'] + time_offset

        combined_output = os.path.abspath(os.path.join("out", f"{combined_filename}"))
        df_combined = pd.concat([df1, df2_shifted.iloc[1:]], ignore_index=True)
        df_combined.to_csv(combined_output, index=False)

        # Generate plots
        plot_orbit_3d(first_output,
                     output_file="plots/pre_manouevre.png",
                     title="First Propagation (Before Delta-V)")
        plot_orbit_3d(second_output,
                     output_file="plots/post_manouevre.png",
                     title="Second Propagation (After Delta-V)")
        plot_orbit_3d(combined_output,
                     output_file="plots/combined.png",
                     title="Combined Propagation (t0 → t1 → t2)")

        # Clean up temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)

        print("\tSuccessfully completed propagate_from_state")
        print("\t-------------------------------------------------")
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

    delta_v = [0.0, 0.0, 50] # m/s
    try:
        csv1, csv2, csv_combined = prop.propagate_from_state("configs/config_orb.yml",
                                                              delta_v=delta_v,
                                                              output_file="results.csv")
        print(f"Successfully generated: {csv1}, {csv2}, {csv_combined}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()