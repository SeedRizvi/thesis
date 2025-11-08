#!/usr/bin/env python3
"""
Script to generate plots for different delta-v configurations.

Runs FGO pipeline with multiple configurations and saves appropriately named plots.
"""

import subprocess
import os
import shutil

# Configuration files and their corresponding output names
configs = [
    ("configs/config_geo_short.yml", "fgo_results_no_manoeuvre.png"),
    ("configs/config_geo_short_deltaZ1.yml", "fgo_results_deltaZ1.png"),
    ("configs/config_geo_short_deltaZ2.yml", "fgo_results_deltaZ2.png"),
    ("configs/config_geo_short_deltaZ5.yml", "fgo_results_no_manoeuvre_deltaZ5.png"),
    ("configs/config_geo_short_deltaZ50.yml", "fgo_results_no_manoeuvre_deltaZ50.png"),
]

def run_fgo_and_save_plot(config_file, output_name):
    """
    Run FGO pipeline with given config and rename the output plots.

    Args:
        config_file: Path to configuration file
        output_name: Desired name for the output plot
    """
    print("=" * 70)
    print(f"Running: {config_file}")
    print("=" * 70)

    # Run FGO pipeline
    cmd = ["python3", "fgo_pipeline.py", "--config", config_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Rename the main output plot
        source = "plots/fgo_results.png"
        destination = f"plots/{output_name}"

        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"✓ Saved main plot to: {destination}")
        else:
            print(f"✗ Warning: Expected plot not found at {source}")

        # Rename the errors plot
        source_errors = "plots/fgo_results_errors.png"
        # Create errors plot name from main plot name
        destination_errors = f"plots/{output_name.replace('.png', '_errors.png')}"

        if os.path.exists(source_errors):
            shutil.move(source_errors, destination_errors)
            print(f"✓ Saved errors plot to: {destination_errors}")
        else:
            print(f"✗ Warning: Expected errors plot not found at {source_errors}")

    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {config_file}:")
        print(e.stderr)
        return False

    print()
    return True

def main():
    """Run all configurations and generate plots."""
    print("\n" + "=" * 70)
    print("Thesis B - Generating Delta-V Manoeuvre Plots")
    print("=" * 70 + "\n")

    successful = 0
    failed = 0

    for config_file, output_name in configs:
        if run_fgo_and_save_plot(config_file, output_name):
            successful += 1
        else:
            failed += 1

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully generated: {successful} configurations")
    print(f"Failed: {failed} configurations")
    print("\nGenerated plots:")
    for _, output_name in configs:
        # Check main plot
        plot_path = f"plots/{output_name}"
        if os.path.exists(plot_path):
            print(f"  ✓ {plot_path}")
        else:
            print(f"  ✗ {plot_path} (missing)")

        # Check errors plot
        errors_path = f"plots/{output_name.replace('.png', '_errors.png')}"
        if os.path.exists(errors_path):
            print(f"  ✓ {errors_path}")
        else:
            print(f"  ✗ {errors_path} (missing)")
    print("=" * 70)

if __name__ == "__main__":
    main()
