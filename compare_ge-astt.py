#!/usr/bin/env python3
"""
Comparison script: Gaussian Estimation vs Adaptive Split-Time Tracking (GE vs ASTT)

Runs each manoeuvre config twice:
  1. Legacy (--no-gaussian): split-propagation, FGO has no manoeuvre model
  2. Gaussian estimation: FGO estimates delta-v via Gaussian impulse approximation

Saves plots side-by-side for comparison.
"""

import subprocess
import os
import shutil

# Manoeuvre configs to compare
configs = [
    ("configs/config_geo_one_rev_deltaRIC1.yml", "deltaRIC1"),
    ("configs/config_geo_one_rev_deltaRIC0.5.yml", "deltaRIC0.5"),
]

# Two modes to run
modes = [
    {"flag": "--no-gaussian", "suffix": "legacy",   "label": "Legacy (no estimation)"},
    {"flag": "",              "suffix": "gaussian",  "label": "Gaussian Estimation"},
]


def run_fgo_and_save_plot(config_file, output_name, extra_flags=None):
    """
    Run FGO pipeline with given config and rename the output plots.

    Args:
        config_file: Path to configuration file
        output_name: Desired name for the output plot
        extra_flags: Additional CLI flags as list of strings
    """
    print("=" * 70)
    print(f"Running: {config_file} -> {output_name}")
    print("=" * 70)

    cmd = ["python3", "fgo_pipeline.py", "--config", config_file]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Rename the main output plot
        source = "plots/fgo_results.png"
        destination = f"plots/{output_name}"

        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"+ Saved main plot to: {destination}")
        else:
            print(f"- Warning: Expected plot not found at {source}")

        # Rename the errors plot
        source_errors = "plots/fgo_results_errors.png"
        destination_errors = f"plots/{output_name.replace('.png', '_errors.png')}"

        if os.path.exists(source_errors):
            shutil.move(source_errors, destination_errors)
            print(f"+ Saved errors plot to: {destination_errors}")
        else:
            print(f"- Warning: Expected errors plot not found at {source_errors}")

    except subprocess.CalledProcessError as e:
        print(f"- Error running {config_file}:")
        print(e.stderr)
        return False

    print()
    return True


def main():
    """Run all manoeuvre configurations with and without Gaussian estimation."""
    print("\n" + "=" * 70)
    print("GE vs ASTT: Gaussian Estimation vs Legacy Split-Propagation")
    print("=" * 70 + "\n")

    successful = 0
    failed = 0
    all_outputs = []

    for config_file, config_tag in configs:
        for mode in modes:
            output_name = f"fgo_results_{config_tag}_{mode['suffix']}.png"
            extra_flags = [mode["flag"]] if mode["flag"] else []

            print(f"\n>>> {config_tag} | {mode['label']}")
            if run_fgo_and_save_plot(config_file, output_name, extra_flags):
                successful += 1
            else:
                failed += 1

            all_outputs.append((config_tag, mode["label"], output_name))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully generated: {successful}/{successful + failed}")
    print(f"Failed: {failed}/{successful + failed}")

    print("\nGenerated plots:")
    for config_tag, mode_label, output_name in all_outputs:
        plot_path = f"plots/{output_name}"
        errors_path = f"plots/{output_name.replace('.png', '_errors.png')}"
        status = "+" if os.path.exists(plot_path) else "-"
        err_status = "+" if os.path.exists(errors_path) else "-"
        print(f"  {status} {config_tag:10s} | {mode_label:25s} | {plot_path}")
        print(f"  {err_status} {'':10s} | {'':25s} | {errors_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
