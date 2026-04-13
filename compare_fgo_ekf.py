#!/usr/bin/env python3
"""
Comparison script: FGO vs EKF

Runs each manoeuvre config in four modes:
  1. FGO Legacy (--no-gaussian): split-propagation, no manoeuvre model
  2. FGO Gaussian: FGO estimates delta-v via Gaussian impulse approximation
  3. EKF Legacy (--no-gaussian): sequential filter, no manoeuvre model
  4. EKF Gaussian: EKF estimates delta-v via Gaussian impulse approximation

Saves plots for comparison.
"""

import subprocess
import os
import shutil

# Manoeuvre configs to compare
configs = [
    ("configs/config_geo_one_rev_deltaRIC1.yml", "deltaRIC1"),
    ("configs/config_geo_one_rev_deltaRIC0.yml", "deltaRIC0"),
    ("configs/config_geo_one_rev_deltaRIC0.5.yml", "deltaRIC0.5"),
]

# Pipelines and modes to run
pipelines = [
    {"script": "fgo_pipeline.py", "tag": "fgo"},
    {"script": "ekf_pipeline.py", "tag": "ekf"},
]

modes = [
    {"flag": "--no-gaussian", "suffix": "legacy",   "label": "Legacy (no estimation)"},
    {"flag": "",              "suffix": "gaussian",  "label": "Gaussian Estimation"},
]


def run_pipeline_and_save_plot(script, config_file, output_name, extra_flags=None):
    """
    Run a pipeline script with given config and rename the output plots.

    Args:
        script: Pipeline script to run (fgo_pipeline.py or ekf_pipeline.py)
        config_file: Path to configuration file
        output_name: Desired name for the output plot
        extra_flags: Additional CLI flags as list of strings
    """
    print("=" * 70)
    print(f"Running: {script} | {config_file} -> {output_name}")
    print("=" * 70)

    cmd = ["python3", script, "--config", config_file]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Determine source plot name based on pipeline
        if "ekf" in script:
            source = "plots/ekf_results.png"
            source_errors = "plots/ekf_results_errors.png"
        else:
            source = "plots/fgo_results.png"
            source_errors = "plots/fgo_results_errors.png"

        destination = f"plots/{output_name}"
        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"+ Saved main plot to: {destination}")
        else:
            print(f"- Warning: Expected plot not found at {source}")

        destination_errors = f"plots/{output_name.replace('.png', '_errors.png')}"
        if os.path.exists(source_errors):
            shutil.move(source_errors, destination_errors)
            print(f"+ Saved errors plot to: {destination_errors}")
        else:
            print(f"- Warning: Expected errors plot not found at {source_errors}")

    except subprocess.CalledProcessError as e:
        print(f"- Error running {script}:")
        print(e.stderr)
        return False

    print()
    return True


def main():
    """Run all configs with FGO and EKF, with and without Gaussian estimation."""
    print("\n" + "=" * 70)
    print("FGO vs EKF Comparison")
    print("=" * 70 + "\n")

    successful = 0
    failed = 0
    all_outputs = []

    for config_file, config_tag in configs:
        for pipeline in pipelines:
            for mode in modes:
                output_name = f"{pipeline['tag']}_{config_tag}_{mode['suffix']}.png"
                extra_flags = [mode["flag"]] if mode["flag"] else []

                print(f"\n>>> {pipeline['tag'].upper()} | {config_tag} | {mode['label']}")
                if run_pipeline_and_save_plot(pipeline['script'], config_file,
                                              output_name, extra_flags):
                    successful += 1
                else:
                    failed += 1

                all_outputs.append((pipeline['tag'], config_tag, mode['label'], output_name))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully generated: {successful}/{successful + failed}")
    print(f"Failed: {failed}/{successful + failed}")

    print("\nGenerated plots:")
    for pipe_tag, config_tag, mode_label, output_name in all_outputs:
        plot_path = f"plots/{output_name}"
        errors_path = f"plots/{output_name.replace('.png', '_errors.png')}"
        status = "+" if os.path.exists(plot_path) else "-"
        err_status = "+" if os.path.exists(errors_path) else "-"
        print(f"  {status} {pipe_tag:4s} | {config_tag:10s} | {mode_label:25s} | {plot_path}")
        print(f"  {err_status} {'':4s} | {'':10s} | {'':25s} | {errors_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
