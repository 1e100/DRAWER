#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys


CONDA_ENV = "isaacsim"


def run_in_conda_env(args, cwd=None, env=None):
    """Run a command inside the configured Conda environment."""
    # Build conda run command so the program executes inside the target env.
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        *args,
    ]

    # Merge current environment with any overrides.
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    # Execute the command and raise on failure.
    subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)


def parse_args(argv=None):
    """Parse command-line arguments."""
    # Create parser with defaults formatter for nicer help output.
    parser = argparse.ArgumentParser(
        description=(
            "Driver script to run Isaac Sim USD composition and simulation "
            "using an SDF reconstruction directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required flag for the scene data root directory.
    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Path to the scene data directory, e.g. "
            "/home/dmitry/Dev/third_party/DRAWER/data/cs_kitchen"
        ),
    )

    # Parse and return the CLI arguments.
    return parser.parse_args(argv)


def main(argv=None):
    """Run Isaac Sim composition and simulation on an existing SDF recon."""
    # Parse CLI arguments from the command line.
    args = parse_args(argv)

    # Resolve repo root and key paths based on this script location.
    repo_root = pathlib.Path(__file__).resolve().parent
    data_dir = pathlib.Path(args.data_root)
    scene_name = data_dir.name

    # Build the path to the SDF experiment directory for this scene.
    sdf_experiment_dir = (
        repo_root / "sdf" / "outputs" / scene_name / f"{scene_name}_sdf_recon"
    ).resolve()

    # Resolve the Isaac Sim repo directory.
    isaac_sim_dir = repo_root / "isaac_sim"

    # Run compose_usd.py to generate the USD from the SDF reconstruction.
    run_in_conda_env(
        [
            "python",
            "compose_usd.py",
            "--sdf_dir",
            str(sdf_experiment_dir),
        ],
        cwd=str(isaac_sim_dir),
    )

    # Run run_simulation.py to launch the simulation using the generated USD.
    run_in_conda_env(
        [
            "python",
            "run_simulation.py",
            "--sdf_dir",
            str(sdf_experiment_dir),
        ],
        cwd=str(isaac_sim_dir),
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # Print a clear error if a subprocess fails and exit with that code.
        print(f"Command failed with return code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except FileNotFoundError as exc:
        # Print missing file/directory issues and exit with error.
        print(f"File not found: {exc}", file=sys.stderr)
        sys.exit(1)
