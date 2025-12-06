#!/usr/bin/env python3
"""Creates transforms.json from existing COLMAP data.

Usage:
python3 create_transforms_json.py \
  --data_root /path/to/scene \
  --colmap_subdir colmap/sparse/0
"""

import argparse
import os
import pathlib
import subprocess
import sys

# Name of the Conda environment that carries nerfstudio + COLMAP utilities.
CONDA_ENV = "drawer_sdf"


def run_in_conda_env(args, cwd=None, env=None):
    """Run a command inside the configured Conda environment."""
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        *args,
    ]

    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a nerfstudio-style transforms.json from existing COLMAP binaries "
            "(cameras.bin and images.bin)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the scene data directory that contains COLMAP outputs.",
    )
    parser.add_argument(
        "--colmap_subdir",
        default="colmap/sparse/0",
        help="Relative path under data_root where cameras.bin/images.bin are stored.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to write transforms.json; defaults to data_root.",
    )
    parser.add_argument(
        "--camera_model",
        choices=["auto", "perspective", "fisheye"],
        default="auto",
        help="Camera model used when COLMAP produced the binaries; use auto to infer from COLMAP.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Convert existing COLMAP binaries to transforms.json."""
    args = parse_args(argv)

    repo_root = pathlib.Path(__file__).resolve().parent
    data_dir = pathlib.Path(args.data_root).resolve()
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else data_dir
    colmap_dir = data_dir / args.colmap_subdir
    cameras_path = colmap_dir / "cameras.bin"
    images_path = colmap_dir / "images.bin"

    if not data_dir.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_dir}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing cameras.bin at {cameras_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Missing images.bin at {images_path}")

    run_in_conda_env(
        [
            "python",
            "create_transforms_json.py",
            "--data_root",
            str(data_dir),
            "--colmap_subdir",
            args.colmap_subdir,
            "--camera_model",
            args.camera_model,
            "--output_dir",
            str(output_dir),
        ],
        cwd=str(repo_root / "sdf"),
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with return code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        sys.exit(1)
