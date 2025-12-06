#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys

# Name of the Conda environment that contains Marigold + SDF deps.
CONDA_ENV = "drawer_sdf"


def run_in_conda_env(args, cwd=None):
    """Run a command inside the configured Conda environment."""
    # Build the full command line using `conda run` so the program runs in the env.
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        *args,
    ]

    # Execute the command and raise if it fails.
    subprocess.run(cmd, check=True, cwd=cwd)


def make_symlink(src, dst):
    """Create a symbolic link, mimicking plain `ln -s` behavior."""
    # Normalize paths to pathlib objects.
    src_path = pathlib.Path(src)
    dst_path = pathlib.Path(dst)

    # Ensure the destination directory exists.
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the symlink; this will raise if the target already exists.
    os.symlink(src_path, dst_path)


def parse_args(argv=None):
    """Parse command-line arguments."""
    # Create the argument parser with default-values formatter.
    parser = argparse.ArgumentParser(
        description=(
            "Driver script to run Marigold mono depth/normals and SDF reconstruction "
            "inside a Conda environment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add required flag for the scene data root directory.
    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Path to the scene data directory."
        ),
    )

    # Add required flag for the image directory name relative to data_root.
    parser.add_argument(
        "--image_dir_name",
        required=True,
        help="Name of the subdirectory under data_root that contains input RGB images.",
    )

    # Add configurable downscale factor with a default of 2.
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=2,
        help="Downscale factor for images used in the SDF pipeline.",
    )

    # Parse and return the arguments.
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full Marigold + SDF pipeline."""
    # Parse all command-line arguments.
    args = parse_args(argv)

    # Resolve paths for the repo, data root, and image directory.
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = pathlib.Path(args.data_root).resolve()
    image_dir = data_dir / args.image_dir_name

    # ----- Monocular depth and normals (Marigold) -----

    # Set the working directory to the marigold subdirectory.
    marigold_dir = base_dir / "marigold"

    # Run depth prediction with Marigold.
    run_in_conda_env(
        [
            "python",
            "run.py",
            "--checkpoint",
            "GonzaloMG/marigold-e2e-ft-depth",
            "--modality",
            "depth",
            "--input_rgb_dir",
            str(image_dir),
            "--output_dir",
            str(data_dir / "marigold_ft"),
        ],
        cwd=str(marigold_dir),
    )

    # Run normals prediction with Marigold.
    run_in_conda_env(
        [
            "python",
            "run.py",
            "--checkpoint",
            "GonzaloMG/marigold-e2e-ft-normals",
            "--modality",
            "normals",
            "--input_rgb_dir",
            str(image_dir),
            "--output_dir",
            str(data_dir / "marigold_ft"),
        ],
        cwd=str(marigold_dir),
    )

    # Post-process Marigold output to create depth/normal folders.
    run_in_conda_env(
        [
            "python",
            "read_marigold.py",
            "--data_dir",
            str(data_dir / "marigold_ft"),
        ],
        cwd=str(marigold_dir),
    )

    # Create depth and normal symlinks at the data root.
    make_symlink(
        src=str(data_dir / "marigold_ft" / "depth"),
        dst=str(data_dir / "depth"),
    )
    make_symlink(
        src=str(data_dir / "marigold_ft" / "normal"),
        dst=str(data_dir / "normal"),
    )

    # ----- SDF reconstruction -----

    # Set the working directory to the sdf subdirectory.
    sdf_repo_dir = base_dir / "sdf"

    # Prepare paths for SDF outputs and experiment name.
    sdf_experiment_name = f"{data_dir.name}_sdf_recon"
    outputs_dir = sdf_repo_dir / "outputs" / data_dir.name
    sdf_experiment_dir = outputs_dir / sdf_experiment_name

    # Train the SDF model using the specified configuration.
    run_in_conda_env(
        [
            "python",
            "scripts/train.py",
            "bakedsdf",
            "--vis",
            "wandb",
            "--output-dir",
            str(outputs_dir),
            "--experiment-name",
            sdf_experiment_name,
            "--trainer.steps-per-eval-image",
            "2000",
            "--trainer.steps-per-eval-all-images",
            "250001",
            "--trainer.max-num-iterations",
            "250001",
            "--trainer.steps-per-eval-batch",
            "250001",
            "--optimizers.fields.scheduler.max-steps",
            "250000",
            "--optimizers.field-background.scheduler.max-steps",
            "250000",
            "--optimizers.proposal-networks.scheduler.max-steps",
            "250000",
            "--pipeline.model.eikonal-anneal-max-num-iters",
            "250000",
            "--pipeline.model.beta-anneal-max-num-iters",
            "250000",
            "--pipeline.model.sdf-field.bias",
            "1.5",
            "--pipeline.model.sdf-field.inside-outside",
            "True",
            "--pipeline.model.eikonal-loss-mult",
            "0.01",
            "--pipeline.model.num-neus-samples-per-ray",
            "24",
            "--pipeline.datamanager.train-num-rays-per-batch",
            "4096",
            "--machine.num-gpus",
            "1",
            "--pipeline.model.scene-contraction-norm",
            "inf",
            "--pipeline.model.mono-normal-loss-mult",
            "0.2",
            "--pipeline.model.mono-depth-loss-mult",
            "1.0",
            "--pipeline.model.near-plane",
            "1e-6",
            "--pipeline.model.far-plane",
            "100",
            "panoptic-data",
            "--data",
            str(data_dir),
            "--panoptic_data",
            "False",
            "--mono_normal_data",
            "True",
            "--mono_depth_data",
            "True",
            "--panoptic_segment",
            "False",
            "--downscale_factor",
            str(args.downscale_factor),
            "--num_max_image",
            "2000",
        ],
        cwd=str(sdf_repo_dir),
    )

    # ----- Mesh extraction -----

    # Extract a mesh from the trained SDF model.
    run_in_conda_env(
        [
            "python",
            "scripts/extract_mesh.py",
            "--load-config",
            str(sdf_experiment_dir / "config.yml"),
            "--output-path",
            str(sdf_experiment_dir / "mesh.ply"),
            "--bounding-box-min",
            "-2.0",
            "-2.0",
            "-2.0",
            "--bounding-box-max",
            "2.0",
            "2.0",
            "2.0",
            "--resolution",
            "2048",
            "--marching_cube_threshold",
            "0.0035",
            "--create_visibility_mask",
            "True",
            "--simplify-mesh",
            "True",
        ],
        cwd=str(sdf_repo_dir),
    )

    # ----- Texturing -----

    # Ensure the directory for the textured mesh exists.
    texture_mesh_dir = sdf_experiment_dir / "texture_mesh"
    texture_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Run mesh texturing.
    run_in_conda_env(
        [
            "python",
            "scripts/texture.py",
            "--load-config",
            str(sdf_experiment_dir / "config.yml"),
            "--output-dir",
            str(texture_mesh_dir),
            "--input_mesh_filename",
            str(sdf_experiment_dir / "mesh-simplify.ply"),
            "--target_num_faces",
            "300000",
        ],
        cwd=str(sdf_repo_dir),
    )

    # ----- Pose saving -----

    # Save camera/pose information for the reconstructed mesh.
    run_in_conda_env(
        [
            "python",
            "scripts/save_pose.py",
            "--ckpt_dir",
            str(sdf_experiment_dir),
            "--save_dir",
            str(data_dir),
        ],
        cwd=str(sdf_repo_dir),
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # Surface a clear error when a subprocess fails and propagate the exit code.
        print(f"Command failed with return code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except FileExistsError as exc:
        # Surface a filesystem error such as an existing symlink.
        print(f"Filesystem error: {exc}", file=sys.stderr)
        sys.exit(1)
