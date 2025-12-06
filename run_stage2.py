#!/usr/bin/env python3

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


CONDA_ENV = "drawer_sdf"


def run_in_conda_env(args, cwd=None, env=None):
    """Run a command inside the configured Conda environment."""
    # Build conda run command to execute inside the target env.
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        *args,
    ]

    # Merge base environment with any overrides.
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # Run the process and raise if it fails.
    subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)


def make_empty_dir(path: pathlib.Path):
    """Ensure directory exists and is emptied of all contents."""
    # Create directory if it does not exist yet.
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    # Remove all children inside the directory.
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_args(argv=None):
    """Parse command-line arguments."""
    # Create parser with defaults formatter for nicer help text.
    parser = argparse.ArgumentParser(
        description=(
            "Driver script to run grounded SAM + perception + door fitting "
            "inside a Conda environment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add required flag for the scene data root directory.
    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Path to the scene data directory, e.g. "
            "/home/dmitry/Dev/third_party/DRAWER/data/my_kitchen"
        ),
    )

    # Add required flag for the image directory name under data_root.
    parser.add_argument(
        "--image_dir_name",
        required=True,
        help="Name of the subdirectory under data_root that contains input RGB images.",
    )

    # Return parsed arguments.
    return parser.parse_args(argv)


def main(argv=None):
    """Run grounded SAM + perception + door fitting pipeline."""
    # Parse command-line flags.
    args = parse_args(argv)

    # Resolve repo root and important paths.
    repo_root = pathlib.Path(__file__).resolve().parent
    data_dir = pathlib.Path(args.data_root)
    image_dir = data_dir / args.image_dir_name
    scene_name = data_dir.name

    # Construct path to SDF experiment directory.
    sdf_experiment_dir = (
        repo_root / "sdf" / "outputs" / scene_name / f"{scene_name}_sdf_recon"
    ).resolve()

    # Prepare common environment overrides.
    common_env = {
        "CC": "/usr/bin/gcc-11",
        "CXX": "/usr/bin/g++-11",
        "OPENAI_KEY": os.environ.get("OPENAI_KEY", ""),
    }

    # ----- Grounded SAM: detect doors -----

    # Set working directory to grounded_sam repo.
    grounded_sam_dir = repo_root / "grounded_sam"

    # Run door detection with Grounded SAM.
    run_in_conda_env(
        [
            "python",
            "grounded_sam_detect_doors.py",
            "--config",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "--grounded_checkpoint",
            "groundingdino_swint_ogc.pth",
            "--sam_checkpoint",
            "sam_vit_h_4b8939.pth",
            "--input_dir",
            str(image_dir),
            "--output_dir",
            str(data_dir / "grounded_sam"),
            "--box_threshold",
            "0.3",
            "--text_threshold",
            "0.25",
            "--text_prompt",
            (
                "drawer. drawer door. cabinet door. drawer face. drawer front. "
                "fridge door. fridge front. refridgerator door. refridgerator front."
            ),
            "--device",
            "cuda",
        ],
        cwd=str(grounded_sam_dir),
        env={**common_env, "CUDA_VISIBLE_DEVICES": "0"},
    )

    # ----- Perception stages 1–3 -----

    # Set working directory to perception repo.
    perception_dir = repo_root / "perception"

    # Run perception stage 1.
    run_in_conda_env(
        [
            "python",
            "percept_stage1.py",
            "--data_dir",
            str(data_dir),
            "--image_dir",
            str(image_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
            "--num_max_frames",
            "1500",
            "--num_faces_simplified",
            "80000",
        ],
        cwd=str(perception_dir),
        env=common_env,
    )

    # Run perception stage 2.
    run_in_conda_env(
        [
            "python",
            "percept_stage2.py",
            "--data_dir",
            str(data_dir),
        ],
        cwd=str(perception_dir),
        env=common_env,
    )

    # Run perception stage 3.
    run_in_conda_env(
        [
            "python",
            "percept_stage3.py",
            "--data_dir",
            str(data_dir),
            "--image_dir",
            str(image_dir),
        ],
        cwd=str(perception_dir),
        env=common_env,
    )

    # Clear vis_groups_back_match_gsam_handle directory.
    back_match_handle_dir = (
        data_dir / "perception" / "vis_groups_back_match_gsam_handle"
    )
    make_empty_dir(back_match_handle_dir)

    # ----- Grounded SAM: detect handles -----

    # Run handle detection with Grounded SAM.
    run_in_conda_env(
        [
            "python",
            "grounded_sam_detect_handles.py",
            "--config",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "--grounded_checkpoint",
            "groundingdino_swint_ogc.pth",
            "--sam_checkpoint",
            "sam_vit_h_4b8939.pth",
            "--input_dir",
            str(data_dir / "perception" / "vis_groups_back_match"),
            "--output_dir",
            str(back_match_handle_dir),
            "--box_threshold",
            "0.3",
            "--text_threshold",
            "0.25",
            "--text_prompt",
            "handle",
            "--device",
            "cuda",
        ],
        cwd=str(grounded_sam_dir),
        env={**common_env, "CUDA_VISIBLE_DEVICES": "0"},
    )

    # ----- Perception stages 4–6 -----

    # Run perception stage 4.
    run_in_conda_env(
        [
            "python",
            "percept_stage4.py",
            "--data_dir",
            str(data_dir),
        ],
        cwd=str(perception_dir),
        env=common_env,
    )

    # Run perception stage 5.
    run_in_conda_env(
        [
            "python",
            "percept_stage5.py",
            "--data_dir",
            str(data_dir),
            # To pass API key as CLI arg, uncomment and wire OPENAI_KEY as needed:
            # "--api_key",
            # common_env["OPENAI_KEY"],
        ],
        cwd=str(perception_dir),
        env=common_env,
    )

    # Run perception stage 6.
    run_in_conda_env(
        [
            "python",
            "percept_stage6.py",
            "--data_dir",
            str(data_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
        ],
        cwd=str(perception_dir),
        env=common_env,
    )

    # Copy final mesh metadata JSON to grounded_sam directory.
    src_json = data_dir / "perception" / "vis_groups_final_mesh" / "all.json"
    dst_dir = data_dir / "grounded_sam"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_json, dst_dir / "all.json")

    # ----- Fit doors: sam_project -----

    # Set working directory to sdf repo root.
    sdf_dir_root = repo_root / "sdf"

    # Run SAM projection onto SDF.
    run_in_conda_env(
        [
            "python",
            "scripts/sam_project.py",
            "--data_dir",
            str(data_dir),
            "--image_dir",
            str(image_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
        ],
        cwd=str(sdf_dir_root),
        env=common_env,
    )

    # ----- SAM propagate -----

    # Set working directory to sam repo.
    sam_dir = repo_root / "sam"

    # Run mask propagation.
    run_in_conda_env(
        [
            "python",
            "propagate.py",
            "--sam_ckpt",
            str(repo_root / "grounded_sam" / "sam_vit_h_4b8939.pth"),
            "--data_dir",
            str(data_dir),
            "--image_dir",
            str(image_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
        ],
        cwd=str(sam_dir),
        env=common_env,
    )

    # ----- 3DOI art inference -----

    # Set working directory to 3DOI repo.
    threedooi_dir = repo_root / "3DOI"

    # Run 3DOI inference.
    run_in_conda_env(
        [
            "python",
            "art_infer.py",
            "--config-name",
            "sam_inference",
            "checkpoint_path=checkpoints/checkpoint_20230515.pth",
            f"output_dir={data_dir / 'art_infer'}",
            f"data_dir={data_dir}",
            f"image_dir={image_dir}",
        ],
        cwd=str(threedooi_dir),
        env=common_env,
    )

    # ----- Final door fitting -----

    # Run door fitting on the SDF mesh.
    run_in_conda_env(
        [
            "python",
            "scripts/fit_doors.py",
            "--data_dir",
            str(data_dir),
            "--image_dir",
            str(image_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
        ],
        cwd=str(sdf_dir_root),
        env=common_env,
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
    except FileExistsError as exc:
        print(f"Filesystem error: {exc}", file=sys.stderr)
        sys.exit(1)
