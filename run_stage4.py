#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys


CONDA_ENV = "drawer_splat"
CUDA_ARCH_LIST = "8.6"


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
            "Run Gaussian splatting and material fusion "
            "for a given SDF reconstruction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Path to the scene data directory, e.g. "
            "/home/dmitry/Dev/third_party/DRAWER/data/cs_kitchen"
        ),
    )

    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=2,
        help="Downscale factor passed into the splatting pipeline.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    repo_root = pathlib.Path(__file__).resolve().parent
    data_dir = pathlib.Path(args.data_root)
    scene_name = data_dir.name

    sdf_experiment_dir = (
        repo_root / "sdf" / "outputs" / scene_name / f"{scene_name}_sdf_recon"
    ).resolve()

    splat_dir_root = repo_root / "splat"

    mesh_path = sdf_experiment_dir / "texture_mesh" / "mesh-simplify.obj"
    save_extra_info_dir = splat_dir_root / "vis" / scene_name / "gs_extra_info"
    run_note = "dense_1"
    area = 2e-5

    cmake_cuda_architectures = " ".join(CUDA_ARCH_LIST.replace(".", "").split(";"))

    env_overrides = {
        "CC": "/usr/bin/gcc-11",
        "CXX": "/usr/bin/g++-11",
        "CUDA_ARCH_LIST": CUDA_ARCH_LIST,
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCH_LIST,
        "CMAKE_CUDA_ARCHITECTURES": cmake_cuda_architectures,
    }

    save_extra_info_dir.mkdir(parents=True, exist_ok=True)

    splat_outputs_root = splat_dir_root / "outputs" / scene_name
    splat_experiment_name = f"{scene_name}_mesh_gauss_splat"
    splat_experiment_dir = splat_outputs_root / splat_experiment_name

    # Train Gaussian splats
    run_in_conda_env(
        [
            "python",
            "nerfstudio/scripts/train.py",
            "splatfacto_on_mesh_uc",
            "--vis",
            "wandb",
            "--output-dir",
            str(splat_outputs_root),
            "--experiment-name",
            splat_experiment_name,
            "--pipeline.model.mesh_area_to_subdivide",
            str(area),
            "--pipeline.model.acm_lambda",
            "1.0",
            "--pipeline.model.elevate_coef",
            "2.0",
            "--pipeline.model.upper_scale",
            "2.0",
            "--pipeline.model.continue_cull_post_densification",
            "True",
            "--pipeline.model.gaussian_save_extra_info_path",
            str(save_extra_info_dir / f"{run_note}.pt"),
            "--pipeline.model.mesh_depth_lambda",
            "1.0",
            "--pipeline.model.reset_alpha_every",
            "30",
            "--pipeline.model.use_scale_regularization",
            "True",
            "--pipeline.model.max_gauss_ratio",
            "1.5",
            "--max-num-iterations",
            "30000",
            "panoptic-data",
            "--data",
            str(data_dir),
            "--mesh_gauss_path",
            str(mesh_path),
            "--mesh_area_to_subdivide",
            str(area),
            "--mesh_depth",
            "True",
            "--downscale_factor",
            str(args.downscale_factor),
            "--num_max_image",
            "2000",
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
    )

    interior_dir = splat_dir_root / "vis" / scene_name / "interior"

    # Matfuse texture generation
    run_in_conda_env(
        [
            "python",
            "scripts/matfuse_texgen.py",
            "--splat_dir",
            str(splat_experiment_dir),
            "--output_dir",
            str(interior_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
            "--ckpt",
            "ckpts/matfuse-full.ckpt",
            "--config",
            "scripts/matfuse_sd/src/configs/diffusion/matfuse-ldm-vq_f8.yaml",
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
    )

    # Paint AO
    run_in_conda_env(
        [
            "python",
            "scripts/paint_ao.py",
            "--src_dir",
            str(interior_dir),
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
    )

    # Merge splats (default)
    run_in_conda_env(
        [
            "python",
            "scripts/splat_merge.py",
            "--splat_dir",
            str(splat_experiment_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
            "--interior_dir",
            str(interior_dir),
            "--save_note",
            "default",
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
    )

    # Validation merge
    val_dir = splat_dir_root / "vis" / scene_name / "val"
    run_in_conda_env(
        [
            "python",
            "scripts/splat_merge_val.py",
            "--splat_dir",
            str(splat_experiment_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
            "--interior_dir",
            str(interior_dir),
            "--save_note",
            "default",
            "--save_dir",
            str(val_dir),
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
    )

    # Objaverse-style merge
    objaverse_dir = splat_dir_root / "vis" / scene_name / "objaverse"
    run_in_conda_env(
        [
            "python",
            "scripts/splat_merge_objaverse.py",
            "--splat_dir",
            str(splat_experiment_dir),
            "--sdf_dir",
            str(sdf_experiment_dir),
            "--interior_dir",
            str(interior_dir),
            "--save_note",
            "default",
            "--save_dir",
            str(objaverse_dir),
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
    )

    # Final export
    run_in_conda_env(
        [
            "python",
            "scripts/splat_export.py",
            "--splat_dir",
            str(splat_experiment_dir),
            "--save_note",
            "default",
        ],
        cwd=str(splat_dir_root),
        env=env_overrides,
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
