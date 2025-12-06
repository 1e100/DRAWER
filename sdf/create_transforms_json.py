#!/usr/bin/env python3
"""Convert existing COLMAP binaries to a nerfstudio-compatible transforms.json.

Unlike the script with the same name above, this one is run inside a conda venv.
"""

import argparse
import pathlib
import sys

from nerfstudio.process_data import colmap_utils, process_data_utils


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate transforms.json from precomputed COLMAP outputs "
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
        choices=["auto", *process_data_utils.CAMERA_MODELS.keys()],
        default="auto",
        help="Camera model used when COLMAP produced the binaries; auto infers from COLMAP.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Create transforms.json using nerfstudio's COLMAP converter."""
    args = parse_args(argv)

    data_dir = pathlib.Path(args.data_root)
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

    cameras = colmap_utils.read_cameras_binary(cameras_path)
    images = colmap_utils.read_images_binary(images_path)

    if len(cameras) != 1:
        raise ValueError(f"Expected exactly one camera; found {len(cameras)}")

    camera = cameras[next(iter(cameras.keys()))]
    params = camera.params

    if len(params) == 3:  # SIMPLE_PINHOLE
        fl_x = fl_y = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif len(params) >= 4:  # PINHOLE / OPENCV variants
        fl_x = float(params[0])
        fl_y = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise ValueError(f"Unexpected number of camera params: {len(params)}")

    # Determine which distortion parameters to include.
    distortion = {}
    if args.camera_model == "auto":
        camera_model_name = camera.model
    else:
        camera_model_name = process_data_utils.CAMERA_MODELS[args.camera_model].value

    if camera_model_name == process_data_utils.CameraModel.OPENCV.value and len(params) >= 8:
        distortion = {
            "k1": float(params[4]),
            "k2": float(params[5]),
            "p1": float(params[6]),
            "p2": float(params[7]),
        }
    elif camera_model_name == process_data_utils.CameraModel.OPENCV_FISHEYE.value and len(params) >= 8:
        distortion = {
            "k1": float(params[4]),
            "k2": float(params[5]),
            "k3": float(params[6]),
            "k4": float(params[7]),
        }

    frames = []
    for _, im_data in images.items():
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = colmap_utils.np.concatenate([rotation, translation], 1)
        w2c = colmap_utils.np.concatenate([w2c, colmap_utils.np.array([[0, 0, 0, 1]])], 0)
        c2w = colmap_utils.np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[colmap_utils.np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        frame = {
            "file_path": pathlib.Path(f"./images/{im_data.name}").as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    output = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": camera.width,
        "h": camera.height,
        "camera_model": camera_model_name,
        **distortion,
        "frames": frames,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_path = output_dir / "transforms.json"
    transforms_path.write_text(colmap_utils.json.dumps(output, indent=4), encoding="utf-8")
    print(f"Wrote transforms.json with {len(frames)} frames to {transforms_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        sys.exit(1)
