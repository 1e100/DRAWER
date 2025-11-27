# Repository Guidelines

## Project Structure & Module Organization
DRAWER stitches together reconstruction, perception, and simulation. `scripts/` holds entrypoints (`run_stage1_sdf.sh`–`run_stage4_gsplat.sh`, `run_system.sh`) plus data/ckpt download helpers. `sdf/` contains the BakedSDF-style recon code and configs; outputs typically live in `sdf/outputs/<scene>/`. `perception/` houses the staged drawer/door localization pipeline; `grounded_sam/`, `sam/`, and `3DOI/` are vendor trees used by perception. `splat/` contains Gaussian Splat on NeRFStudio, `isaac_sim/` handles USD export and simulation, `assets/` stores visuals, and `marigold/`/`3DOI` hold supporting models.

## Build, Test, and Development Commands
Set up environments per module: `bash sdf/env.sh`, `bash splat/env.sh`, and `bash isaac_sim/env.sh` (CUDA 11.8 expected). Download sample data with `bash scripts/data.sh` and checkpoints via `bash scripts/ckpt.sh`. Typical runs: `bash scripts/run_stage1_sdf.sh` for SDF reconstruction, `bash scripts/run_stage2_perception.sh` for detection+fitting (requires `OPENAI_KEY`), `bash scripts/run_stage3_isaacsim.sh` for USD simulation, `bash scripts/run_stage4_gsplat.sh` for Gaussian splats, or `bash scripts/run_system.sh` for end-to-end. Adjust `data_dir`, `image_dir`, and GPU `CUDA_VISIBLE_DEVICES` inside scripts rather than hardcoding.

## Coding Style & Naming Conventions
Prefer PEP8 Python (4-space indents, snake_case, minimal side effects), mirroring existing stage files (`percept_stage1.py`, etc.). Keep randomness controlled (seed defaults to 42 in perception) and make device selection explicit. For shell, use Bash, keep stage entrypoints under `scripts/` (`run_stageX_*.sh`), and write paths relative to repo root. Include short docstrings or comments only where behavior is non-obvious and note any new env vars.

## Testing Guidelines
There is no dedicated unit-test harness; validation is stage-driven. After changes, rerun the smallest affected stage against a small scene (e.g., the sample downloaded by `scripts/data.sh` with a higher `downscale_factor`), and confirm outputs: SDF meshes in `sdf/outputs/<scene>/`, perception groups in `<data_dir>/perception/vis_groups_final_mesh/`, USD exports in `isaac_sim/outputs/`, and splats in `splat/outputs/`. Capture warnings from tqdm/logging and compare key metrics (e.g., number of detected doors, mesh triangle counts) to previous runs when possible.

## Commit & Pull Request Guidelines
Keep commit subjects short and imperative, optionally scoped (e.g., `[perception] handle empty masks`). In PRs, describe which stage is affected, list commands run and their outputs, call out new dependencies or GPU/driver expectations, and attach qualitative artifacts (sample render, segmentation overlay) when visuals change. Link to related issues or papers if introducing new algorithms, and note any data/ckpt URLs you added.

## Security & Configuration Tips
Store API keys (e.g., `OPENAI_KEY` for `percept_stage5.py`) and private dataset paths in your shell env, not in tracked files. Avoid absolute paths; parameterize `data_dir`/`image_dir` so runs are reproducible across machines. Large downloads should stay outside the repo history—use the provided scripts and `.gitignore` patterns to keep outputs from being committed.
