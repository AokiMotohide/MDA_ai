# Windows DA3 Setup Notes

This fork is configured locally for the DA3 MDA checkpoint on Windows.

## Local Environment

- OS shell: PowerShell
- GPU validated here: NVIDIA GeForce RTX 3060 Ti, 8 GB VRAM
- Conda env: `mda`
- Python: `3.10`
- PyTorch: `2.7.1+cu128`
- xformers: `0.0.31.post1`
- Checkpoint: `checkpoints/MDA/DA3_MOG_Sky_LogL2.ckpt`

The checkpoint and inference outputs are intentionally ignored by Git.

## Setup

Run from the repository root:

```powershell
.\scripts\setup_windows_mda_da3.ps1
```

The script installs Miniforge3 and ffmpeg through `winget` when they are missing,
creates the `mda` conda environment, installs the Windows-compatible dependency
pins, and downloads the DA3 MDA checkpoint.

## Demo

Run a single-image DA3 smoke test:

```powershell
.\scripts\run_demo_da3.ps1
```

The default input is:

```text
assets/examples/mono/painting/painting.jpeg
```

The default output is:

```text
eval_results/demo_windows_da3/mda_mog_sky_l2
```

To use another input:

```powershell
.\scripts\run_demo_da3.ps1 -InputPath path\to\image_or_folder_or_video.mp4 -OutputDir eval_results/my_run
```

Use `-Viewer` to launch the bundled viser viewer after inference:

```powershell
.\scripts\run_demo_da3.ps1 -Viewer
```

For a completed run, the viewer can also be opened separately:

```powershell
conda run -n mda python view.py --data_dir eval_results/demo_windows_da3/mda_mog_sky_l2 --port 8080
```

## Notes

- `PYTHONNOUSERSITE=1` is set by the scripts so the conda environment does not
  pick up packages from the user-level Python site-packages directory.
- `setuptools<81` is pinned because `lightning-bolts` still imports
  `pkg_resources` while loading this checkpoint.
- `xformers==0.0.31.post1` matches `torch==2.7.1` on Windows.
- The Triton warning emitted by xformers on Windows did not block inference.
