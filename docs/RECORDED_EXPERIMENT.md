# Original reconstruction run — 20 February 2026

Source: user-provided plain-text export of `Gaussian_Splat.ipynb`. These excerpts are retained for provenance; the original media, full model and image artifacts were not supplied to the repository reconstruction. No fresh training run was performed to produce these values.

The notebook cloned `graphdeco-inria/gaussian-splatting` recursively, installed its CUDA extensions and used COLMAP 3.7 with CPU SIFT and exhaustive matching. FFmpeg extracted 48 frames from a 15.86-second video at 3 FPS. COLMAP registered all 48 frames. The initial training invocation preceded a correction to the sparse model layout (`sparse/0/`). The subsequent successful run used 7,000 iterations, without `--eval`.

Selected verbatim output lines (installation and progress noise omitted):

```text
Frames extracted:
48
 => Reconstruction with 48 images and 7495 points
Reading camera 48/48 [20/02 17:36:57]
Number of points at initialisation :  7495 [20/02 17:36:58]
[ITER 7000] Evaluating train: L1 0.012118528038263321 PSNR 34.7835433959961 [20/02 17:40:35]
[ITER 7000] Saving Gaussians [20/02 17:40:35]
Training complete. [20/02 17:40:38]
Rendering progress: 100% 48/48 [00:12<00:00,  3.95it/s]
Rendering progress: 0it [00:00, ?it/s]
```

The export includes calls to download `output_room.zip`, `output_room/rendered_video.mp4` and `output_room/point_cloud/iteration_7000/point_cloud.ply`. These files are not available in the supplied materials. If recovered, add consent-cleared render examples and checksums; do not fabricate replacements or present synthetic tests as original artifacts.

This run establishes historical reconstruction and training-view rendering only. It does not establish rover hardware deployment, a 27.4 dB held-out benchmark, measured 60% runtime memory reduction, or metric pose errors of 4.5 cm / 2.1 degrees. These separate results require original records or independent validation before being reported as verified repository results.
