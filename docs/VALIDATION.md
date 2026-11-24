# Experimental protocol

## Mapping and novel-view rendering

Capture a textured static scene with sufficient overlap and translational parallax. Run reconstruction with `--eval` to request upstream's held-out rendering split. Record the dataset, device, exact dependency versions and upstream commit, seed, iteration count and train/test split. Report training and held-out PSNR separately. The upstream split does not exclude held-out views from COLMAP's structure estimation. For a stricter evaluation, obtain calibrated poses independently or construct a training-only map.

## Compression

Train once, then test multiple pruning fractions on the same map (e.g. 0, 0.25, 0.50, 0.60, 0.75). Keep the unpruned checkpoint immutable. To render a pruned version, make a separate copy of the upstream output directory, then replace its iteration PLY with the newly pruned file. Use the same upstream renderer, dataset, iteration and split. For example, on Linux, with the upstream checkout already installed:

```bash
cp -a runs/room/gs_output runs/room/gs_pruned
cp runs/room/pruned.ply runs/room/gs_pruned/point_cloud/iteration_7000/point_cloud.ply
python /path/to/gaussian-splatting/render.py -m runs/room/gs_pruned -s runs/room/undistorted --iteration 7000 --eval
python -m benchmarks.evaluate images --reference runs/room/gs_pruned/test/ours_7000/gt --renders runs/room/gs_pruned/test/ours_7000/renders
```

Use a fresh `gs_pruned` directory. The render command uses the external checkout's environment. Report actual file sizes, counts, PSNR degradation, render time and peak device memory separately. Opacity pruning is a simple baseline and does not guarantee quality retention. A 60% point reduction is not automatically 60% peak memory reduction.

## Relocalisation

Build the reference index with the exact undistorted COLMAP images. The current baseline associates detected ORB features to nearby SIFT observations (3 pixel default); this is approximate and may reduce accuracy or coverage. Retrieval ranks candidate keyframes using ratio-filtered Hamming matches; PnP-RANSAC uses the stored sparse 3D landmarks. The implementation rebuilds the index each run, has no persistent database and is not real-time validated.

Use separately captured query images excluded from mapping and indexing. Supply actual query-camera intrinsics and distortion. Obtain ground truth from motion capture, surveyed fiducials or another independent method. Align ground truth into the reconstruction frame and determine scale independently (e.g. surveyed distances). Report median and percentile position/rotation errors, success rate including failures, reprojection threshold and runtime. Never assume monocular SfM coordinates are metres.

Ground truth JSON format: `camera_center_world` (three numbers), `R_world_to_camera` (3×3 matrix). The evaluator assumes the same frame and rotation convention as the prediction. Supply metres-per-scene-unit only after scale calibration.

## Device integration

The generic video-capture helper is a new utility, not recovered Jetson deployment code. Confirm camera driver, supported resolution, frame rate, thermal mode and power budget. Benchmark latency and memory on the actual target. No autonomous control or safety claims are made.

## Local component tests

Synthetic tests validate binary PLY round trips, opacity ranking, PnP with synthetic outliers, same-image ORB retrieval, metric arithmetic and mocked mapping command orchestration. They are smoke tests, not scene-level performance evidence. Full reconstruction, rendering and hardware tests require external dependencies and data.
