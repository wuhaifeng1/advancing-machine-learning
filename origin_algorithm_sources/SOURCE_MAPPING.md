# Co-LRIO Algorithm Source Mapping

This folder keeps compact, non-buildable excerpts from the original Co-LRIO
source tree. They are used as report evidence only. The runnable coursework
code is the ROS-free Python implementation under `src/`.

Original source root:

```text
/home/wurenche/Co_lio_orgin/src/Co-LRIO/
```

## Scan Context Candidate Recall

Original files:

```text
include/scanContextDescriptor.h
src/scanContextDescriptor.cpp
```

Key original functions:

- `makeScancontext`: converts a point cloud to a ring-sector matrix.
- `makeRingkeyFromScancontext`: computes the rowwise ring key.
- `makeSectorkeyFromScancontext`: computes the columnwise sector key.
- `distanceBtnScanContext`: aligns sector shifts and computes descriptor distance.
- `detectLoopClosure`: retrieves top candidates using ring keys and checks SC distance.

Coursework counterpart:

```text
src/scan_context.py
```

## GICP Local Verification

Original file:

```text
src/lidarOdometry.cpp
```

Key original functions:

- `loopClosureHandler`: receives a Scan Context candidate and asks the source
  robot for the source keyframe.
- `calculateTransformation`: builds source and target point clouds, initializes
  yaw from the descriptor shift, runs Fast-GICP, checks `hasConverged` and
  `fitness_score_threshold`, and creates the loop measurement.
- `saveLoopDebugClouds`: exports the compact GICP evidence used in this project.

Coursework counterpart:

```text
src/gicp_evidence.py
```

## PCM Consistency Filtering

Original files:

```text
include/outlierRejection.h
include/robustOptimizer.h
```

Key original functions:

- `computePairwiseConsistentMeasurementsMatrix`: builds the PCM consistency graph.
- `computeConsistencyError`: computes the loop pair consistency residual.
- `selectConsistentLoopClosureIndexes`: runs maximum clique selection.
- `addPendingInterRobotLoopClosure` and `acceptInterRobotLoopClosures`: manage
  pending, accepted, and final-used inter-robot loop constraints.

Coursework counterpart:

```text
src/pcm_consistency.py
```

## Lightweight Dataset Policy

The long1 and long2 bags are not included. The coursework keeps only the compact
GICP-passed evidence:

```text
generated_data/gicp_passed_pcm_status_rerun_20260528/
  scenario_*/loop_pairs.csv
  scenario_*/robot_*/poses.csv
  scenario_*/evidence_pairs/*/{source_raw,target_submap,source_init_in_target,source_gicp_in_target}.pcd
outputs/diagnosis/gicp_transforms.json
```

The offline scripts simulate the loop-processing chain on this compact set:

```text
Scan Context evidence -> GICP-passed evidence -> PCM/final-used filtering
```
