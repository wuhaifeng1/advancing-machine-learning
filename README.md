# Multi-Robot Loop Matching Analysis

This project is a lightweight, ROS-free coursework extraction of the
cross-robot loop matching chain in Co-LRIO. It does not keep the full long1 and
long2 bags and does not train supervised models. The retained data are only the
compact candidates that already passed GICP and have saved point cloud evidence.

## Goal

The report focuses on one failure-prone part of underground multi-robot mapping:

```text
Scan Context candidate evidence -> GICP local verification -> PCM/final-used filtering
```

The core questions are:

- how many true cross-robot loops remain usable;
- how many hard negatives pass local GICP verification;
- how many false loops are rejected or still accepted/final-used by PCM;
- how far a lightweight geometric plausibility gate can reduce false-match
  acceptance without dropping true loops.

## Proposed improvement: geometric plausibility gate

GICP fitness and PCM cannot reject groups of mutually consistent false loops in
repetitive mining corridors (in long2, 9/10 hard negatives are accepted). Two
scans can only share structure if their sensor centers lie within the LiDAR
sensing radius, so a candidate whose implied inter-robot translation greatly
exceeds that radius cannot be a genuine co-observation. `src/plausibility_gate.py`
rejects candidates with `|implied translation| > tau` (tau = 0.5 x Scan Context
max radius = 40 m). On the retained data this keeps all true loops (retention
100%) while lowering false-match acceptance from 76.5% to 23.5%; the result is
unchanged for any tau in roughly [15, 78] m.

## Original Source Evidence

Compact excerpts from the original Co-LRIO code are kept in:

```text
origin_algorithm_sources/
```

The mapping to the original Co-LRIO source files is:

- Scan Context: `Co-LRIO/src/scanContextDescriptor.cpp`
- GICP loop verification: `Co-LRIO/src/lidarOdometry.cpp`
- PCM consistency filtering: `Co-LRIO/include/outlierRejection.h`
- final-used backend flow: `Co-LRIO/include/robustOptimizer.h`

See `origin_algorithm_sources/SOURCE_MAPPING.md` for details.

## Retained Dataset

Only GICP-passed compact evidence is retained:

```text
generated_data/gicp_passed_pcm_status_rerun_20260528/
  dataset_summary.json
  scenario_01_long1_rerun_20260528/
    loop_pairs.csv
    robot_0/poses.csv
    robot_1/poses.csv
    evidence_pairs/*/
  scenario_02_long2_rerun_20260528/
    loop_pairs.csv
    robot_0/poses.csv
    robot_1/poses.csv
    evidence_pairs/*/
outputs/diagnosis/gicp_transforms.json
```

Each evidence pair stores:

```text
source_raw.pcd
target_submap.pcd
source_init_in_target.pcd
source_gicp_in_target.pcd
```

Current retained counts:

```text
GICP-passed candidates: 20
true loops: 3
false loops / hard negatives: 17
Co-LRIO PCM accepted and final-used: 16
Co-LRIO PCM rejected or pending/not-used: 4
Co-LRIO PCM accepted false positives: 13
```

## Requirements

The pipeline is ROS-free and needs only Python and three common packages:

```text
Python >= 3.8
numpy
matplotlib
PyYAML
```

## Run

1. Clone the repository and enter the project root:

```bash
git clone https://github.com/wuhaifeng1/advancing-machine-learning.git
cd advancing-machine-learning
```

2. Install dependencies (a virtual environment is recommended):

```bash
pip install -r requirements.txt
```

3. Run the full analysis (Scan Context -> GICP -> PCM -> plausibility gate):

```bash
python3 run_experiment.py --config configs/gicp_passed_pcm.yaml
```

Expected console output:

```text
loop matching analysis complete
  candidates: 20
  GICP-passed true/false: 3/17
  Co-LRIO PCM accepted false positives: 13
  false match rejection rate: 0.235
  false match acceptance rate: 0.765
  + plausibility gate -> false acceptance 0.765 -> 0.235 (true retention 1.00)
```

All result tables, metrics and figures are written to
`outputs/loop_matching_analysis/` (see the Outputs section below).

Optional overrides (custom data / output / transform paths):

```bash
python3 run_experiment.py \
  --config configs/gicp_passed_pcm.yaml \
  --data-dir generated_data/gicp_passed_pcm_status_rerun_20260528 \
  --output-dir outputs/custom_loop_matching_analysis \
  --transforms outputs/diagnosis/gicp_transforms.json
```

The plausibility gate can also be run standalone on an existing PCM summary,
e.g. to sweep the threshold:

```bash
python3 src/plausibility_gate.py --max-translation-m 40.0
```

## Outputs

The main output directory is:

```text
outputs/loop_matching_analysis/
```

Generated files:

- `scan_context_candidates.csv`: Scan Context distance, sector shift, label and status for each retained candidate.
- `gicp_evidence_summary.csv`: GICP status, fitness, convergence, point counts and evidence geometry.
- `pcm_filter_summary.csv`: Co-LRIO PCM/final-used status plus offline PCM-style clique membership.
- `pcm_pairwise_matrices.json`: per-scenario pairwise consistency matrices.
- `loop_filter_metrics.json`: true-loop retention rate, false-match rejection rate and false-match acceptance rate (also embeds the plausibility-gate before/after metrics).
- `plausibility_gate_summary.csv`: per-candidate implied translation, gate decision and gated final-used status.
- `plausibility_gate_metrics.json`: PCM-only vs PCM+gate comparison (retention, false acceptance, precision).
- `true_loop_visualization.png`: representative true-loop GICP overlay.
- `hard_negative_visualization.png`: representative hard-negative/false-loop GICP overlay.

## Source Layout

```text
run_experiment.py                     main analysis entry
configs/gicp_passed_pcm.yaml          data/output/algorithm parameters
origin_algorithm_sources/             original Co-LRIO algorithm excerpts
src/scan_context.py                   ROS-free Scan Context implementation
src/gicp_evidence.py                  GICP-passed evidence summary
src/pcm_consistency.py                PCM-style consistency matrix and clique
src/plausibility_gate.py              proposed geometric plausibility gate
src/loop_matching_analysis.py         end-to-end lightweight pipeline
src/pcd_io.py                         ASCII/binary PCD reader
generated_data/                       retained GICP-passed compact evidence
outputs/diagnosis/gicp_transforms.json
                                      recovered relative transforms used by PCM analysis
outputs/loop_matching_analysis/       final CSV/JSON/PNG outputs
```

Exploratory diagnosis tools, full-map PCD examples and cache files are
intentionally excluded from the cleaned structure.

## Contributor
wuhaifeng
