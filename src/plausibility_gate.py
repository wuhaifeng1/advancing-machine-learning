"""Geometric plausibility gate for cross-robot loop candidates.

Motivation
----------
GICP fitness and PCM pairwise consistency cannot reject groups of mutually
consistent false loops produced by repetitive mining corridors (see the
long2 scenario, where 9/10 hard negatives are accepted into the backend).

This module adds a lightweight, sensor-grounded check that is independent of
PCM: two LiDAR scans can only share observed structure if their sensor centers
lie within the effective sensing radius. The Scan Context descriptor uses a
maximum radius ``R`` (80 m in this project), so a cross-robot loop whose implied
inter-robot translation ``|t|`` greatly exceeds ``R`` cannot correspond to a
genuine co-observation and is rejected as geometrically implausible.

The gate reads the per-candidate implied inter-robot transform already exported
in ``pcm_filter_summary.csv`` (columns ``implied_tx``/``implied_ty``) and reports
its effect on the GICP-passed / PCM-accepted candidate set. It introduces no ROS,
PCL or training dependency.
"""
from pathlib import Path
import csv
import json


def load_pcm_filter_summary(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def apply_gate(rows, max_translation_m):
    """Annotate each candidate with the plausibility-gate decision.

    gate_pass == 1  -> implied inter-robot translation is plausible (kept)
    gate_pass == 0  -> implied translation exceeds the sensing-overlap bound
    """
    out = []
    for r in rows:
        tx = float(r["implied_tx"])
        ty = float(r["implied_ty"])
        dist = (tx * tx + ty * ty) ** 0.5
        gate_pass = int(dist <= max_translation_m)
        colrio_final_used = int(r.get("colrio_final_used", 0))
        out.append(
            {
                "pair_id": r["pair_id"],
                "scenario": r["scenario"],
                "label": int(r["label"]),
                "scene_type": r.get("scene_type", ""),
                "failure_type": r.get("failure_type", ""),
                "implied_tx": tx,
                "implied_ty": ty,
                "implied_translation_m": dist,
                "colrio_final_used": colrio_final_used,
                "gate_pass": gate_pass,
                # final decision when the gate is added on top of Co-LRIO's PCM:
                "gated_final_used": int(colrio_final_used == 1 and gate_pass == 1),
            }
        )
    return out


def summarize(gated_rows, max_translation_m):
    n_all = len(gated_rows)
    n_true = sum(1 for r in gated_rows if r["label"] == 1)
    n_false = sum(1 for r in gated_rows if r["label"] == 0)

    # Baseline: Co-LRIO PCM only.
    base_used = [r for r in gated_rows if r["colrio_final_used"] == 1]
    base_true = sum(1 for r in base_used if r["label"] == 1)
    base_false = sum(1 for r in base_used if r["label"] == 0)

    # With the plausibility gate added on top of PCM.
    gated_used = [r for r in gated_rows if r["gated_final_used"] == 1]
    gated_true = sum(1 for r in gated_used if r["label"] == 1)
    gated_false = sum(1 for r in gated_used if r["label"] == 0)

    # False candidates rejected by the gate alone.
    false_rejected_by_gate = sum(1 for r in gated_rows if r["label"] == 0 and r["gate_pass"] == 0)
    true_rejected_by_gate = sum(1 for r in gated_rows if r["label"] == 1 and r["gate_pass"] == 0)

    def rate(a, b):
        return (a / b) if b else 0.0

    return {
        "max_translation_threshold_m": max_translation_m,
        "num_candidates": n_all,
        "num_true": n_true,
        "num_false": n_false,
        "baseline_pcm_only": {
            "final_used": len(base_used),
            "final_used_true": base_true,
            "final_used_false": base_false,
            "true_retention_rate": rate(base_true, n_true),
            "false_acceptance_rate": rate(base_false, n_false),
            "final_used_precision": rate(base_true, len(base_used)),
        },
        "pcm_plus_gate": {
            "final_used": len(gated_used),
            "final_used_true": gated_true,
            "final_used_false": gated_false,
            "true_retention_rate": rate(gated_true, n_true),
            "false_acceptance_rate": rate(gated_false, n_false),
            "final_used_precision": rate(gated_true, len(gated_used)),
        },
        "gate_alone": {
            "false_rejected": false_rejected_by_gate,
            "true_rejected": true_rejected_by_gate,
        },
    }


def write_gate_summary(path, gated_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "scenario",
        "label",
        "scene_type",
        "failure_type",
        "implied_tx",
        "implied_ty",
        "implied_translation_m",
        "colrio_final_used",
        "gate_pass",
        "gated_final_used",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in gated_rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                out[field] = value
            writer.writerow(out)


def run(pcm_summary_path, out_csv, out_json, max_translation_m=40.0):
    rows = load_pcm_filter_summary(pcm_summary_path)
    gated = apply_gate(rows, max_translation_m)
    metrics = summarize(gated, max_translation_m)
    write_gate_summary(out_csv, gated)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply the geometric plausibility gate.")
    parser.add_argument(
        "--pcm-summary",
        default="outputs/loop_matching_analysis/pcm_filter_summary.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/loop_matching_analysis/plausibility_gate_summary.csv",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/loop_matching_analysis/plausibility_gate_metrics.json",
    )
    parser.add_argument("--max-translation-m", type=float, default=40.0)
    args = parser.parse_args()

    metrics = run(args.pcm_summary, args.out_csv, args.out_json, args.max_translation_m)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
