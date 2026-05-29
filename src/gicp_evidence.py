from pathlib import Path
import csv
import math

import numpy as np

from .pcd_io import read_pcd_xyz


def _safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _safe_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def _bbox(points):
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.ptp(points, axis=0)


def _centroid(points):
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.mean(points, axis=0)


def analyze_gicp_pair(row):
    evidence = Path(row["evidence_abs"])
    source_raw = read_pcd_xyz(evidence / "source_raw.pcd")
    target = read_pcd_xyz(evidence / "target_submap.pcd")
    source_init = read_pcd_xyz(evidence / "source_init_in_target.pcd")
    source_gicp = read_pcd_xyz(evidence / "source_gicp_in_target.pcd")

    scan_raw = max(_safe_float(row, "scan_raw"), 1.0)
    map_raw = max(_safe_float(row, "map_raw"), 1.0)
    scan_filtered = _safe_float(row, "scan_filtered")
    map_filtered = _safe_float(row, "map_filtered")
    target_bbox = _bbox(target)
    source_bbox = _bbox(source_raw)
    centroid_shift = float(np.linalg.norm(_centroid(source_gicp) - _centroid(source_init)))
    centroid_distance_after = float(np.linalg.norm(_centroid(source_gicp) - _centroid(target)))

    return {
        "pair_id": row["pair_id"],
        "scenario": row["scenario"],
        "source_robot": row["source_robot"],
        "source_key": int(row["source_key"]),
        "target_robot": row["target_robot"],
        "target_key": int(row["target_key"]),
        "label": int(row["label"]),
        "scene_type": row.get("scene_type", ""),
        "failure_type": row.get("failure_type", ""),
        "gicp_status": row.get("gicp_status", row.get("status", "")),
        "fitness": _safe_float(row, "fitness"),
        "fitness_threshold": _safe_float(row, "fitness_threshold"),
        "converged": _safe_int(row, "converged"),
        "scan_raw": int(scan_raw),
        "map_raw": int(map_raw),
        "scan_filtered": int(scan_filtered),
        "map_filtered": int(map_filtered),
        "scan_filtered_ratio": scan_filtered / scan_raw,
        "map_filtered_ratio": map_filtered / map_raw,
        "source_raw_points": int(len(source_raw)),
        "target_submap_points": int(len(target)),
        "source_gicp_points": int(len(source_gicp)),
        "source_bbox_x": float(source_bbox[0]),
        "source_bbox_y": float(source_bbox[1]),
        "source_bbox_z": float(source_bbox[2]),
        "target_bbox_x": float(target_bbox[0]),
        "target_bbox_y": float(target_bbox[1]),
        "target_bbox_z": float(target_bbox[2]),
        "centroid_shift_init_to_gicp": centroid_shift,
        "centroid_distance_after_gicp": centroid_distance_after,
        "pcm_status": row.get("pcm_status", ""),
        "entered_pcm": _safe_int(row, "entered_pcm"),
        "final_used": _safe_int(row, "final_used"),
        "loop_symbol_pair": row.get("loop_symbol_pair", ""),
    }


def build_gicp_evidence_summary(pairs):
    return [analyze_gicp_pair(row) for row in pairs]


def write_gicp_evidence_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "scenario",
        "source_robot",
        "source_key",
        "target_robot",
        "target_key",
        "label",
        "scene_type",
        "failure_type",
        "gicp_status",
        "fitness",
        "fitness_threshold",
        "converged",
        "scan_raw",
        "map_raw",
        "scan_filtered",
        "map_filtered",
        "scan_filtered_ratio",
        "map_filtered_ratio",
        "source_raw_points",
        "target_submap_points",
        "source_gicp_points",
        "source_bbox_x",
        "source_bbox_y",
        "source_bbox_z",
        "target_bbox_x",
        "target_bbox_y",
        "target_bbox_z",
        "centroid_shift_init_to_gicp",
        "centroid_distance_after_gicp",
        "pcm_status",
        "entered_pcm",
        "final_used",
        "loop_symbol_pair",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float) and math.isfinite(value):
                    value = f"{value:.6f}"
                out[field] = value
            writer.writerow(out)
