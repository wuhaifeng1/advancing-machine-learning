from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np

from .pcd_io import read_pcd_xyz


@dataclass
class ScanContextConfig:
    num_rings: int = 20
    num_sectors: int = 60
    max_radius_m: float = 80.0
    distance_threshold: float = 0.4
    top_k: int = 20


def clean_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)
    return points[np.isfinite(points).all(axis=1)]


def make_scan_context(points, config=None):
    """Build the Co-LRIO-style ring-sector Scan Context matrix."""
    cfg = config or ScanContextConfig()
    points = clean_points(points)
    desc = np.zeros((cfg.num_rings, cfg.num_sectors), dtype=np.float32)
    if len(points) == 0:
        return desc

    xy = points[:, :2]
    radius = np.linalg.norm(xy, axis=1)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    angle = np.where(angle < 0.0, angle + 2.0 * np.pi, angle)
    mask = radius <= cfg.max_radius_m
    if not np.any(mask):
        return desc

    radius = radius[mask]
    angle = angle[mask]
    z = points[mask, 2]
    ring = np.ceil((radius / cfg.max_radius_m) * cfg.num_rings).astype(np.int32)
    sector = np.ceil((angle / (2.0 * np.pi)) * cfg.num_sectors).astype(np.int32)
    ring = np.maximum(np.minimum(ring, cfg.num_rings), 1) - 1
    sector = np.maximum(np.minimum(sector, cfg.num_sectors), 1) - 1

    for r, s, h in zip(ring, sector, z + 1.0):
        if h > desc[r, s]:
            desc[r, s] = h
    return desc


def make_ring_key(context):
    return np.asarray(context, dtype=np.float32).mean(axis=1)


def make_sector_key(context):
    return np.asarray(context, dtype=np.float32).mean(axis=0)


def _circshift_cols(mat, shift):
    return np.roll(mat, int(shift), axis=1)


def _sector_distance(sc1, sc2):
    valid = (np.linalg.norm(sc1, axis=0) > 0.0) & (np.linalg.norm(sc2, axis=0) > 0.0)
    if not np.any(valid):
        return 1.0
    a = sc1[:, valid]
    b = sc2[:, valid]
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    denom = np.maximum(denom, 1e-12)
    similarity = np.sum(a * b, axis=0) / denom
    return float(1.0 - np.mean(similarity))


def fast_align_using_sector_key(vkey1, vkey2):
    best_shift = 0
    best_norm = float("inf")
    for shift in range(len(vkey1)):
        diff = vkey1 - np.roll(vkey2, shift)
        norm = float(np.linalg.norm(diff))
        if norm < best_norm:
            best_shift = shift
            best_norm = norm
    return best_shift


def distance_between_contexts(context_a, context_b):
    """Return (distance, best_sector_shift), matching Co-LRIO's search pattern."""
    sc1 = np.asarray(context_a, dtype=np.float32)
    sc2 = np.asarray(context_b, dtype=np.float32)
    vkey1 = make_sector_key(sc1)
    vkey2 = make_sector_key(sc2)
    center_shift = fast_align_using_sector_key(vkey1, vkey2)
    search_radius = int(round(0.5 * 0.1 * sc1.shape[1]))
    shifts = [center_shift]
    for offset in range(1, search_radius + 1):
        shifts.append((center_shift + offset + sc1.shape[1]) % sc1.shape[1])
        shifts.append((center_shift - offset + sc1.shape[1]) % sc1.shape[1])
    shifts = sorted(set(shifts))

    best_shift = 0
    best_distance = float("inf")
    for shift in shifts:
        dist = _sector_distance(sc1, _circshift_cols(sc2, shift))
        if dist < best_distance:
            best_shift = shift
            best_distance = dist
    return float(best_distance), int(best_shift)


def load_gicp_passed_pairs(data_dir):
    rows = []
    data_dir = Path(data_dir)
    for scenario_dir in sorted(data_dir.glob("scenario_*")):
        pair_csv = scenario_dir / "loop_pairs.csv"
        if not pair_csv.exists():
            continue
        with open(pair_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pcd_saved") != "1":
                    continue
                row = dict(row)
                row["scenario_dir"] = str(scenario_dir)
                row["evidence_abs"] = str(scenario_dir / row["evidence_dir"])
                rows.append(row)
    return rows


def analyze_pair_scan_context(row, config=None):
    cfg = config or ScanContextConfig()
    evidence = Path(row["evidence_abs"])
    target = read_pcd_xyz(evidence / "target_submap.pcd")
    source_raw = read_pcd_xyz(evidence / "source_raw.pcd")
    source_init = read_pcd_xyz(evidence / "source_init_in_target.pcd")
    source_gicp = read_pcd_xyz(evidence / "source_gicp_in_target.pcd")

    target_sc = make_scan_context(target, cfg)
    raw_sc = make_scan_context(source_raw, cfg)
    init_sc = make_scan_context(source_init, cfg)
    gicp_sc = make_scan_context(source_gicp, cfg)

    raw_distance, raw_shift = distance_between_contexts(raw_sc, target_sc)
    init_distance, init_shift = distance_between_contexts(init_sc, target_sc)
    gicp_distance, gicp_shift = distance_between_contexts(gicp_sc, target_sc)
    source_ring = make_ring_key(gicp_sc)
    target_ring = make_ring_key(target_sc)
    ring_distance = float(np.linalg.norm(source_ring - target_ring))

    return {
        "pair_id": row["pair_id"],
        "scenario": row["scenario"],
        "query_robot": row["source_robot"],
        "query_key": int(row["source_key"]),
        "candidate_robot": row["target_robot"],
        "candidate_key": int(row["target_key"]),
        "label": int(row["label"]),
        "scene_type": row.get("scene_type", ""),
        "failure_type": row.get("failure_type", ""),
        "gicp_status": row.get("gicp_status", row.get("status", "")),
        "pcm_status": row.get("pcm_status", ""),
        "final_used": int(row.get("final_used") or 0),
        "sc_distance_raw_to_target": raw_distance,
        "sc_shift_raw_to_target": raw_shift,
        "sc_distance_init_to_target": init_distance,
        "sc_shift_init_to_target": init_shift,
        "sc_distance_gicp_to_target": gicp_distance,
        "sc_shift_gicp_to_target": gicp_shift,
        "ring_key_l2_gicp_to_target": ring_distance,
        "within_threshold_after_gicp": int(gicp_distance < cfg.distance_threshold),
    }


def build_scan_context_summary(pairs, config=None):
    rows = [analyze_pair_scan_context(row, config) for row in pairs]
    grouped = {}
    for row in rows:
        key = (row["scenario"], row["query_robot"], row["query_key"])
        grouped.setdefault(key, []).append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r["sc_distance_gicp_to_target"])
        for rank, row in enumerate(group, start=1):
            row["candidate_rank_within_query"] = rank
    return rows


def write_scan_context_candidates(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "scenario",
        "query_robot",
        "query_key",
        "candidate_robot",
        "candidate_key",
        "candidate_rank_within_query",
        "sc_distance_raw_to_target",
        "sc_shift_raw_to_target",
        "sc_distance_init_to_target",
        "sc_shift_init_to_target",
        "sc_distance_gicp_to_target",
        "sc_shift_gicp_to_target",
        "ring_key_l2_gicp_to_target",
        "within_threshold_after_gicp",
        "label",
        "scene_type",
        "failure_type",
        "gicp_status",
        "pcm_status",
        "final_used",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                out[field] = value
            writer.writerow(out)
