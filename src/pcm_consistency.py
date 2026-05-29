from pathlib import Path
import csv
import json
import itertools

import numpy as np


DEFAULT_YAW_WEIGHT_M_PER_RAD = 10.0


def yaw_from_R(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def angle_diff(a, b):
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def T_inverse(T):
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def compute_T_r1_in_r0(record):
    """Recover the inter-robot transform implied by one loop measurement."""
    T_world_src = np.asarray(record["T_world_source"], dtype=np.float64)
    T_world_tgt = np.asarray(record["T_world_target"], dtype=np.float64)
    T_gicp = np.asarray(record["T_gicp"], dtype=np.float64)
    if record["source_robot"] == "robot_0" and record["target_robot"] == "robot_1":
        return T_world_src @ T_inverse(T_gicp) @ T_inverse(T_world_tgt)
    if record["source_robot"] == "robot_1" and record["target_robot"] == "robot_0":
        return T_world_tgt @ T_gicp @ T_inverse(T_world_src)
    raise ValueError(f"unexpected robot pair: {record['source_robot']}-{record['target_robot']}")


def se2_distance(a, b, yaw_weight=DEFAULT_YAW_WEIGHT_M_PER_RAD):
    dxy = float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))
    dyaw = abs(angle_diff(float(a[2]), float(b[2])))
    return dxy + yaw_weight * dyaw


def build_pairwise_consistency_matrix(poses_se2, threshold):
    n = len(poses_se2)
    matrix = np.zeros((n, n), dtype=np.int32)
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = se2_distance(poses_se2[i], poses_se2[j])
            distances[i, j] = d
            matrix[i, j] = int(d <= threshold)
    return matrix, distances


def maximum_clique_bruteforce(consistency_matrix):
    """Small-data exact maximum clique. Scenario sizes are 10 loops here."""
    n = consistency_matrix.shape[0]
    best = []
    nodes = range(n)
    for size in range(n, 0, -1):
        for combo in itertools.combinations(nodes, size):
            ok = True
            for i, j in itertools.combinations(combo, 2):
                if consistency_matrix[i, j] != 1 or consistency_matrix[j, i] != 1:
                    ok = False
                    break
            if ok:
                return list(combo)
    return best


def load_transform_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_pcm(records, threshold=15.0, min_clique_size=3):
    rows = []
    matrix_by_scenario = {}
    for scenario in sorted({r["scenario"] for r in records}):
        scenario_records = [r for r in records if r["scenario"] == scenario]
        poses = []
        for rec in scenario_records:
            T = compute_T_r1_in_r0(rec)
            poses.append(np.array([T[0, 3], T[1, 3], yaw_from_R(T[:3, :3])], dtype=np.float64))
        consistency, distances = build_pairwise_consistency_matrix(poses, threshold)
        clique = maximum_clique_bruteforce(consistency)
        if len(clique) < min_clique_size:
            clique = []
        clique_set = set(clique)
        matrix_by_scenario[scenario] = {
            "pair_ids": [r["pair_id"] for r in scenario_records],
            "consistency_matrix": consistency.tolist(),
            "distance_matrix": distances.tolist(),
            "max_clique_pair_ids": [scenario_records[i]["pair_id"] for i in clique],
            "max_clique_size": len(clique),
        }
        for i, rec in enumerate(scenario_records):
            pose = poses[i]
            colrio_final_used = int(rec.get("final_used", 0))
            label = int(rec.get("label", 0))
            rows.append(
                {
                    "pair_id": rec["pair_id"],
                    "scenario": scenario,
                    "source_robot": rec["source_robot"],
                    "source_key": int(rec["source_key"]),
                    "target_robot": rec["target_robot"],
                    "target_key": int(rec["target_key"]),
                    "label": label,
                    "scene_type": rec.get("scene_type", ""),
                    "failure_type": rec.get("failure_type", ""),
                    "fitness": float(rec.get("fitness", 0.0)),
                    "colrio_pcm_status": rec.get("pcm_status", ""),
                    "colrio_final_used": colrio_final_used,
                    "colrio_accepted_false_positive": int(colrio_final_used == 1 and label == 0),
                    "implied_tx": float(pose[0]),
                    "implied_ty": float(pose[1]),
                    "implied_yaw_deg": float(np.degrees(pose[2])),
                    "pairwise_consistent_count": int(consistency[i].sum()),
                    "offline_pcm_clique_member": int(i in clique_set),
                    "offline_pcm_status": "accepted" if i in clique_set else "rejected_or_pending",
                    "offline_clique_size": len(clique),
                }
            )
    return rows, matrix_by_scenario


def write_pcm_filter_summary(path, rows):
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
        "fitness",
        "colrio_pcm_status",
        "colrio_final_used",
        "colrio_accepted_false_positive",
        "implied_tx",
        "implied_ty",
        "implied_yaw_deg",
        "pairwise_consistent_count",
        "offline_pcm_clique_member",
        "offline_pcm_status",
        "offline_clique_size",
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


def write_pcm_matrices(path, matrix_by_scenario):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(matrix_by_scenario, f, ensure_ascii=False, indent=2)
        f.write("\n")
