from pathlib import Path
import argparse
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .gicp_evidence import build_gicp_evidence_summary, write_gicp_evidence_summary
from .pcd_io import read_pcd_xyz
from .pcm_consistency import (
    analyze_pcm,
    load_transform_records,
    write_pcm_filter_summary,
    write_pcm_matrices,
)
from .plausibility_gate import apply_gate, summarize as summarize_gate, write_gate_summary
from .scan_context import (
    ScanContextConfig,
    build_scan_context_summary,
    load_gicp_passed_pairs,
    write_scan_context_candidates,
)


def resolve_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(path).resolve().parent
    for key in ["data_dir", "output_dir", "transforms"]:
        if key in cfg:
            p = Path(cfg[key])
            cfg[key] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    return cfg


def count(rows, predicate):
    return int(sum(1 for row in rows if predicate(row)))


def compute_metrics(pairs, pcm_rows):
    total = len(pairs)
    true_total = count(pairs, lambda r: int(r["label"]) == 1)
    false_total = count(pairs, lambda r: int(r["label"]) == 0)
    gicp_passed = count(pairs, lambda r: r.get("gicp_status", r.get("status", "")) == "passed")
    gicp_true = count(pairs, lambda r: int(r["label"]) == 1)
    gicp_false = count(pairs, lambda r: int(r["label"]) == 0)
    pcm_accepted = count(pcm_rows, lambda r: int(r["colrio_final_used"]) == 1)
    pcm_rejected_or_pending = count(pcm_rows, lambda r: int(r["colrio_final_used"]) == 0)
    pcm_accepted_true = count(pcm_rows, lambda r: int(r["colrio_final_used"]) == 1 and int(r["label"]) == 1)
    pcm_accepted_false = count(pcm_rows, lambda r: int(r["colrio_final_used"]) == 1 and int(r["label"]) == 0)
    false_rejected = count(pcm_rows, lambda r: int(r["colrio_final_used"]) == 0 and int(r["label"]) == 0)
    offline_accepted = count(pcm_rows, lambda r: int(r["offline_pcm_clique_member"]) == 1)
    offline_accepted_false = count(
        pcm_rows,
        lambda r: int(r["offline_pcm_clique_member"]) == 1 and int(r["label"]) == 0,
    )
    return {
        "dataset_policy": "Only GICP-passed compact evidence is retained; long1/long2 bags are not required.",
        "scan_context_candidate_count": total,
        "gicp_passed_candidates": gicp_passed,
        "gicp_passed_true_loop": gicp_true,
        "gicp_passed_false_or_hard_negative": gicp_false,
        "true_loop_total": true_total,
        "false_loop_or_hard_negative_total": false_total,
        "colrio_pcm_input_candidates": total,
        "colrio_pcm_accepted_final_used": pcm_accepted,
        "colrio_pcm_rejected_or_pending_not_used": pcm_rejected_or_pending,
        "colrio_pcm_accepted_true_loop": pcm_accepted_true,
        "colrio_pcm_accepted_false_positive": pcm_accepted_false,
        "true_loop_retention_rate": pcm_accepted_true / max(true_total, 1),
        "false_match_rejection_rate": false_rejected / max(false_total, 1),
        "false_match_acceptance_rate": pcm_accepted_false / max(false_total, 1),
        "offline_pcm_style_accepted": offline_accepted,
        "offline_pcm_style_accepted_false_positive": offline_accepted_false,
    }


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _downsample(points, max_points=18000):
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def save_pair_visualization(pair, output_path, title):
    evidence = Path(pair["evidence_abs"])
    target = _downsample(read_pcd_xyz(evidence / "target_submap.pcd"), 22000)
    source = _downsample(read_pcd_xyz(evidence / "source_gicp_in_target.pcd"), 12000)
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.scatter(target[:, 0], target[:, 1], s=0.25, c="#64748b", alpha=0.42, label="target submap")
    ax.scatter(source[:, 0], source[:, 1], s=0.35, c="#2563eb", alpha=0.72, label="source after GICP")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.22)
    ax.axis("equal")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_visualizations(pairs, output_dir):
    true_pair = next((p for p in pairs if int(p["label"]) == 1), None)
    hard_negative = next(
        (
            p
            for p in pairs
            if int(p["label"]) == 0
            and p.get("failure_type") == "pcm_accepted_false_positive"
        ),
        None,
    )
    if true_pair is not None:
        save_pair_visualization(
            true_pair,
            Path(output_dir) / "true_loop_visualization.png",
            f"True loop: {true_pair['pair_id']}",
        )
    if hard_negative is not None:
        save_pair_visualization(
            hard_negative,
            Path(output_dir) / "hard_negative_visualization.png",
            f"Hard negative / false loop: {hard_negative['pair_id']}",
        )


def run_analysis(config_path, data_dir=None, output_dir=None, transforms=None):
    cfg = resolve_config(config_path)
    if data_dir is not None:
        cfg["data_dir"] = data_dir
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if transforms is not None:
        cfg["transforms"] = transforms

    out_dir = Path(cfg.get("output_dir", "outputs/loop_matching_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_gicp_passed_pairs(cfg["data_dir"])
    if not pairs:
        raise SystemExit("No GICP-passed PCD evidence pairs found.")

    sc_cfg_raw = cfg.get("scan_context", {})
    sc_cfg = ScanContextConfig(
        num_rings=int(sc_cfg_raw.get("num_rings", 20)),
        num_sectors=int(sc_cfg_raw.get("num_sectors", 60)),
        max_radius_m=float(sc_cfg_raw.get("max_radius_m", 80.0)),
        distance_threshold=float(sc_cfg_raw.get("distance_threshold", cfg.get("distance_threshold", 0.4))),
        top_k=int(sc_cfg_raw.get("top_k", 20)),
    )
    pcm_cfg = cfg.get("pcm", {})
    pcm_threshold = float(pcm_cfg.get("se2_threshold", 15.0))
    min_clique_size = int(pcm_cfg.get("min_clique_size", 3))

    sc_rows = build_scan_context_summary(pairs, sc_cfg)
    write_scan_context_candidates(out_dir / "scan_context_candidates.csv", sc_rows)

    gicp_rows = build_gicp_evidence_summary(pairs)
    write_gicp_evidence_summary(out_dir / "gicp_evidence_summary.csv", gicp_rows)

    records = load_transform_records(cfg["transforms"])
    pcm_rows, pcm_matrices = analyze_pcm(records, threshold=pcm_threshold, min_clique_size=min_clique_size)
    write_pcm_filter_summary(out_dir / "pcm_filter_summary.csv", pcm_rows)
    write_pcm_matrices(out_dir / "pcm_pairwise_matrices.json", pcm_matrices)

    gate_cfg = cfg.get("plausibility_gate", {})
    gate_threshold = float(gate_cfg.get("max_translation_m", 40.0))
    gated_rows = apply_gate(pcm_rows, gate_threshold)
    write_gate_summary(out_dir / "plausibility_gate_summary.csv", gated_rows)
    gate_metrics = summarize_gate(gated_rows, gate_threshold)
    save_metrics(out_dir / "plausibility_gate_metrics.json", gate_metrics)

    metrics = compute_metrics(pairs, pcm_rows)
    metrics["scan_context_parameters"] = sc_cfg.__dict__
    metrics["pcm_style_parameters"] = {
        "se2_threshold": pcm_threshold,
        "min_clique_size": min_clique_size,
    }
    metrics["plausibility_gate"] = gate_metrics
    save_metrics(out_dir / "loop_filter_metrics.json", metrics)
    write_visualizations(pairs, out_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=str(root / "configs" / "gicp_passed_pcm.yaml"))
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--transforms")
    args = parser.parse_args()
    metrics = run_analysis(args.config, args.data_dir, args.output_dir, args.transforms)
    print("loop matching analysis complete")
    print(f"  candidates: {metrics['scan_context_candidate_count']}")
    print(f"  GICP-passed true/false: {metrics['gicp_passed_true_loop']}/{metrics['gicp_passed_false_or_hard_negative']}")
    print(f"  Co-LRIO PCM accepted false positives: {metrics['colrio_pcm_accepted_false_positive']}")
    print(f"  false match rejection rate: {metrics['false_match_rejection_rate']:.3f}")
    print(f"  false match acceptance rate: {metrics['false_match_acceptance_rate']:.3f}")
    gate = metrics["plausibility_gate"]
    print(
        "  + plausibility gate -> false acceptance "
        f"{gate['baseline_pcm_only']['false_acceptance_rate']:.3f}"
        f" -> {gate['pcm_plus_gate']['false_acceptance_rate']:.3f}"
        f" (true retention {gate['pcm_plus_gate']['true_retention_rate']:.2f})"
    )


if __name__ == "__main__":
    main()
