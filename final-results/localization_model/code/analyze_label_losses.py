#!/usr/bin/env python3
"""Decompose frozen Pipeline24 event-decoder label losses and plot the flow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
P24 = HERE.parent
RESULTS = P24.parent
OUTPUT = HERE / "label_loss_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def overlaps_truth(frame: pd.DataFrame, truth_row: pd.Series, p24, cutoff: float = 0.5) -> bool:
    possible = frame[
        (frame.sample_id.astype(str) == str(truth_row.sample_id))
        & (frame.chrom.astype(str) == str(truth_row.chrom))
    ]
    return any(
        p24.overlap_coefficient(
            int(row.start), int(row.end) + 1,
            int(truth_row.start), int(truth_row.end) + 1,
        ) >= cutoff
        for row in possible.itertuples()
    )


def geometry_table(events: pd.DataFrame, selected: pd.Series, p24) -> pd.DataFrame:
    frame = events.copy()
    if str(selected.region_mode) == "representative":
        source_start, source_end = frame.start, frame.end
    elif str(selected.region_mode) == "envelope":
        source_start, source_end = frame.envelope_start, frame.envelope_end
    else:
        raise ValueError(selected.region_mode)
    intervals = [
        p24.scale_interval(int(start), int(end), float(selected.scale))
        for start, end in zip(source_start, source_end, strict=True)
    ]
    frame["start"] = [value[0] for value in intervals]
    frame["end"] = [value[1] - 1 for value in intervals]
    return frame


def maximum_matching_ids(predictions: pd.DataFrame, truth: pd.DataFrame, p24) -> set[str]:
    """Return caller IDs retained by the same maximum-cardinality criterion."""
    matched_ids: set[str] = set()
    for keys, calls in truth.groupby(["sample_id", "chrom", "label"], sort=False):
        possible = predictions.copy()
        for column, value in zip(["sample_id", "chrom", "label"], keys):
            possible = possible[possible[column].astype(str) == str(value)]
        if possible.empty:
            continue
        prediction_rows = list(possible.itertuples())
        call_rows = list(calls.itertuples())
        edges = []
        for prediction in prediction_rows:
            edges.append([
                index for index, call in enumerate(call_rows)
                if p24.overlap_coefficient(
                    int(prediction.start), int(prediction.end) + 1,
                    int(call.start), int(call.end) + 1,
                ) >= 0.5
            ])
        matched_call = [-1] * len(call_rows)

        def augment(prediction_index: int, seen: list[bool]) -> bool:
            for call_index in edges[prediction_index]:
                if seen[call_index]:
                    continue
                seen[call_index] = True
                if matched_call[call_index] < 0 or augment(matched_call[call_index], seen):
                    matched_call[call_index] = prediction_index
                    return True
            return False

        for prediction_index in range(len(prediction_rows)):
            augment(prediction_index, [False] * len(call_rows))
        for call_index, prediction_index in enumerate(matched_call):
            if prediction_index >= 0:
                matched_ids.add(str(call_rows[call_index].region_id))
    return matched_ids


def draw_flow(losses: pd.DataFrame, output_path: Path) -> None:
    order = [
        ("proposal_miss", "Candidate\nproposal"),
        ("event_geometry", "Event\ngeometry"),
        ("score_threshold", "Class score\nthreshold"),
        ("nms", "Cross-cluster\nNMS"),
        ("output_cap", "Per-sample\noutput cap"),
        ("one_to_one_collision", "One-to-one\nassignment"),
    ]
    colors = {
        "proposal_miss": "#9C755F",
        "event_geometry": "#BAB0AC",
        "score_threshold": "#E15759",
        "nms": "#F28E2B",
        "output_cap": "#EDC948",
        "one_to_one_collision": "#B07AA1",
        "recovered": "#59A14F",
    }
    total = len(losses)
    retained = [total]
    for key, _ in order:
        retained.append(retained[-1] - int((losses.loss_stage == key).sum()))

    fig = plt.figure(figsize=(16, 9.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.34)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(-0.4, len(retained) - 0.6)
    ax.set_ylim(-1.15, 1.2)
    ax.axis("off")
    stage_names = ["Caller\nlabels"] + [name for _, name in order]
    for index, (name, count) in enumerate(zip(stage_names, retained, strict=True)):
        face = "#4E79A7" if index == 0 else ("#59A14F" if index == len(retained) - 1 else "#DCE6F2")
        box = FancyBboxPatch(
            (index - 0.35, -0.22), 0.70, 0.58,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=face, edgecolor="#34495E", linewidth=1.2,
        )
        ax.add_patch(box)
        text_color = "white" if index in {0, len(retained) - 1} else "#1F2933"
        ax.text(index, 0.12, f"{count}", ha="center", va="center", fontsize=17, fontweight="bold", color=text_color)
        ax.text(index, -0.52, name, ha="center", va="top", fontsize=10.5)
        if index < len(retained) - 1:
            ax.add_patch(FancyArrowPatch(
                (index + 0.36, 0.07), (index + 0.64, 0.07),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#566573",
            ))
            key = order[index][0]
            lost = int((losses.loss_stage == key).sum())
            ax.text(
                index + 0.5, 0.63, f"−{lost}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=colors[key],
            )
    ax.text(0, 1.02, "Where frozen Pipeline24 event decoding loses labels", fontsize=17, fontweight="bold", ha="left")
    ax.text(0, 0.80, "Counts are exclusive first-failure stages; overlap coefficient threshold = 0.5", fontsize=10.5, color="#52616B", ha="left")
    threshold = losses[losses.loss_stage == "score_threshold"]
    wrong_class = int((threshold.threshold_subtype == "localized_wrong_class").sum())
    no_call = int((threshold.threshold_subtype == "no_surviving_localization").sum())
    geometry = losses[losses.loss_stage == "event_geometry"]
    scale_loss = int(geometry.representative_unscaled_available.sum())
    representative_loss = int((~geometry.representative_unscaled_available & geometry.envelope_unscaled_available).sum())
    envelope_loss = int((~geometry.envelope_unscaled_available).sum())
    ax.text(
        3.0, -1.01,
        f"Threshold losses: {no_call} no surviving localization + {wrong_class} wrong-class calls     |     "
        f"Geometry losses: {representative_loss} representative choice + {scale_loss} scaling + {envelope_loss} cluster envelope",
        ha="center", va="center", fontsize=9.5, color="#455A64",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F9FA", edgecolor="#CFD8DC"),
    )

    ax2 = fig.add_subplot(grid[1])
    class_order = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
    display = {"seismic_amplification": "seismic amp."}
    stack_order = [key for key, _ in order] + ["recovered"]
    left = np.zeros(len(class_order))
    for stage in stack_order:
        values = np.array([
            int(((losses.truth_label == label) & (losses.loss_stage == stage)).sum())
            for label in class_order
        ])
        ax2.barh(
            np.arange(len(class_order)), values, left=left,
            color=colors[stage], edgecolor="white", linewidth=0.7,
            label=stage.replace("_", " "),
        )
        for row, (value, offset) in enumerate(zip(values, left, strict=True)):
            if value:
                ax2.text(offset + value / 2, row, str(value), ha="center", va="center", fontsize=9, color="white" if stage not in {"event_geometry", "output_cap"} else "#263238", fontweight="bold")
        left += values
    ax2.set_yticks(np.arange(len(class_order)), [display.get(label, label) for label in class_order])
    ax2.invert_yaxis()
    ax2.set_xlabel("Caller labels")
    ax2.set_title("Exclusive loss stage by class", loc="left", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.15)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p24 = load_module("p24", P24 / "four_class_f2_recommended.py")
    frozen = load_module("frozen_event", HERE / "run_one.py")
    config = json.loads((P24 / "config_recommended.json").read_text())
    candidates = pd.read_csv(RESULTS / "pipeline18/merged_candidate_regions.csv")
    truth = pd.read_csv(P24 / "labels.tsv", sep="\t")
    features, ids = p24.load_features(
        str(RESULTS / "pipeline18/candidate_region_classifier_general/embeddings.npz"),
        str(RESULTS / "pipeline18/candidate_region_classifier_general/selected_embedding_features.npz"),
        str(RESULTS / "pipeline18/candidate_region_classifier_general/tabular_features.npz"),
        candidates,
    )
    candidates = candidates.set_index("candidate_id").loc[ids].reset_index()
    all_final = []
    audit_rows = []

    for sample in sorted(truth.sample_id.astype(str).unique()):
        checkpoint = torch.load(P24 / "loo_check/runs" / sample / "model.pt", map_location="cpu", weights_only=False)
        model = p24.ProposalBagModel(int(checkpoint["input_dim"]), int(config["hidden_dim"]), float(config["dropout"]))
        model.load_state_dict(checkpoint["model"])
        normalized = np.clip((features - checkpoint["feature_mean"]) / checkpoint["feature_std"], -8, 8).astype(np.float32)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(normalized))).numpy()
        sample_mask = candidates.sample_id.astype(str) == sample
        sample_candidates = candidates[sample_mask]
        selected_table = pd.read_csv(HERE / "runs" / sample / "calibration_selected.tsv", sep="\t").set_index("label")
        sample_truth = truth[truth.sample_id.astype(str) == sample]
        final_by_class = {}
        stage_by_class = {}
        for class_index, label in enumerate(p24.CLASSES):
            events = frozen.prepare_events(sample_candidates, scores[sample_mask.to_numpy(), class_index])
            selected = selected_table.loc[label]
            geometry = geometry_table(events, selected, p24)
            thresholded = geometry[geometry.score >= float(selected.threshold)].copy()
            after_nms = frozen.decode_events(
                p24, events, label, float(selected.threshold), float(selected.scale),
                str(selected.region_mode), str(selected.nms_mode), 999999,
            )
            final = frozen.decode_events(
                p24, events, label, float(selected.threshold), float(selected.scale),
                str(selected.region_mode), str(selected.nms_mode), int(selected.maximum_per_sample),
            )
            final_by_class[label] = final
            stage_by_class[label] = (events, geometry, thresholded, after_nms, final)
            all_final.append(final)

        for _, call in sample_truth.iterrows():
            label = str(call.label)
            raw = sample_candidates[sample_candidates.chrom.astype(str) == str(call.chrom)].rename(columns={"start": "raw_start", "end": "raw_end"})
            proposal_available = any(
                p24.overlap_coefficient(int(row.raw_start), int(row.raw_end) + 1, int(call.start), int(call.end) + 1) >= 0.5
                for row in raw.itertuples()
            )
            events, geometry, thresholded, after_nms, final = stage_by_class[label]
            envelope_unscaled = events.copy()
            envelope_unscaled["start"] = envelope_unscaled.envelope_start
            envelope_unscaled["end"] = envelope_unscaled.envelope_end
            representative_unscaled_available = overlaps_truth(events, call, p24)
            envelope_unscaled_available = overlaps_truth(envelope_unscaled, call, p24)
            geometry_available = overlaps_truth(geometry, call, p24)
            threshold_available = overlaps_truth(thresholded, call, p24)
            nms_available = overlaps_truth(after_nms, call, p24)
            final_available = overlaps_truth(final, call, p24)
            if not proposal_available:
                stage = "proposal_miss"
            elif not geometry_available:
                stage = "event_geometry"
            elif not threshold_available:
                stage = "score_threshold"
            elif not nms_available:
                stage = "nms"
            elif not final_available:
                stage = "output_cap"
            else:
                stage = "pending_assignment"
            audit_rows.append({
                "region_id": call.region_id, "sample_id": sample, "chrom": call.chrom,
                "truth_start": call.start, "truth_end": call.end, "truth_label": label,
                "proposal_available": proposal_available,
                "representative_unscaled_available": representative_unscaled_available,
                "envelope_unscaled_available": envelope_unscaled_available,
                "event_geometry_available": geometry_available,
                "threshold_available": threshold_available,
                "nms_available": nms_available,
                "output_cap_available": final_available,
                "loss_stage": stage,
            })

    final_predictions = pd.concat(all_final, ignore_index=True)
    matched_ids = maximum_matching_ids(final_predictions, truth, p24)
    audit = pd.DataFrame(audit_rows)
    pending = audit.loss_stage == "pending_assignment"
    audit.loc[pending & audit.region_id.astype(str).isin(matched_ids), "loss_stage"] = "recovered"
    audit.loc[pending & ~audit.region_id.astype(str).isin(matched_ids), "loss_stage"] = "one_to_one_collision"
    localized_by_any_class = []
    for row in audit.itertuples():
        call = pd.Series({
            "sample_id": row.sample_id, "chrom": row.chrom,
            "start": row.truth_start, "end": row.truth_end,
        })
        localized_by_any_class.append(overlaps_truth(final_predictions, call, p24))
    audit["localized_by_any_class"] = localized_by_any_class
    audit["threshold_subtype"] = ""
    threshold_mask = audit.loss_stage == "score_threshold"
    audit.loc[threshold_mask & audit.localized_by_any_class, "threshold_subtype"] = "localized_wrong_class"
    audit.loc[threshold_mask & ~audit.localized_by_any_class, "threshold_subtype"] = "no_surviving_localization"
    audit.to_csv(OUTPUT / "label_loss_assignments.tsv", sep="\t", index=False)
    summary = audit.groupby(["truth_label", "loss_stage"]).size().rename("labels").reset_index()
    summary.to_csv(OUTPUT / "loss_by_class.tsv", sep="\t", index=False)
    overall = audit.groupby("loss_stage").size().rename("labels").reset_index()
    overall.to_csv(OUTPUT / "loss_overall.tsv", sep="\t", index=False)
    draw_flow(audit, OUTPUT / "label_loss_pipeline.png")

    if len(matched_ids) != 53:
        raise AssertionError(f"expected 53 one-to-one matches, reconstructed {len(matched_ids)}")
    print(overall.to_string(index=False))
    print("\n", summary.to_string(index=False))
    print(f"\nReconstructed {len(matched_ids)}/108 one-to-one matches")


if __name__ == "__main__":
    main()
