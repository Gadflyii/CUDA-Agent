#!/usr/bin/env python3
"""Build canonical agent trajectories from closed campaign ledgers and Git evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from common import ROOT, jsonl_bytes, latest_raw_record, load_jsonl, raw_manifest, utc_now, write_generated


PRODUCER = "scripts/build_campaign_episodes.py@1"
OUTPUT = "normalized/campaign_episodes.jsonl"
REPORT = "reports/trajectory_builder.json"

EMPTY_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "not-applicable",
    "not applicable",
    "not-available",
    "unknown",
}
CODE_PATH_PREFIXES = (
    "src/",
    "include/",
    "tests/",
    "bench/",
    "tools/",
    "cmake/",
    "CMakeLists.txt",
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|github_pat|sk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in EMPTY_VALUES else text


def first(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = clean(row.get(name))
        if value:
            return value
    return ""


def slug(value: str, *, fallback: str = "item") -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or fallback


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def latest_timestamp(values: Iterable[str]) -> str:
    candidates = [value for value in values if value]
    if not candidates:
        return utc_now()
    return max(candidates, key=parse_datetime)


def source_ref(source_id: str, fragment: str) -> str:
    return f"{source_id}#{slug(fragment)}"


def load_registry() -> dict[str, dict[str, Any]]:
    return {item["source_id"]: item for item in load_jsonl(ROOT / "sources/registry.jsonl")}


def collected_source(source_id: str, registry: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], Path]:
    if source_id not in registry:
        raise KeyError(f"trajectory source is not registered: {source_id}")
    registered = registry[source_id]
    if registered["source_class"] == "ginfer_live_reference" or registered["ingestion_status"] != "collected":
        raise ValueError(f"trajectory source is not immutable/collected: {source_id}")
    record = latest_raw_record(source_id)
    return record, ROOT / record["relative_path"]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def history_records(
    source_ids: list[str], registry: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    records: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    for source_id in source_ids:
        _, path = collected_source(source_id, registry)
        for item in load_jsonl(path):
            commit = clean(item.get("commit"))
            if not commit:
                continue
            previous = records.get(commit)
            if previous is not None and clean(previous.get("patch")) != clean(item.get("patch")):
                raise ValueError(f"conflicting immutable patches for commit {commit}")
            # A later source may repair a missing exact-revert edge without changing the patch.
            if previous is None or (not previous.get("reverts_commit") and item.get("reverts_commit")):
                records[commit] = item
                owners[commit] = source_id
    return records, owners


def resolve_commit(value: str, records: dict[str, dict[str, Any]]) -> str | None:
    candidate = clean(value).lower()
    if not candidate or not re.fullmatch(r"[0-9a-f]{7,40}", candidate):
        return None
    if candidate in records:
        return candidate
    matches = [commit for commit in records if commit.startswith(candidate)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous commit prefix {candidate}: {matches[:4]}")
    return candidate if len(candidate) == 40 else None


def split_patch(patch: str) -> list[tuple[str, str]]:
    segments = re.split(r"(?m)(?=^diff --git )", patch)
    result: list[tuple[str, str]] = []
    for segment in segments:
        if not segment.startswith("diff --git "):
            continue
        first_line = segment.splitlines()[0]
        match = re.match(r"diff --git a/(.+?) b/(.+)$", first_line)
        if not match:
            continue
        result.append((match.group(2), segment.rstrip() + "\n"))
    return result


def filter_code_patch(patch: str) -> str:
    return "".join(
        segment
        for path, segment in split_patch(patch)
        if path.startswith(CODE_PATH_PREFIXES)
    ).rstrip()


def preimage_context(patch: str) -> str:
    blocks: list[str] = []
    for path, segment in split_patch(patch):
        if not path.startswith(CODE_PATH_PREFIXES):
            continue
        lines = segment.splitlines()
        hunks: list[str] = []
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                hunks.append(line)
                continue
            if not in_hunk:
                continue
            if line.startswith("diff --git "):
                in_hunk = False
                continue
            if line.startswith(" ") or (line.startswith("-") and not line.startswith("---")):
                hunks.append(line)
        if hunks:
            blocks.append(f"### {path}\n" + "\n".join(hunks))
    return "\n\n".join(blocks)


def structural_fingerprint(patch: str, mechanism: str, files: str, op_family: str) -> str:
    if patch:
        normalized = []
        for line in patch.splitlines():
            if line.startswith("index "):
                continue
            line = re.sub(r"@@ -[^ ]+ \+[^ ]+ @@", "@@", line)
            line = re.sub(r"\b(?:sm|cc)[_-]?(?:86|89|120a?)\b", "sm_x", line, flags=re.IGNORECASE)
            line = re.sub(r"\b[0-9a-f]{40}\b", "<commit>", line)
            normalized.append(line.rstrip())
        material = "\n".join(normalized)
    else:
        material = "|".join((op_family, mechanism.lower(), files.lower()))
        material = re.sub(r"\b(?:sm|cc)[_-]?(?:86|89|120a?)\b", "sm_x", material)
        material = re.sub(r"\s+", " ", material).strip()
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def infer_op_family(text: str, default: str) -> str:
    lowered = text.lower()
    if default == "cross_kernel_consolidation":
        return default
    if "dflash2" in lowered and any(word in lowered for word in ("accept", "select", "sampling")):
        return "dflash2_accept"
    if any(word in lowered for word in ("gqa", "attention", "softmax", "key split")):
        return "gqa_attention"
    if any(word in lowered for word in ("recurrent", "replay", "gated delta", "gdn")):
        return "gated_delta_net"
    if "swiglu" in lowered:
        return "linear_swiglu"
    if "nvfp4" in lowered:
        return "nvfp4_linear"
    if "fp8" in lowered:
        return "fp8_linear"
    if "q4" in lowered:
        return "q4_linear"
    if "w8" in lowered or "q8" in lowered:
        return "w8_linear"
    if "convolution" in lowered:
        return "grouped_dynamic_convolution"
    return default


def cycle_and_candidate(row: dict[str, str], source_id: str, index: int) -> tuple[str, str, str]:
    cycle_candidate = first(row, "cycle_candidate")
    cycle = first(row, "cycle")
    candidate = first(row, "candidate")
    order = first(row, "order")
    if cycle_candidate:
        parts = cycle_candidate.split(".", 1)
        cycle = cycle or parts[0]
        candidate = candidate or (parts[1] if len(parts) == 2 else cycle_candidate)
        order = order or (
            f"c{cycle}_{int(candidate):02d}" if candidate.isdigit() else cycle_candidate
        )
    if not cycle:
        match = re.search(r"(?:^|-)cycle(\d+)(?:-|$)", source_id)
        if match:
            cycle = match.group(1)
    if not cycle:
        match = re.match(r"c(?:c)?(\d+)_", order or candidate)
        if match:
            cycle = match.group(1)
    cycle = cycle or "unknown"
    candidate = candidate or str(index)
    order = order or (
        f"c{cycle}_{int(candidate):02d}" if candidate.isdigit() else candidate
    )
    return cycle, candidate, order


def normalize_decision(value: str) -> str:
    lowered = clean(value).lower()
    if lowered.startswith("accept"):
        return "accepted"
    if lowered.startswith("reject"):
        return "rejected"
    return "inconclusive"


def correctness(value: str) -> str:
    lowered = clean(value).lower()
    if not lowered:
        return "not_observed"
    if lowered.startswith("fail") or "oracle fail" in lowered:
        return "failed"
    if lowered.startswith("pass") or " pass" in lowered:
        return "passed"
    if "not run" in lowered or "not-run" in lowered:
        return "not_run"
    return "not_observed"


def strip_embedded_decision(value: str) -> str:
    return re.sub(r"^(?:accept|accepted|reject|rejected)\s*:\s*", "", clean(value), flags=re.IGNORECASE)


def format_program_result(row: dict[str, str]) -> str:
    direct = first(row, "real_boundary_result", "real_boundary")
    if direct:
        return strip_embedded_decision(direct)
    fields = []
    boundary = first(row, "program_boundary")
    if boundary:
        fields.append(boundary)
    baseline = first(row, "program_baseline_ms")
    if baseline:
        fields.append(f"baseline {baseline} ms")
    for rep in (1, 2):
        measured = first(row, f"program_rep{rep}_ms")
        delta = first(row, f"program_rep{rep}_delta_pct")
        if measured:
            suffix = f" ({delta}% improvement)" if delta else ""
            fields.append(f"rep{rep} {measured} ms{suffix}")
    return "; ".join(fields)


def correction_matches(correction: dict[str, str], cycle: str, candidate: str, order: str) -> bool:
    corrected_candidate = first(correction, "candidate", "affected_candidate")
    if not corrected_candidate:
        return False
    identities = {candidate.lower(), order.lower()}
    if candidate.isdigit() and cycle.isdigit():
        identities.add(f"c{cycle}_{int(candidate):02d}")
        identities.add(f"cc{cycle}_{int(candidate):02d}")
    if corrected_candidate.lower() not in identities:
        return False
    corrected_cycle = first(correction, "cycle", "discovered_cycle")
    return not corrected_cycle or corrected_cycle == cycle


def build_correction_text(rows: list[tuple[str, dict[str, str]]]) -> str:
    blocks = []
    for source_id, row in rows:
        parts = [f"Correction from {source_id}."]
        for name in (
            "issue",
            "original_claim",
            "corrected_claim",
            "corrected_public_baseline",
            "corrected_public_candidate",
            "corrected_projection",
            "corrected_decision",
            "decision_effect",
            "impact",
        ):
            value = clean(row.get(name))
            if value:
                parts.append(f"{name.replace('_', ' ')}: {value}")
        blocks.append(" ".join(parts))
    return "\n".join(blocks)


def build_candidate(
    campaign: dict[str, Any],
    source_id: str,
    row: dict[str, str],
    index: int,
    histories: dict[str, dict[str, Any]],
    history_owners: dict[str, str],
    corrections: list[tuple[str, dict[str, str]]],
    registry: dict[str, dict[str, Any]],
    raw_records: dict[str, dict[str, Any]],
    limits: dict[str, int],
) -> dict[str, Any]:
    cycle, candidate, order = cycle_and_candidate(row, source_id, index)
    cycle_assignment = campaign.get("cycle_assignments", {}).get(cycle, {})
    assigned_split = cycle_assignment.get("assigned_split", campaign["assigned_split"])
    evaluation_tier = cycle_assignment.get("evaluation_tier", campaign["evaluation_tier"])
    if assigned_split not in {"train", "validation", "test", "quarantine"}:
        raise ValueError(
            f"invalid split {assigned_split!r} for {campaign['campaign_id']} cycle {cycle}"
        )
    if evaluation_tier not in {"train", "in_arch", "cross_arch", "real_hardware", "quarantine"}:
        raise ValueError(
            f"invalid evaluation tier {evaluation_tier!r} for "
            f"{campaign['campaign_id']} cycle {cycle}"
        )
    mechanism = first(row, "mechanism", "changed_mechanism", "package_hypothesis", "scope")
    hypothesis = first(row, "hypothesis", "package_hypothesis")
    prediction = first(row, "predicted_effect")
    rejection = first(row, "rejection_observation")
    dispatch = first(row, "dispatch_domain", "dispatch_domains", "scope")
    files = first(row, "files")
    public_baseline = first(row, "public_op_baseline", "combined_public_baseline")
    public_candidate = first(
        row,
        "public_op_candidate",
        "combined_public_candidate",
        "public_op_result",
        "public_or_profile",
        "result",
    )
    public_candidate = strip_embedded_decision(public_candidate)
    program_result = format_program_result(row)
    oracle_result = first(row, "oracle_result", "oracle")
    compile_result = first(row, "compile_result")
    seams = first(row, "unaffected_seams")
    sentinels = first(row, "engine_sentinels")
    raw_decision = first(row, "decision")
    matching_corrections = [
        (correction_source, correction)
        for correction_source, correction in corrections
        if correction_matches(correction, cycle, candidate, order)
    ]
    for _, correction in matching_corrections:
        corrected_baseline = first(correction, "corrected_public_baseline")
        corrected_candidate = first(correction, "corrected_public_candidate")
        if corrected_baseline:
            public_baseline = corrected_baseline
        if corrected_candidate:
            public_candidate = corrected_candidate
        corrected = first(correction, "corrected_decision")
        if corrected:
            raw_decision = corrected
    disposition = normalize_decision(raw_decision)
    corrected_reason = " ".join(
        first(correction, "corrected_projection", "corrected_claim", "impact", "decision_effect")
        for _, correction in matching_corrections
    ).strip()
    reason = corrected_reason or first(row, "reason") or public_candidate or program_result or raw_decision

    raw_candidate_commit = first(row, "candidate_commit", "commit")
    raw_restoration_commit = first(row, "restoration_commit", "restoration")
    candidate_commit = resolve_commit(raw_candidate_commit, histories)
    restoration_commit = resolve_commit(raw_restoration_commit, histories)
    limitations: list[str] = []

    linked_candidate = candidate_commit
    linked_restoration = restoration_commit
    if disposition == "rejected" and candidate_commit:
        restoration_record = histories.get(restoration_commit or "")
        if not restoration_record or restoration_record.get("reverts_commit") != candidate_commit:
            limitations.append("The ledger disposition is usable for judgment, but immutable Git evidence does not prove the candidate/restoration edge.")
            linked_candidate = None
            linked_restoration = None
    if raw_candidate_commit and candidate_commit is None:
        limitations.append("The ledger candidate commit could not be resolved uniquely in the declared immutable history.")

    history = histories.get(candidate_commit or "")
    patch = filter_code_patch(clean(history.get("patch")) if history else "")
    preimage = preimage_context(patch)
    if patch and len(patch) > limits["max_patch_chars"]:
        limitations.append("The exact code patch exceeds the configured implementation-target limit; no implementation view is emitted.")
        patch = ""
        preimage = ""
    if preimage and len(preimage) > limits["max_preimage_chars"]:
        limitations.append("The reconstructed pre-change source context exceeds the configured limit; no implementation view is emitted.")
        patch = ""
        preimage = ""

    op_text = " ".join((mechanism, hypothesis, dispatch, files, first(row, "component_mechanisms")))
    op_family = infer_op_family(op_text, campaign["default_op_family"])
    fingerprint = structural_fingerprint(patch, mechanism, files, op_family)

    source_ids = [
        source_id,
        *campaign["context_source_ids"],
        *(source_id for source_id, _ in matching_corrections),
    ]
    history_source = history_owners.get(candidate_commit or "")
    restoration_history_source = history_owners.get(restoration_commit or "")
    if history_source:
        source_ids.append(history_source)
    if restoration_history_source:
        source_ids.append(restoration_history_source)
    source_ids = list(dict.fromkeys(source_ids))
    snapshots = [raw_records[item]["record_id"] for item in source_ids]
    evidence_cutoff = latest_timestamp(
        [
            clean(registry[item].get("source_date")) or raw_records[item]["created_at"]
            for item in source_ids
        ]
    )

    correction_text = build_correction_text(matching_corrections)
    correction_refs = [source_ref(item, order) for item, _ in matching_corrections]
    ledger_ref = source_ref(source_id, order)
    history_ref = source_ref(history_source, candidate_commit) if history_source and candidate_commit else ledger_ref
    restoration_ref = (
        source_ref(restoration_history_source, restoration_commit)
        if restoration_history_source and restoration_commit
        else ledger_ref
    )
    performance_parts = []
    if public_candidate:
        performance_parts.append(f"public Op: {public_candidate}")
    if program_result:
        performance_parts.append(f"real boundary: {program_result}")
    if correction_text:
        performance_parts.append(f"correction: {correction_text}")

    return {
        "campaign": campaign,
        "campaign_id": campaign["campaign_id"],
        "cycle": cycle,
        "candidate": candidate,
        "order": order,
        "mechanism": mechanism or dispatch or order,
        "hypothesis": hypothesis,
        "prediction": prediction,
        "rejection": rejection,
        "dispatch": dispatch,
        "files": files,
        "public_baseline": public_baseline,
        "public_candidate": public_candidate,
        "program_baseline": first(row, "program_baseline_ms"),
        "program_result": program_result,
        "compile_result": compile_result,
        "oracle_result": oracle_result,
        "seams": seams,
        "sentinels": sentinels,
        "disposition": disposition,
        "reason": reason,
        "correctness": correctness(oracle_result),
        "candidate_commit": linked_candidate,
        "restoration_commit": linked_restoration,
        "raw_candidate_commit": raw_candidate_commit,
        "raw_restoration_commit": raw_restoration_commit,
        "patch": patch,
        "preimage": preimage,
        "op_family": op_family,
        "fingerprint": fingerprint,
        "source_ids": source_ids,
        "snapshot_refs": snapshots,
        "evidence_cutoff": evidence_cutoff,
        "ledger_ref": ledger_ref,
        "history_ref": history_ref,
        "restoration_ref": restoration_ref,
        "correction_text": correction_text,
        "correction_refs": correction_refs,
        "performance_claim": "; ".join(performance_parts) or "No performance measurement was observed.",
        "limitations": limitations,
        "assigned_split": assigned_split,
        "evaluation_tier": evaluation_tier,
    }


def event(
    events: list[dict[str, Any]],
    actor: str,
    event_type: str,
    information_class: str,
    payload: Any,
    refs: list[str],
) -> str:
    event_id = f"e{len(events):03d}"
    events.append(
        {
            "event_id": event_id,
            "sequence": len(events),
            "actor": actor,
            "event_type": event_type,
            "information_class": information_class,
            "payload": payload,
            "evidence_refs": list(dict.fromkeys(refs)),
        }
    )
    return event_id


def scope(candidate: dict[str, Any]) -> dict[str, Any]:
    campaign = candidate["campaign"]
    return {
        "repository": "GInfer",
        "target": campaign["target"],
        "boundary": campaign["boundary"],
        "op_family": candidate["op_family"],
        "architectures": [campaign["architecture"]],
        "hardware_skus": [campaign["hardware_sku"]],
        "workload": campaign["workload"],
    }


def context_payload(candidate: dict[str, Any]) -> str:
    campaign = candidate["campaign"]
    parts = [
        f"Target: {campaign['target']}.",
        f"Hardware: {campaign['hardware_sku']} ({campaign['architecture']}).",
        f"Boundary: {campaign['boundary']}.",
        f"Workload: {json.dumps(campaign['workload'], sort_keys=True)}.",
    ]
    if candidate["dispatch"]:
        parts.append(f"Dispatch scope: {candidate['dispatch']}.")
    if candidate["files"]:
        parts.append(f"Affected files: {candidate['files']}.")
    return " ".join(parts)


def result_events(events: list[dict[str, Any]], candidate: dict[str, Any]) -> list[str]:
    result_ids = []
    validation_parts = []
    if candidate["compile_result"]:
        validation_parts.append(f"Compile: {candidate['compile_result']}")
    if candidate["oracle_result"]:
        validation_parts.append(f"Oracle: {candidate['oracle_result']}")
    if validation_parts:
        result_ids.append(
            event(events, "tool", "validation", "post_candidate_result", "; ".join(validation_parts), [candidate["ledger_ref"]])
        )
    measurement_parts = []
    if candidate["public_candidate"]:
        measurement_parts.append(f"Public Op: {candidate['public_candidate']}")
    if candidate["program_result"]:
        measurement_parts.append(f"Program/Engine: {candidate['program_result']}")
    if candidate["seams"]:
        measurement_parts.append(f"Unaffected controls: {candidate['seams']}")
    if candidate["sentinels"]:
        measurement_parts.append(f"Engine sentinels: {candidate['sentinels']}")
    if measurement_parts:
        result_ids.append(
            event(events, "tool", "measurement", "post_candidate_result", "\n".join(measurement_parts), [candidate["ledger_ref"]])
        )
    if candidate["correction_text"]:
        result_ids.append(
            event(
                events,
                "evaluator",
                "validation",
                "post_candidate_result",
                candidate["correction_text"],
                candidate["correction_refs"],
            )
        )
    return result_ids


def decision_text(candidate: dict[str, Any]) -> str:
    verb = {
        "accepted": "Accept",
        "rejected": "Reject",
        "inconclusive": "Mark inconclusive",
    }[candidate["disposition"]]
    reason = candidate["reason"] or candidate["performance_claim"]
    suffix = ""
    if candidate["disposition"] == "rejected" and candidate["restoration_commit"]:
        suffix = " Restore the candidate through the recorded normal revert and preserve the evidence."
    if candidate["disposition"] == "accepted":
        suffix = " Retain the source only after the declared correctness, control, and real-boundary gates pass."
    return f"{verb}. {reason}.{suffix}".replace("..", ".")


def base_episode(candidate: dict[str, Any], view: str) -> dict[str, Any]:
    family = f"family-auto-{candidate['fingerprint'][:20]}"
    lineage_material = "+".join(
        item for item in (candidate["candidate_commit"], candidate["restoration_commit"]) if item
    ) or f"{candidate['campaign_id']}:{candidate['cycle']}:{candidate['order']}"
    return {
        "schema_version": "1.0.0",
        "episode_id": f"ginfer-auto-{slug(candidate['campaign_id'])}-{slug(candidate['order'])}-{view}",
        "family_id": family,
        "task_view": view,
        "scope": scope(candidate),
        "provenance": {
            "source_ids": candidate["source_ids"],
            "snapshot_refs": candidate["snapshot_refs"],
            "license_status": "internal",
            "evidence_cutoff": candidate["evidence_cutoff"],
        },
        "split_group": {
            "group_id": family.removeprefix("family-"),
            "lineage_id": lineage_material,
            "campaign_id": candidate["campaign_id"],
            "mechanism_cluster": candidate["op_family"],
            "assigned_split": candidate["assigned_split"],
            "evaluation_tier": candidate["evaluation_tier"],
        },
        "events": [],
        "view": {},
        "outcome": {
            "disposition": candidate["disposition"],
            "candidate_commit": candidate["candidate_commit"],
            "restoration_commit": candidate["restoration_commit"],
            "correctness": candidate["correctness"],
            "performance_claim": candidate["performance_claim"],
            "limitations": candidate["limitations"],
        },
        "quality": {
            "review_status": "auto_normalized",
            "hard_negative": candidate["disposition"] == "rejected",
            "contains_hidden_chain_of_thought": False,
            "secret_scan": "passed",
            "dedup_key": f"campaign-auto|{candidate['fingerprint']}",
            "notes": [
                "Deterministically generated from an immutable ledger and, when available, its exact Git patch/restoration history.",
                "Auto-normalized examples require sampled review before a release training run.",
            ],
        },
    }


def diagnosis_episode(candidate: dict[str, Any]) -> dict[str, Any] | None:
    if not candidate["public_baseline"] or not (candidate["hypothesis"] or candidate["prediction"]):
        return None
    episode = base_episode(candidate, "diagnosis")
    events = episode["events"]
    inputs = [
        event(
            events,
            "system",
            "instruction",
            "task_context",
            "Diagnose one falsifiable kernel candidate from only the pre-candidate contract, source, and baseline evidence. Name affected and neutral domains plus the observation that would reject it.",
            [candidate["ledger_ref"]],
        ),
        event(events, "user", "context", "task_context", context_payload(candidate), [candidate["ledger_ref"]]),
    ]
    if candidate["preimage"]:
        inputs.append(
            event(
                events,
                "tool",
                "observation",
                "task_context",
                "Pre-change source hunks reconstructed from the immutable candidate diff:\n```diff\n"
                + candidate["preimage"]
                + "\n```",
                [candidate["history_ref"]],
            )
        )
    inputs.append(
        event(
            events,
            "tool",
            "measurement",
            "pre_candidate_evidence",
            f"Frozen public-Op baseline: {candidate['public_baseline']}"
            + (f"; Program baseline: {candidate['program_baseline']} ms" if candidate["program_baseline"] else ""),
            [candidate["ledger_ref"]],
        )
    )
    rationale_parts = [f"Candidate mechanism: {candidate['mechanism']}."]
    if candidate["hypothesis"]:
        rationale_parts.append(f"Hypothesis: {candidate['hypothesis']}.")
    if candidate["prediction"]:
        rationale_parts.append(f"Expected affected/neutral behavior: {candidate['prediction']}.")
    if candidate["rejection"]:
        rationale_parts.append(f"Reject this hypothesis if: {candidate['rejection']}.")
    target = event(
        events,
        "assistant",
        "technical_rationale",
        "candidate_artifact",
        " ".join(rationale_parts).replace("..", "."),
        [candidate["ledger_ref"]],
    )
    hidden = [target]
    if candidate["patch"]:
        hidden.append(event(events, "assistant", "patch", "candidate_artifact", "```diff\n" + candidate["patch"] + "\n```", [candidate["history_ref"]]))
    hidden.extend(result_events(events, candidate))
    hidden.append(event(events, "assistant", "decision", "disposition_label", decision_text(candidate), [candidate["ledger_ref"]]))
    episode["view"] = {
        "candidate_generation": True,
        "input_event_ids": inputs,
        "target_event_ids": [target],
        "withheld_event_ids": hidden,
        "result_label_visible": False,
    }
    return episode


def implementation_episode(candidate: dict[str, Any]) -> dict[str, Any] | None:
    if not candidate["patch"] or not candidate["preimage"] or not candidate["candidate_commit"]:
        return None
    episode = base_episode(candidate, "implementation")
    events = episode["events"]
    inputs = [
        event(
            events,
            "system",
            "instruction",
            "task_context",
            "Implement the declared narrow candidate in the supplied pre-change source context. Return an exact unified diff and the focused gate sequence; do not use post-candidate results.",
            [candidate["ledger_ref"]],
        ),
        event(events, "user", "context", "task_context", context_payload(candidate), [candidate["ledger_ref"]]),
        event(
            events,
            "tool",
            "observation",
            "pre_candidate_evidence",
            " ".join(
                part
                for part in (
                    f"Mechanism: {candidate['mechanism']}.",
                    f"Hypothesis: {candidate['hypothesis']}." if candidate["hypothesis"] else "",
                    f"Expected behavior: {candidate['prediction']}." if candidate["prediction"] else "",
                    f"Rejection observation: {candidate['rejection']}." if candidate["rejection"] else "",
                )
                if part
            ),
            [candidate["ledger_ref"]],
        ),
        event(
            events,
            "tool",
            "observation",
            "task_context",
            "Pre-change source hunks reconstructed from the immutable candidate diff:\n```diff\n"
            + candidate["preimage"]
            + "\n```",
            [candidate["history_ref"]],
        ),
    ]
    patch_target = event(
        events,
        "assistant",
        "patch",
        "candidate_artifact",
        "```diff\n" + candidate["patch"] + "\n```",
        [candidate["history_ref"]],
    )
    plan_target = event(
        events,
        "assistant",
        "tool_decision",
        "candidate_artifact",
        "Build the focused exact-SM targets, run the independent oracle at affected shapes and dispatch seams, then measure the public Op. Run the declared Program/Engine boundary and unaffected controls only if the lower gates pass.",
        [candidate["ledger_ref"]],
    )
    hidden = [patch_target, plan_target]
    hidden.extend(result_events(events, candidate))
    hidden.append(event(events, "assistant", "decision", "disposition_label", decision_text(candidate), [candidate["ledger_ref"]]))
    if candidate["restoration_commit"]:
        hidden.append(
            event(
                events,
                "tool",
                "restoration",
                "restoration",
                f"Normal revert commit {candidate['restoration_commit']} restores candidate {candidate['candidate_commit']}.",
                [candidate["restoration_ref"]],
            )
        )
    episode["view"] = {
        "candidate_generation": True,
        "input_event_ids": inputs,
        "target_event_ids": [patch_target, plan_target],
        "withheld_event_ids": hidden,
        "result_label_visible": False,
    }
    return episode


def judgment_episode(candidate: dict[str, Any]) -> dict[str, Any]:
    episode = base_episode(candidate, "judgment")
    events = episode["events"]
    candidate_payload = (
        "```diff\n" + candidate["patch"] + "\n```"
        if candidate["patch"]
        else f"Candidate {candidate['order']}: {candidate['mechanism']}."
        + (f" Exact code commit: {candidate['candidate_commit']}." if candidate["candidate_commit"] else "")
    )
    inputs = [
        event(
            events,
            "system",
            "instruction",
            "task_context",
            "Judge the candidate from the independent correctness and pointwise performance evidence. A local Op win does not establish a real-boundary or cross-device claim.",
            [candidate["ledger_ref"]],
        ),
        event(events, "user", "context", "task_context", context_payload(candidate), [candidate["ledger_ref"]]),
        event(
            events,
            "assistant",
            "patch",
            "candidate_artifact",
            candidate_payload,
            [candidate["history_ref"]],
        ),
    ]
    inputs.extend(result_events(events, candidate))
    target = event(events, "assistant", "decision", "disposition_label", decision_text(candidate), [candidate["ledger_ref"]])
    hidden = [target]
    if candidate["restoration_commit"]:
        hidden.append(
            event(
                events,
                "tool",
                "restoration",
                "restoration",
                f"Normal revert commit {candidate['restoration_commit']} restores candidate {candidate['candidate_commit']}.",
                [candidate["restoration_ref"]],
            )
        )
    episode["view"] = {
        "candidate_generation": False,
        "input_event_ids": inputs,
        "target_event_ids": [target],
        "withheld_event_ids": hidden,
        "result_label_visible": False,
    }
    return episode


def orchestration_episode(candidate: dict[str, Any], terminal_no_win: bool) -> dict[str, Any]:
    episode = base_episode(candidate, "orchestration_reporting")
    events = episode["events"]
    inputs = [
        event(
            events,
            "system",
            "instruction",
            "task_context",
            "Choose the next campaign action after a candidate disposition while preserving the immutable ledger and accepted source.",
            [candidate["ledger_ref"]],
        ),
        event(events, "assistant", "decision", "disposition_label", decision_text(candidate), [candidate["ledger_ref"]]),
    ]
    if candidate["disposition"] == "accepted":
        action = "Retain the accepted source, close and push the cycle evidence, then collect a fresh baseline/profile before starting the next cycle."
    elif terminal_no_win:
        action = "Verify the accepted source is restored, freeze the complete no-win ledger, and enter only the bounded evidence-supported consolidation phase defined by the campaign authority."
    else:
        action = "Restore the rejected candidate through its normal revert, verify the accepted source, append the disposition/restoration evidence, and advance to the next unused candidate slot."
    target = event(events, "assistant", "report", "report", action, [candidate["ledger_ref"]])
    hidden = [target]
    if candidate["restoration_commit"]:
        hidden.append(
            event(
                events,
                "tool",
                "restoration",
                "restoration",
                f"Recorded restoration commit: {candidate['restoration_commit']}.",
                [candidate["restoration_ref"]],
            )
        )
    episode["view"] = {
        "candidate_generation": False,
        "input_event_ids": inputs,
        "target_event_ids": [target],
        "withheld_event_ids": hidden,
        "result_label_visible": True,
    }
    return episode


def reconcile_splits(candidates: list[dict[str, Any]]) -> None:
    priority = {"train": 0, "validation": 1, "test": 2, "quarantine": 3}
    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_fingerprint[candidate["fingerprint"]].append(candidate)
    for group in by_fingerprint.values():
        owner = max(group, key=lambda item: priority[item["assigned_split"]])
        for candidate in group:
            candidate["assigned_split"] = owner["assigned_split"]
            candidate["evaluation_tier"] = owner["evaluation_tier"]


def contains_secret(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False)
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def build_campaign_episodes() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads((ROOT / "config/trajectory_campaigns.json").read_text(encoding="utf-8"))
    registry = load_registry()
    raw_by_source = {record["source_id"]: latest_raw_record(record["source_id"]) for record in raw_manifest() if record.get("source_id")}
    schema = json.loads((ROOT / "schemas/canonical_episode.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    excluded_commits = {
        episode["outcome"]["candidate_commit"]
        for episode in json.loads((ROOT / "config/seed_episodes.json").read_text(encoding="utf-8"))
        if episode["outcome"].get("candidate_commit")
    }

    candidates: list[dict[str, Any]] = []
    omissions: Counter[str] = Counter()
    for campaign in config["campaigns"]:
        all_sources = [
            *campaign["ledger_source_ids"],
            *campaign["history_source_ids"],
            *campaign["context_source_ids"],
            *campaign["correction_source_ids"],
        ]
        for source_id in all_sources:
            collected_source(source_id, registry)
            if source_id not in raw_by_source:
                raise KeyError(f"trajectory source has no raw manifest record: {source_id}")
        histories, owners = history_records(campaign["history_source_ids"], registry)
        corrections: list[tuple[str, dict[str, str]]] = []
        for correction_source in campaign["correction_source_ids"]:
            _, path = collected_source(correction_source, registry)
            corrections.extend((correction_source, row) for row in csv_rows(path))
        for ledger_source in campaign["ledger_source_ids"]:
            _, ledger_path = collected_source(ledger_source, registry)
            if ledger_path.suffix.lower() != ".csv":
                raise ValueError(f"trajectory builder currently requires CSV ledgers: {ledger_source}")
            for index, row in enumerate(csv_rows(ledger_path), 1):
                candidate = build_candidate(
                    campaign,
                    ledger_source,
                    row,
                    index,
                    histories,
                    owners,
                    corrections,
                    registry,
                    raw_by_source,
                    config["limits"],
                )
                if candidate["candidate_commit"] in excluded_commits:
                    omissions["seed_candidate_already_curated"] += 1
                    continue
                candidates.append(candidate)

    reconcile_splits(candidates)
    cycle_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        cycle_groups[(candidate["campaign_id"], candidate["cycle"])].append(candidate)

    episodes: list[dict[str, Any]] = []
    for candidate in candidates:
        diagnosis = diagnosis_episode(candidate)
        if diagnosis:
            episodes.append(diagnosis)
        else:
            omissions["diagnosis_missing_pre_candidate_baseline_or_hypothesis"] += 1
        implementation = implementation_episode(candidate)
        if implementation:
            episodes.append(implementation)
        else:
            omissions["implementation_missing_exact_patch_or_preimage"] += 1
        episodes.append(judgment_episode(candidate))

        group = cycle_groups[(candidate["campaign_id"], candidate["cycle"])]
        has_accept = any(item["disposition"] == "accepted" for item in group)
        is_last = candidate is group[-1]
        if candidate["disposition"] == "accepted" or (is_last and not has_accept):
            episodes.append(orchestration_episode(candidate, terminal_no_win=is_last and not has_accept))

    ids = [episode["episode_id"] for episode in episodes]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate generated episode IDs: {duplicates[:10]}")
    for episode in episodes:
        if contains_secret(episode):
            raise ValueError(f"generated episode contains a secret-like value: {episode['episode_id']}")
        validator.validate(episode)

    episodes.sort(key=lambda item: item["episode_id"])
    stats = {
        "producer": PRODUCER,
        "generated_at": latest_timestamp(record["created_at"] for record in raw_manifest()),
        "candidate_families": len(candidates),
        "candidate_families_with_exact_patch": sum(bool(item["patch"] and item["preimage"]) for item in candidates),
        "candidate_families_with_corrections": sum(bool(item["correction_text"]) for item in candidates),
        "episodes": len(episodes),
        "by_view": dict(sorted(Counter(item["task_view"] for item in episodes).items())),
        "by_split": dict(sorted(Counter(item["split_group"]["assigned_split"] for item in episodes).items())),
        "candidates_by_split_and_disposition": {
            "|".join(key): value
            for key, value in sorted(
                Counter(
                    (item["assigned_split"], item["disposition"])
                    for item in candidates
                ).items()
            )
        },
        "episodes_by_split_and_view": {
            "|".join(key): value
            for key, value in sorted(
                Counter(
                    (item["split_group"]["assigned_split"], item["task_view"])
                    for item in episodes
                ).items()
            )
        },
        "by_architecture": dict(sorted(Counter(item["scope"]["architectures"][0] for item in episodes).items())),
        "by_campaign": dict(sorted(Counter(item["split_group"]["campaign_id"] for item in episodes).items())),
        "by_disposition": dict(sorted(Counter(item["outcome"]["disposition"] for item in episodes).items())),
        "omissions": dict(sorted(omissions.items())),
    }
    return episodes, stats


def write_campaign_episodes() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes, stats = build_campaign_episodes()
    write_generated(OUTPUT, jsonl_bytes(episodes))
    write_generated(REPORT, (json.dumps(stats, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return episodes, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="build and validate in memory without writing outputs")
    args = parser.parse_args()
    episodes, stats = build_campaign_episodes() if args.check else write_campaign_episodes()
    print(json.dumps(stats, sort_keys=True))
    if not episodes:
        raise SystemExit("no campaign episodes were generated")


if __name__ == "__main__":
    main()
