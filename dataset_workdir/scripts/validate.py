#!/usr/bin/env python3
"""Validate schemas, hashes, lineage splits, leakage controls, and source hygiene."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import FormatChecker

from common import ROOT, load_jsonl, sha256_file, utc_now, write_generated


ALLOWED_GENERATION_INPUT_CLASSES = {"task_context", "pre_candidate_evidence"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access_token": re.compile(r"\b(?:ghp|github_pat|sk)-[A-Za-z0-9_-]{20,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "ssh_command": re.compile(r"\bssh\s+(?:-[^\s]+\s+)*[^\s@]+@[^\s]+"),
}


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.counts: dict[str, int] = {}

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def validate_jsonl_schema(checks: Checks, path: Path, schema: dict[str, Any], label: str) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    validator = jsonschema.Draft202012Validator(schema, format_checker=FormatChecker())
    for index, record in enumerate(records, 1):
        for error in validator.iter_errors(record):
            checks.errors.append(f"{label}:{index}: {error.message}")
    checks.counts[label] = len(records)
    return records


def validate_manifest(checks: Checks, relative: str, layer: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    records = validate_jsonl_schema(checks, ROOT / relative, schema, f"{layer}_manifest")
    seen_paths: dict[str, str] = {}
    for record in records:
        path = ROOT / record["relative_path"]
        checks.require(path.exists(), f"manifest path missing: {record['relative_path']}")
        if not path.exists():
            continue
        actual = sha256_file(path)
        checks.require(actual == record["sha256"], f"hash mismatch: {record['relative_path']}")
        checks.require(path.stat().st_size == record["bytes"], f"byte count mismatch: {record['relative_path']}")
        previous = seen_paths.get(record["relative_path"])
        checks.require(previous in (None, record["sha256"]), f"one path has conflicting hashes: {record['relative_path']}")
        seen_paths[record["relative_path"]] = record["sha256"]
        checks.require(record["layer"] == layer, f"wrong layer in {relative}: {record['record_id']}")
    return records


def scan_secrets(checks: Checks, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists() or path.suffix.lower() in {".pdf", ".sqlite", ".rep"}:
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                checks.errors.append(f"possible {name} in {path.relative_to(ROOT)}")


def main() -> None:
    checks = Checks()
    schemas = {}
    for path in sorted((ROOT / "schemas").glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            checks.errors.append(f"invalid schema {path.name}: {exc.message}")
        schemas[path.name] = schema
    checks.counts["schemas"] = len(schemas)

    registry = validate_jsonl_schema(checks, ROOT / "sources/registry.jsonl", schemas["source_record.schema.json"], "source_registry")
    registry_by_id = {record["source_id"]: record for record in registry}
    checks.require(len(registry_by_id) == len(registry), "duplicate source_id in registry")

    raw = validate_manifest(checks, "raw/manifest.jsonl", "raw", schemas["layer_manifest.schema.json"])
    normalized_manifest = validate_manifest(checks, "normalized/manifest.jsonl", "normalized", schemas["layer_manifest.schema.json"])
    curated_manifest = validate_manifest(checks, "curated/manifest.jsonl", "curated", schemas["layer_manifest.schema.json"])
    raw_ids = {record["record_id"] for record in raw}
    raw_source_ids = {record["source_id"] for record in raw}
    commit_records: dict[str, dict[str, Any]] = {}
    for record in raw:
        if record["metadata"].get("version") and record["relative_path"].endswith(".jsonl"):
            for item in load_jsonl(ROOT / record["relative_path"]):
                if "commit" in item:
                    commit_records[item["commit"]] = item
    # Immutable raw revisions remain append-only. Validate the latest canonical record for each
    # commit so a provenance repair can supersede an older incomplete record without deleting it.
    for item in commit_records.values():
        if item.get("kind") == "restoration":
            checks.require(bool(item.get("reverts_commit")), f"restoration commit lacks candidate link: {item['commit']}")

    campaign_episodes = validate_jsonl_schema(
        checks,
        ROOT / "normalized/campaign_episodes.jsonl",
        schemas["canonical_episode.schema.json"],
        "campaign_episodes",
    )
    episodes = validate_jsonl_schema(checks, ROOT / "normalized/episodes.jsonl", schemas["canonical_episode.schema.json"], "episodes")
    campaign_by_id = {episode["episode_id"]: episode for episode in campaign_episodes}
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}
    checks.require(
        len(campaign_by_id) == len(campaign_episodes),
        "duplicate episode ID in generated campaign episodes",
    )
    checks.require(len(episode_by_id) == len(episodes), "duplicate episode ID in merged episodes")
    checks.require(
        set(campaign_by_id) <= set(episode_by_id),
        "generated campaign episodes are missing from normalized/episodes.jsonl",
    )
    for episode_id, campaign_episode in campaign_by_id.items():
        checks.require(
            episode_by_id.get(episode_id) == campaign_episode,
            f"merged campaign episode differs from generated authority: {episode_id}",
        )
    family_splits: dict[str, set[str]] = defaultdict(set)
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    dedup_splits: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        events = {event["event_id"]: event for event in episode["events"]}
        event_ids = list(events)
        checks.require(len(event_ids) == len(episode["events"]), f"duplicate event ID: {episode['episode_id']}")
        checks.require([event["sequence"] for event in episode["events"]] == list(range(len(episode["events"]))), f"non-contiguous event sequence: {episode['episode_id']}")
        view = episode["view"]
        input_ids = set(view["input_event_ids"])
        target_ids = set(view["target_event_ids"])
        withheld_ids = set(view["withheld_event_ids"])
        checks.require(not input_ids & target_ids, f"input/target overlap: {episode['episode_id']}")
        checks.require(not input_ids & withheld_ids, f"input/withheld overlap: {episode['episode_id']}")
        checks.require((input_ids | target_ids | withheld_ids) <= set(events), f"view names unknown event: {episode['episode_id']}")
        if view["candidate_generation"]:
            classes = {events[event_id]["information_class"] for event_id in input_ids}
            checks.require(classes <= ALLOWED_GENERATION_INPUT_CLASSES, f"candidate result/label leaks into generation input: {episode['episode_id']} ({sorted(classes)})")
            checks.require(not view["result_label_visible"], f"generation view exposes result label: {episode['episode_id']}")
        if episode["task_view"] == "implementation":
            target_events = [events[event_id] for event_id in target_ids]
            input_events = [events[event_id] for event_id in input_ids]
            checks.require(
                any(
                    item["event_type"] == "patch"
                    for item in target_events
                ),
                f"implementation view lacks a patch target: {episode['episode_id']}",
            )
            if episode["quality"]["review_status"] == "auto_normalized":
                checks.require(
                    any(
                        item["event_type"] == "patch"
                        and "diff --git" in str(item["payload"])
                        for item in target_events
                    ),
                    f"auto-normalized implementation lacks an exact unified diff target: {episode['episode_id']}",
                )
                checks.require(
                    any(
                        item["event_type"] == "observation"
                        and "Pre-change source hunks" in str(item["payload"])
                        for item in input_events
                    ),
                    f"auto-normalized implementation lacks reconstructed preimage: {episode['episode_id']}",
                )
        if not view["result_label_visible"]:
            checks.require(all(events[event_id]["information_class"] != "disposition_label" for event_id in input_ids), f"hidden label event appears in input: {episode['episode_id']}")
        outcome = episode["outcome"]
        if outcome["disposition"] == "rejected" and outcome["candidate_commit"]:
            checks.require(bool(outcome["restoration_commit"]), f"rejected committed candidate lacks restoration: {episode['episode_id']}")
            restoration = commit_records.get(outcome["restoration_commit"] or "")
            checks.require(restoration is not None, f"restoration commit missing from raw Git evidence: {episode['episode_id']}")
            if restoration:
                checks.require(restoration.get("kind") == "restoration", f"linked restoration is not classified as a revert: {episode['episode_id']}")
                checks.require(restoration.get("reverts_commit") == outcome["candidate_commit"], f"restoration does not revert linked candidate: {episode['episode_id']}")
        if outcome["candidate_commit"] in commit_records:
            checks.require(commit_records[outcome["candidate_commit"]].get("kind") != "restoration", f"revert is treated as candidate outcome: {episode['episode_id']}")
        for source_id in episode["provenance"]["source_ids"]:
            checks.require(source_id in registry_by_id, f"episode references unknown source: {episode['episode_id']} -> {source_id}")
        for snapshot in episode["provenance"]["snapshot_refs"]:
            checks.require(snapshot in raw_ids, f"episode references unknown raw snapshot: {episode['episode_id']} -> {snapshot}")
        split = episode["split_group"]["assigned_split"]
        family_splits[episode["family_id"]].add(split)
        lineage_splits[episode["split_group"]["lineage_id"]].add(split)
        dedup_splits[episode["quality"]["dedup_key"]].add(split)
        checks.require(episode["quality"]["contains_hidden_chain_of_thought"] is False, f"hidden CoT flag set: {episode['episode_id']}")
        checks.require(episode["quality"]["secret_scan"] == "passed", f"episode secret scan not passed: {episode['episode_id']}")

    for family_id, splits in family_splits.items():
        checks.require(len(splits) == 1, f"family crosses splits: {family_id} -> {sorted(splits)}")
    for lineage_id, splits in lineage_splits.items():
        checks.require(len(splits) == 1, f"lineage crosses splits: {lineage_id} -> {sorted(splits)}")
    for dedup_key, splits in dedup_splits.items():
        checks.require(len(splits) == 1, f"dedup cluster crosses splits: {dedup_key} -> {sorted(splits)}")

    for arch in ("sm_86", "sm_89", "sm_120a"):
        path = ROOT / f"normalized/architecture_cards/{arch}.json"
        card = json.loads(path.read_text(encoding="utf-8"))
        for error in jsonschema.Draft202012Validator(schemas["architecture_card.schema.json"]).iter_errors(card):
            checks.errors.append(f"architecture card {arch}: {error.message}")
        for fact in card["compiler_observations"]:
            checks.require(bool(fact["toolchain"] and fact["build_revision"]), f"compiler fact lacks toolchain/build: {arch}/{fact['fact_id']}")
        for fact in card["device_measurements"]:
            checks.require(bool(fact["hardware_sku"] and fact["workload_key"]), f"device fact lacks SKU/workload: {arch}/{fact['fact_id']}")

    deltas = validate_jsonl_schema(checks, ROOT / "normalized/cross_sm_deltas.jsonl", schemas["cross_sm_delta.schema.json"], "cross_sm_deltas")
    for delta in deltas:
        for source_id in delta["source_refs"]:
            checks.require(source_id in registry_by_id, f"delta references unknown source: {delta['delta_id']} -> {source_id}")

    live_ids = {record["source_id"] for record in registry if record["source_class"] == "ginfer_live_reference"}
    checks.require(not (live_ids & raw_source_ids), f"live campaign content entered raw layer: {sorted(live_ids & raw_source_ids)}")

    for record in registry:
        if record["source_class"] == "nvidia_official" and record["ingestion_status"] == "collected":
            checks.require(bool(record["locator"].get("url") and record["sha256"] and record["local_relative_path"]), f"incomplete NVIDIA provenance: {record['source_id']}")
        if record["source_class"].startswith("third_party"):
            checks.require(record["license"]["status"] in {"compatible", "metadata_only"}, f"third-party content lacks compatible/metadata-only status: {record['source_id']}")

    # Public eval IDs and private label IDs must match, but content must stay in distinct files.
    for split in ("validation", "test"):
        public = load_jsonl(ROOT / f"curated/eval/{split}.public.jsonl")
        private = load_jsonl(ROOT / f"curated/eval/{split}.private.jsonl")
        checks.require({row["id"] for row in public} == {row["id"] for row in private}, f"eval prompt/label ID mismatch: {split}")
        checks.require(all("reference_answer" not in row and "outcome" not in row for row in public), f"labels leak into public eval: {split}")

    manifest_paths = [ROOT / record["relative_path"] for record in [*raw, *normalized_manifest, *curated_manifest]]
    scan_secrets(checks, [ROOT / "config/seed_episodes.json", ROOT / "sources/registry.jsonl", *manifest_paths])

    report = {
        "validated_at": utc_now(),
        "status": "pass" if not checks.errors else "fail",
        "counts": checks.counts,
        "errors": checks.errors,
        "warnings": checks.warnings,
        "invariants": {
            "schemas_validated": True,
            "manifest_hashes_checked": True,
            "candidate_generation_label_leakage_checked": True,
            "campaign_episode_merge_checked": True,
            "implementation_patch_preimage_checked": True,
            "revert_links_checked": True,
            "split_lineages_checked": True,
            "live_sources_excluded_from_raw": True,
            "secret_patterns_checked": True,
        },
    }
    write_generated("reports/validation.json", (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": report["status"], "counts": checks.counts, "errors": len(checks.errors)}, sort_keys=True))
    if checks.errors:
        for error in checks.errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
