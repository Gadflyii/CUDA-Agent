#!/usr/bin/env python3
"""Build canonical episodes and architecture records from reviewed seed annotations."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import jsonschema

from build_campaign_episodes import PRODUCER as CAMPAIGN_PRODUCER
from build_campaign_episodes import write_campaign_episodes
from common import (
    ROOT,
    jsonl_bytes,
    latest_raw_record,
    raw_manifest,
    manifest_record,
    upsert_jsonl,
    write_generated,
)


PRODUCER = "scripts/build_normalized.py@2"
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|github_pat|sk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


def contains_secret(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False)
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def resolve_snapshots(episode: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(episode)
    resolved = []
    for ref in result["provenance"]["snapshot_refs"]:
        if ref.startswith("latest:"):
            resolved.append(latest_raw_record(ref.removeprefix("latest:"))["record_id"])
        else:
            resolved.append(ref)
    result["provenance"]["snapshot_refs"] = resolved
    result["quality"]["secret_scan"] = "failed" if contains_secret(result) else "passed"
    return result


def authored_source(
    source_id: str,
    title: str,
    relative: str,
    created_at: str,
    producer: str,
) -> dict[str, Any]:
    path = ROOT / relative
    from common import sha256_file

    return {
        "source_id": source_id,
        "source_class": "authored_structure",
        "title": title,
        "locator": {"path": relative, "revision": producer},
        "version": "1.0.0",
        "source_date": created_at,
        "retrieved_at": created_at,
        "local_relative_path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "mime_type": "application/x-ndjson" if relative.endswith(".jsonl") else "application/json",
        "sm_applicability": ["cross_sm"],
        "license": {"status": "internal", "identifier": None, "use_note": "Authored dataset structure; internal until reviewed for release."},
        "ingestion_status": "collected",
        "notes": "Generated deterministically from declared immutable inputs and raw manifest references."
    }


def main() -> None:
    raw_records = raw_manifest()
    if not raw_records:
        raise RuntimeError("raw manifest is empty; run collectors first")
    built_at = max(record["created_at"] for record in raw_records)
    episode_schema = json.loads((ROOT / "schemas/canonical_episode.schema.json").read_text(encoding="utf-8"))
    card_schema = json.loads((ROOT / "schemas/architecture_card.schema.json").read_text(encoding="utf-8"))
    delta_schema = json.loads((ROOT / "schemas/cross_sm_delta.schema.json").read_text(encoding="utf-8"))
    seed = json.loads((ROOT / "config/seed_episodes.json").read_text(encoding="utf-8"))
    architecture = json.loads((ROOT / "config/architecture_cards.json").read_text(encoding="utf-8"))

    campaign_episodes, campaign_stats = write_campaign_episodes()
    episodes = [resolve_snapshots(item) for item in seed]
    episodes.extend(campaign_episodes)
    episode_ids = [item["episode_id"] for item in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("seed and generated campaign episodes contain duplicate episode IDs")
    for episode in episodes:
        jsonschema.Draft202012Validator(episode_schema).validate(episode)
    write_generated("normalized/episodes.jsonl", jsonl_bytes(sorted(episodes, key=lambda x: x["episode_id"])))

    generated_paths: list[tuple[str, str, list[str], str, str]] = [
        (
            "normalized/campaign_episodes.jsonl",
            "auto-normalized closed-campaign trajectories",
            sorted({ref for ep in campaign_episodes for ref in ep["provenance"]["snapshot_refs"]}),
            "normalized-campaign-episodes-v1",
            CAMPAIGN_PRODUCER,
        ),
        ("normalized/episodes.jsonl", "canonical episodes", sorted({ref for ep in episodes for ref in ep["provenance"]["snapshot_refs"]}), "normalized-episodes-v1", PRODUCER)
    ]
    for card in architecture["cards"]:
        jsonschema.Draft202012Validator(card_schema).validate(card)
        relative = f"normalized/architecture_cards/{card['architecture_id']}.json"
        write_generated(relative, (json.dumps(card, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        refs = sorted({ref for field in ("immutable_facts", "compiler_observations", "device_measurements") for fact in card[field] for ref in fact["source_refs"]})
        generated_paths.append((relative, f"{card['architecture_id']} architecture card", refs, f"architecture-card-{card['architecture_id']}", PRODUCER))

    for delta in architecture["deltas"]:
        jsonschema.Draft202012Validator(delta_schema).validate(delta)
    write_generated("normalized/cross_sm_deltas.jsonl", jsonl_bytes(sorted(architecture["deltas"], key=lambda x: x["delta_id"])))
    delta_refs = sorted({ref for delta in architecture["deltas"] for ref in delta["source_refs"]})
    generated_paths.append(("normalized/cross_sm_deltas.jsonl", "cross-SM delta records", delta_refs, "cross-sm-deltas-v1", PRODUCER))

    manifests = [
        manifest_record(
            layer="normalized", relative_path=relative, producer=producer, input_refs=refs,
            source_id=source_id, metadata={"title": title}, created_at=built_at,
        )
        for relative, title, refs, source_id, producer in generated_paths
    ]
    write_generated("normalized/manifest.jsonl", jsonl_bytes(sorted(manifests, key=lambda x: x["record_id"])))
    upsert_jsonl(
        "sources/registry.jsonl",
        [
            authored_source(source_id, title, relative, built_at, producer)
            for relative, title, _, source_id, producer in generated_paths
        ],
        "source_id",
    )
    print(
        f"built {len(episodes)} canonical episodes "
        f"({campaign_stats['episodes']} from closed campaigns), "
        f"{len(architecture['cards'])} cards, and {len(architecture['deltas'])} deltas"
    )


if __name__ == "__main__":
    main()
