#!/usr/bin/env python3
"""Derive SFT, preference, eval, and quarantine exports from canonical events."""

from __future__ import annotations

import json
from typing import Any

from common import ROOT, jsonl_bytes, load_jsonl, manifest_record, utc_now, write_generated


PRODUCER = "scripts/export_dataset.py@1"


def payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)


def select_events(episode: dict[str, Any], event_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {event["event_id"]: event for event in episode["events"]}
    return [by_id[event_id] for event_id in event_ids]


def render(events: list[dict[str, Any]]) -> str:
    blocks = []
    for event in events:
        refs = ", ".join(event["evidence_refs"]) or "none"
        blocks.append(f"[{event['event_type']} | evidence: {refs}]\n{payload_text(event['payload'])}")
    return "\n\n".join(blocks)


def prompt_for(episode: dict[str, Any]) -> str:
    return render(select_events(episode, episode["view"]["input_event_ids"]))


def target_for(episode: dict[str, Any]) -> str:
    return render(select_events(episode, episode["view"]["target_event_ids"]))


def main() -> None:
    normalized_records = load_jsonl(ROOT / "normalized/manifest.jsonl")
    if not normalized_records:
        raise RuntimeError("normalized manifest is empty; build normalized data first")
    exported_at = max(record["created_at"] for record in normalized_records)
    episodes = load_jsonl(ROOT / "normalized/episodes.jsonl")
    written: list[tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]] = []

    for split in ("train", "validation", "test"):
        rows = []
        for episode in episodes:
            if episode["split_group"]["assigned_split"] != split:
                continue
            rows.append(
                {
                    "id": episode["episode_id"],
                    "messages": [
                        {"role": "system", "content": "Act as an evidence-disciplined GInfer C++/CUDA kernel optimization engineer. Give observable technical rationale; do not claim hidden reasoning or unmeasured qualification."},
                        {"role": "user", "content": prompt_for(episode)},
                        {"role": "assistant", "content": target_for(episode)},
                    ],
                    "metadata": {
                        "family_id": episode["family_id"],
                        "task_view": episode["task_view"],
                        "architectures": episode["scope"]["architectures"],
                        "snapshot_refs": episode["provenance"]["snapshot_refs"],
                    },
                }
            )
        relative = f"curated/sft/{split}.jsonl"
        write_generated(relative, jsonl_bytes(rows))
        written.append((relative, rows, [item["id"] for item in rows], {"format": "messages-sft", "split": split}))

    preference_rows = []
    for episode in episodes:
        if episode["split_group"]["assigned_split"] != "train" or not episode.get("preference"):
            continue
        preference_rows.append(
            {
                "id": episode["episode_id"],
                "prompt": prompt_for(episode),
                "chosen": render(select_events(episode, episode["preference"]["chosen_event_ids"])),
                "rejected": episode["preference"]["rejected_response"],
                "failure_mode": episode["preference"]["failure_mode"],
                "family_id": episode["family_id"],
                "snapshot_refs": episode["provenance"]["snapshot_refs"],
            }
        )
    write_generated("curated/preference/train.jsonl", jsonl_bytes(preference_rows))
    written.append(("curated/preference/train.jsonl", preference_rows, [item["id"] for item in preference_rows], {"format": "pairwise-preference", "split": "train"}))

    for split in ("validation", "test"):
        public_rows = []
        private_rows = []
        for episode in episodes:
            if episode["split_group"]["assigned_split"] != split:
                continue
            public_rows.append(
                {
                    "id": episode["episode_id"],
                    "prompt": prompt_for(episode),
                    "task_view": episode["task_view"],
                    "architecture_scope": episode["scope"]["architectures"],
                    "rubric": ["contract correctness", "evidence calibration", "architecture scope", "appropriate verification"],
                }
            )
            private_rows.append(
                {
                    "id": episode["episode_id"],
                    "reference_answer": target_for(episode),
                    "outcome": episode["outcome"],
                    "required_snapshot_refs": episode["provenance"]["snapshot_refs"],
                }
            )
        public_path = f"curated/eval/{split}.public.jsonl"
        private_path = f"curated/eval/{split}.private.jsonl"
        write_generated(public_path, jsonl_bytes(public_rows))
        write_generated(private_path, jsonl_bytes(private_rows))
        written.append((public_path, public_rows, [item["id"] for item in public_rows], {"format": "evaluation-prompts", "split": split, "contains_labels": False}))
        written.append((private_path, private_rows, [item["id"] for item in private_rows], {"format": "evaluation-labels", "split": split, "contains_labels": True}))

    quarantine_rows = [
        {
            "id": episode["episode_id"],
            "reason": episode["outcome"]["limitations"],
            "review_status": episode["quality"]["review_status"],
            "source_ids": episode["provenance"]["source_ids"],
        }
        for episode in episodes
        if episode["split_group"]["assigned_split"] == "quarantine"
    ]
    write_generated("curated/quarantine/index.jsonl", jsonl_bytes(quarantine_rows))
    written.append(("curated/quarantine/index.jsonl", quarantine_rows, [item["id"] for item in quarantine_rows], {"format": "quarantine-index"}))

    manifests = [
        manifest_record(
            layer="curated", relative_path=relative, producer=PRODUCER,
            input_refs=["normalized-episodes-v1", *episode_ids], metadata=metadata, created_at=exported_at,
        )
        for relative, _, episode_ids, metadata in written
    ]
    write_generated("curated/manifest.jsonl", jsonl_bytes(sorted(manifests, key=lambda item: item["record_id"])))
    print(f"wrote {len(written)} curated export files")


if __name__ == "__main__":
    main()
