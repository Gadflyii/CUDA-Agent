#!/usr/bin/env python3
"""Materialize explicitly supplied evidence while preserving its pending-provenance status."""

from __future__ import annotations

import json

from common import ROOT, manifest_record, relative_to_root, upsert_jsonl, utc_now, write_immutable_raw


PRODUCER = "scripts/collect_external.py@1"


def main() -> None:
    config = json.loads((ROOT / "config/external_evidence.json").read_text(encoding="utf-8"))
    collected_at = utc_now()
    manifests = []
    registry = []
    for item in config["sources"]:
        payload = {
            "schema_version": config["schema_version"],
            "source_id": item["source_id"],
            "title": item["title"],
            "received_at": item["received_at"],
            "provenance_status": item["provenance_status"],
            "evidence": item["evidence"],
        }
        data = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        digest, path = write_immutable_raw(item["source_id"], data, ".json", "external")
        relative = relative_to_root(path)
        manifests.append(
            manifest_record(
                layer="raw", relative_path=relative, producer=PRODUCER,
                input_refs=[item["source_id"], item["provenance_status"]], source_id=item["source_id"],
                metadata={"source_class": "external_user_supplied", "provenance_status": item["provenance_status"]},
                created_at=collected_at,
            )
        )
        registry.append(
            {
                "source_id": item["source_id"],
                "source_class": "external_user_supplied",
                "title": item["title"],
                "locator": {"path": "parent-task-message/rtx5090-corrected-retest", "revision": "pending-repository-log"},
                "version": "externally-supplied-v1",
                "source_date": item["received_at"],
                "retrieved_at": collected_at,
                "local_relative_path": relative,
                "sha256": digest,
                "bytes": len(data),
                "mime_type": "application/json",
                "sm_applicability": item["sm_applicability"],
                "license": {
                    "status": "internal",
                    "identifier": None,
                    "use_note": "User-supplied internal measurement evidence; do not redistribute or promote without repository-log provenance."
                },
                "ingestion_status": "collected",
                "notes": "Externally supplied and quarantined pending immutable repository result/log provenance."
            }
        )
    upsert_jsonl("raw/manifest.jsonl", manifests, "record_id")
    upsert_jsonl("sources/registry.jsonl", registry, "source_id")
    print(f"collected {len(manifests)} externally supplied evidence records")


if __name__ == "__main__":
    main()
