#!/usr/bin/env python3
"""Collect only declared immutable or explicitly closed GInfer evidence."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from common import (
    ROOT,
    jsonl_bytes,
    manifest_record,
    relative_to_root,
    sha256_bytes,
    upsert_jsonl,
    utc_now,
    write_immutable_raw,
)


PRODUCER = "scripts/collect_ginfer.py@1"


def git(repo: Path, *args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    return result.stdout


def commit_time(repo: Path, revision: str) -> str:
    return str(git(repo, "show", "-s", "--format=%cI", revision, text=True)).strip()


def collect_git_range(repo: Path, item: dict[str, Any]) -> bytes:
    start = item["revision_start_exclusive"]
    end = item["revision_end_inclusive"]
    revision_range = f"{start}..{end}"
    commit_ids = str(git(repo, "rev-list", "--reverse", "--topo-order", revision_range, text=True)).splitlines()
    records: list[dict[str, Any]] = []
    for commit_id in commit_ids:
        meta = str(
            git(
                repo,
                "show",
                "-s",
                "--format=%H%x00%P%x00%cI%x00%s%x00%B",
                commit_id,
                text=True,
            )
        ).split("\x00", 4)
        patch = str(
            git(
                repo,
                "show",
                "--no-ext-diff",
                "--no-color",
                "--find-renames",
                "--format=",
                commit_id,
                text=True,
            )
        )
        body = meta[4].strip()
        revert_match = re.search(r"This reverts commit ([0-9a-f]{40})", body)
        records.append(
            {
                "commit": meta[0],
                "parents": meta[1].split(),
                "committed_at": meta[2],
                "subject": meta[3],
                "body": body,
                "kind": "restoration" if meta[3].startswith("Revert ") else "candidate_or_retained_change",
                "reverts_commit": revert_match.group(1) if revert_match else None,
                "patch": patch,
            }
        )
    return jsonl_bytes(records)


def source_record(
    item: dict[str, Any],
    *,
    repo: Path,
    retrieved_at: str,
    local_path: str,
    digest: str,
    size: int,
    source_class: str,
    version: str,
    source_date: str | None,
) -> dict[str, Any]:
    locator: dict[str, str] = {"repository": str(repo)}
    if item["kind"] == "git_file":
        locator.update({"path": item["path"], "revision": item["revision"]})
    elif item["kind"] == "git_range":
        locator.update({"revision": version})
    else:
        locator.update({"path": item["path"], "revision": item["closure_evidence_revision"]})
    return {
        "source_id": item["source_id"],
        "source_class": source_class,
        "title": item["title"],
        "locator": locator,
        "version": version,
        "source_date": source_date,
        "retrieved_at": retrieved_at,
        "local_relative_path": local_path,
        "sha256": digest,
        "bytes": size,
        "mime_type": item["mime_type"],
        "sm_applicability": item["sm_applicability"],
        "license": {
            "status": "internal",
            "identifier": None,
            "use_note": "Internal GInfer evidence; redistribution requires project-owner approval."
        },
        "ingestion_status": "collected",
        "notes": "Read from an immutable Git object or an explicitly closed, hash-pinned local ledger."
    }


def main() -> None:
    config = json.loads((ROOT / "config/ginfer_sources.json").read_text(encoding="utf-8"))
    default_repo = Path(config["repository"])
    collected_at = utc_now()
    manifests: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for item in config["collections"]:
        repo = Path(item.get("repository", default_repo))
        kind = item["kind"]
        if kind == "git_file":
            data = bytes(git(repo, "show", f"{item['revision']}:{item['path']}"))
            version = item["revision"]
            date = commit_time(repo, item["revision"])
            suffix = Path(item["path"]).suffix or ".bin"
            source_class = "ginfer_committed"
        elif kind == "git_range":
            data = collect_git_range(repo, item)
            version = f"{item['revision_start_exclusive']}..{item['revision_end_inclusive']}"
            date = commit_time(repo, item["revision_end_inclusive"])
            suffix = ".jsonl"
            source_class = "ginfer_committed"
        elif kind == "closed_local_file":
            path = Path(item["path"])
            data = path.read_bytes()
            observed = sha256_bytes(data)
            if observed != item["expected_sha256"]:
                raise RuntimeError(
                    f"closed ledger changed after declaration: {path}; expected {item['expected_sha256']}, got {observed}"
                )
            git(repo, "cat-file", "-e", f"{item['closure_evidence_revision']}^{{commit}}")
            version = f"closed-at-{item['closure_evidence_revision']}"
            date = commit_time(repo, item["closure_evidence_revision"])
            suffix = path.suffix or ".bin"
            source_class = "ginfer_closed_local"
        else:
            raise ValueError(f"unsupported GInfer source kind: {kind}")

        digest, raw_path = write_immutable_raw(item["source_id"], data, suffix, "ginfer")
        relative = relative_to_root(raw_path)
        manifest = manifest_record(
            layer="raw",
            relative_path=relative,
            producer=PRODUCER,
            input_refs=[item["source_id"], version],
            source_id=item["source_id"],
            metadata={"source_class": source_class, "version": version},
            created_at=collected_at,
        )
        manifests.append(manifest)
        source_records.append(
            source_record(
                item,
                repo=repo,
                retrieved_at=collected_at,
                local_path=relative,
                digest=digest,
                size=len(data),
                source_class=source_class,
                version=version,
                source_date=date,
            )
        )

    for item in config["live_references"]:
        locator: dict[str, str] = {"path": item["path"]}
        if item.get("repository"):
            locator["repository"] = item["repository"]
        if item.get("observed_revision"):
            locator["revision"] = item["observed_revision"]
        if item.get("host_label"):
            locator["host_label"] = item["host_label"]
        source_records.append(
            {
                "source_id": item["source_id"],
                "source_class": "ginfer_live_reference",
                "title": item["title"],
                "locator": locator,
                "version": item.get("observed_revision"),
                "source_date": None,
                "retrieved_at": collected_at,
                "local_relative_path": None,
                "sha256": None,
                "bytes": None,
                "mime_type": "application/octet-stream",
                "sm_applicability": item["sm_applicability"],
                "license": {
                    "status": "internal",
                    "identifier": None,
                    "use_note": "Location only; active content is excluded until closure is established."
                },
                "ingestion_status": "live_reference",
                "notes": item["note"]
            }
        )

    upsert_jsonl("raw/manifest.jsonl", manifests, "record_id")
    upsert_jsonl("sources/registry.jsonl", source_records, "source_id")
    print(f"collected {len(manifests)} immutable GInfer objects; registered {len(config['live_references'])} live locations")


if __name__ == "__main__":
    main()
