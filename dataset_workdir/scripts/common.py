#!/usr/bin/env python3
"""Shared deterministic file/provenance helpers for the dataset pipeline."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative_path: str) -> Any:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    return records


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def write_generated(relative_path: str, data: bytes) -> Path:
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return path
    path.write_bytes(data)
    return path


def write_immutable_raw(source_id: str, data: bytes, suffix: str, category: str) -> tuple[str, Path]:
    digest = sha256_bytes(data)
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    relative = Path("raw") / category / source_id / f"{digest}{safe_suffix}"
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"immutable raw object conflicts with its name: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)
    return digest, path


def upsert_jsonl(relative_path: str, records: Iterable[dict[str, Any]], key: str) -> Path:
    path = ROOT / relative_path
    merged = {record[key]: record for record in load_jsonl(path)}
    for record in records:
        existing = merged.get(record[key])
        if existing is not None:
            old_stable = {name: value for name, value in existing.items() if name not in {"created_at", "retrieved_at"}}
            new_stable = {name: value for name, value in record.items() if name not in {"created_at", "retrieved_at"}}
            if old_stable == new_stable:
                continue
        merged[record[key]] = record
    return write_generated(relative_path, jsonl_bytes(merged[name] for name in sorted(merged)))


def manifest_record(
    *,
    layer: str,
    relative_path: str,
    producer: str,
    input_refs: list[str],
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    path = ROOT / relative_path
    digest = sha256_file(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "record_id": f"{layer}:{(source_id or path.stem)}:{digest[:16]}",
        "layer": layer,
        "relative_path": relative_path,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "mime_type": mime_type,
        "created_at": created_at or utc_now(),
        "producer": producer,
        "input_refs": sorted(set(input_refs)),
        "source_id": source_id,
        "metadata": metadata or {},
    }


def raw_manifest() -> list[dict[str, Any]]:
    return load_jsonl(ROOT / "raw/manifest.jsonl")


def latest_raw_record(source_id: str) -> dict[str, Any]:
    matches = [record for record in raw_manifest() if record.get("source_id") == source_id]
    if not matches:
        raise KeyError(f"no collected raw snapshot for source {source_id}")
    registry_path = ROOT / "sources/registry.jsonl"
    if registry_path.exists():
        registered = {record["source_id"]: record for record in load_jsonl(registry_path)}.get(source_id)
        if registered and registered.get("sha256"):
            pinned = [record for record in matches if record["sha256"] == registered["sha256"]]
            if len(pinned) == 1:
                return pinned[0]
    return sorted(matches, key=lambda item: (item["created_at"], item["record_id"]))[-1]


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()
