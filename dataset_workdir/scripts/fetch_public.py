#!/usr/bin/env python3
"""Fetch declared NVIDIA documents and license-limited third-party metadata."""

from __future__ import annotations

import json
import argparse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from common import ROOT, load_jsonl, manifest_record, relative_to_root, sha256_file, upsert_jsonl, utc_now, write_immutable_raw


PRODUCER = "scripts/fetch_public.py@1"
MAX_BYTES = 128 * 1024 * 1024
USER_AGENT = "ginfer-dataset-collector/1.0 (+provenance-only)"
NVIDIA_HOSTS = {"docs.nvidia.com", "developer.nvidia.com", "www.nvidia.com", "nvidia.github.io"}


def reusable_record(source_id: str, url: str, refresh: bool) -> dict[str, Any] | None:
    if refresh:
        return None
    records = {record["source_id"]: record for record in load_jsonl(ROOT / "sources/registry.jsonl")}
    record = records.get(source_id)
    if not record or record.get("ingestion_status") != "collected" or record.get("locator", {}).get("url") != url:
        return None
    relative = record.get("local_relative_path")
    if not relative:
        return None
    path = ROOT / relative
    if not path.exists() or sha256_file(path) != record.get("sha256"):
        return None
    return record


def fetch(url: str) -> tuple[bytes, dict[str, str | None]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_BYTES:
            raise RuntimeError(f"source exceeds {MAX_BYTES} bytes: {url}")
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise RuntimeError(f"source exceeds {MAX_BYTES} bytes: {url}")
        return data, {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "final_url": response.geturl(),
            "content_type": response.headers.get_content_type(),
        }


def source_record(
    *,
    source_id: str,
    source_class: str,
    title: str,
    url: str,
    version: str | None,
    source_date: str | None,
    retrieved_at: str,
    relative: str | None,
    digest: str | None,
    size: int | None,
    mime_type: str,
    sm_applicability: list[str],
    license_status: str,
    license_identifier: str | None,
    license_note: str,
    status: str,
    notes: str,
    http: dict[str, str | None] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    locator: dict[str, str] = {"url": url}
    if revision:
        locator["revision"] = revision
    record: dict[str, Any] = {
        "source_id": source_id,
        "source_class": source_class,
        "title": title,
        "locator": locator,
        "version": version,
        "source_date": source_date,
        "retrieved_at": retrieved_at,
        "local_relative_path": relative,
        "sha256": digest,
        "bytes": size,
        "mime_type": mime_type,
        "sm_applicability": sm_applicability,
        "license": {"status": license_status, "identifier": license_identifier, "use_note": license_note},
        "ingestion_status": status,
        "notes": notes,
    }
    if http:
        record["http"] = {name: http.get(name) for name in ("etag", "last_modified", "final_url")}
    return record


def extension(mime_type: str, url: str) -> str:
    if mime_type == "application/pdf":
        return ".pdf"
    if mime_type == "text/html":
        return ".html"
    suffix = Path(urlparse(url).path).suffix
    return suffix or ".bin"


def collect_nvidia(retrieved_at: str, refresh: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads((ROOT / "config/nvidia_sources.json").read_text(encoding="utf-8"))
    manifests: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for item in config["sources"]:
        host = (urlparse(item["url"]).hostname or "").lower()
        if host not in NVIDIA_HOSTS:
            raise RuntimeError(f"NVIDIA source host is not allowlisted: {item['url']}")
        if "enterprise" in (item["title"] + " " + item["url"]).lower():
            raise RuntimeError(f"NVIDIA Enterprise material is excluded: {item['source_id']}")
        reusable = reusable_record(item["source_id"], item["url"], refresh)
        if reusable:
            registry.append(reusable)
            continue
        try:
            data, http = fetch(item["url"])
            digest, path = write_immutable_raw(
                item["source_id"], data, extension(item["mime_type"], item["url"]), "nvidia"
            )
            relative = relative_to_root(path)
            manifests.append(
                manifest_record(
                    layer="raw",
                    relative_path=relative,
                    producer=PRODUCER,
                    input_refs=[item["source_id"], item["url"]],
                    source_id=item["source_id"],
                    metadata={"source_class": "nvidia_official", "http": http, "topics": item["topics"]},
                    created_at=retrieved_at,
                )
            )
            registry.append(
                source_record(
                    source_id=item["source_id"], source_class="nvidia_official", title=item["title"],
                    url=item["url"], version=item["version"], source_date=item["source_date"],
                    retrieved_at=retrieved_at, relative=relative, digest=digest, size=len(data),
                    mime_type=item["mime_type"], sm_applicability=item["sm_applicability"],
                    license_status="official_public", license_identifier=None,
                    license_note="Official public NVIDIA documentation collected under the user's explicit dataset permission; preserve notices and attribution.",
                    status="collected", notes="Full public document/page snapshot where practical.", http=http,
                )
            )
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            registry.append(
                source_record(
                    source_id=item["source_id"], source_class="nvidia_official", title=item["title"],
                    url=item["url"], version=item["version"], source_date=item["source_date"],
                    retrieved_at=retrieved_at, relative=None, digest=None, size=None,
                    mime_type=item["mime_type"], sm_applicability=item["sm_applicability"],
                    license_status="official_public", license_identifier=None,
                    license_note="Official public NVIDIA documentation; collection failed and no content was used.",
                    status="failed", notes=f"Fetch failed: {type(exc).__name__}: {exc}",
                )
            )
    return manifests, registry


def github_raw_url(repository_url: str, revision: str, path: str) -> str:
    parsed = urlparse(repository_url)
    repo_path = parsed.path.removesuffix(".git").strip("/")
    return f"https://raw.githubusercontent.com/{repo_path}/{revision}/{path}"


def collect_third_party(retrieved_at: str, refresh: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads((ROOT / "config/third_party_sources.json").read_text(encoding="utf-8"))
    manifests: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for repo in config["repositories"]:
        registry.append(
            source_record(
                source_id=repo["source_id"], source_class="third_party_metadata_only",
                title=repo["title"], url=repo["web_url"], version=repo["revision"], source_date=None,
                retrieved_at=retrieved_at, relative=None, digest=None, size=None, mime_type="application/vnd.git.repository",
                sm_applicability=repo["sm_applicability"], license_status=repo["license_status"],
                license_identifier=repo["license_spdx"],
                license_note="Repository provenance and license registration; only separately listed allowed paths are collected.",
                status="declared", notes=repo["note"], revision=repo["revision"],
            )
        )
        if repo["license_status"] != "compatible":
            continue
        for allowed_path in repo["allowed_paths"]:
            child = allowed_path.lower().replace(".", "-").replace("/", "-")
            source_id = f"{repo['source_id']}--{child}"
            url = github_raw_url(repo["repository_url"], repo["revision"], allowed_path)
            mime = "text/markdown" if allowed_path.lower().endswith(".md") else "text/plain"
            reusable = reusable_record(source_id, url, refresh)
            if reusable:
                registry.append(reusable)
                continue
            try:
                data, http = fetch(url)
                digest, path = write_immutable_raw(source_id, data, Path(allowed_path).suffix or ".txt", "third_party")
                relative = relative_to_root(path)
                manifests.append(
                    manifest_record(
                        layer="raw", relative_path=relative, producer=PRODUCER,
                        input_refs=[repo["source_id"], repo["revision"], allowed_path], source_id=source_id,
                        metadata={"source_class": "third_party_licensed", "license": repo["license_spdx"]},
                        created_at=retrieved_at,
                    )
                )
                registry.append(
                    source_record(
                        source_id=source_id, source_class="third_party_licensed",
                        title=f"{repo['title']} — {allowed_path}", url=url, version=repo["revision"], source_date=None,
                        retrieved_at=retrieved_at, relative=relative, digest=digest, size=len(data), mime_type=mime,
                        sm_applicability=repo["sm_applicability"], license_status="compatible",
                        license_identifier=repo["license_spdx"],
                        license_note="Pinned license-compatible repository material; initial import is restricted to LICENSE and README.",
                        status="collected", notes=repo["note"], http=http, revision=repo["revision"],
                    )
                )
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                registry.append(
                    source_record(
                        source_id=source_id, source_class="third_party_metadata_only",
                        title=f"{repo['title']} — {allowed_path}", url=url, version=repo["revision"], source_date=None,
                        retrieved_at=retrieved_at, relative=None, digest=None, size=None, mime_type=mime,
                        sm_applicability=repo["sm_applicability"], license_status="metadata_only",
                        license_identifier=repo["license_spdx"],
                        license_note="License was registered but content collection failed.", status="failed",
                        notes=f"Fetch failed: {type(exc).__name__}: {exc}", revision=repo["revision"],
                    )
                )
    return manifests, registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refetch web sources and preserve any new content-addressed revisions")
    args = parser.parse_args()
    retrieved_at = utc_now()
    nv_manifests, nv_registry = collect_nvidia(retrieved_at, args.refresh)
    tp_manifests, tp_registry = collect_third_party(retrieved_at, args.refresh)
    upsert_jsonl("raw/manifest.jsonl", [*nv_manifests, *tp_manifests], "record_id")
    upsert_jsonl("sources/registry.jsonl", [*nv_registry, *tp_registry], "source_id")
    failures = sum(record["ingestion_status"] == "failed" for record in [*nv_registry, *tp_registry])
    print(f"collected {len(nv_manifests)} new NVIDIA and {len(tp_manifests)} new third-party objects; failures={failures}; refresh={args.refresh}")


if __name__ == "__main__":
    main()
