#!/usr/bin/env python3
"""Pack training-only SFT text into the pinned ModelOpt NVFP4 calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence


PROFILE_ID = "qwen3_8_27b-modelopt-nvfp4-mlp0-55-mse-v1"
SAMPLES = 512
SEQUENCE_LENGTH = 4096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row_id = value.get("id")
            messages = value.get("messages")
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: messages must be nonempty")
            if any(
                not isinstance(item, dict)
                or item.get("role") not in ("system", "user", "assistant", "tool")
                or not isinstance(item.get("content"), str)
                for item in messages
            ):
                raise ValueError(f"{path}:{line_number}: invalid message")
            ids.add(row_id)
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: training export is empty")
    return sorted(
        rows,
        key=lambda item: hashlib.sha256(
            f"nvfp4-calibration-v1\0{item['id']}".encode()
        ).digest(),
    )


def _tokenizer_hashes(root: Path) -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise ValueError(f"tokenizer source is missing {path}")
        result[name] = _sha256(path)
    return result


def build(
    train_path: str | Path,
    tokenizer_path: str | Path,
    out_path: str | Path,
) -> tuple[Path, Path]:
    """Build exactly 512 round-trippable 4096-token text rows."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to build calibration rows") from error

    train = Path(train_path).resolve()
    tokenizer_root = Path(tokenizer_path).resolve()
    output = Path(out_path).resolve()
    if output.exists() or Path(str(output) + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {output}")
    rows = _load_rows(train)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, trust_remote_code=False
    )
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise ValueError("tokenizer must define one nonnegative EOS token id")

    stream: list[int] = []
    source_ids: list[str] = []
    required = SAMPLES * SEQUENCE_LENGTH
    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        if not isinstance(token_ids, list) or any(
            not isinstance(item, int) for item in token_ids
        ):
            raise ValueError(f"{row['id']}: tokenizer returned invalid token ids")
        stream.extend(token_ids)
        if not token_ids or token_ids[-1] != eos:
            stream.append(eos)
        source_ids.append(row["id"])
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise ValueError(
            f"training export supplies {len(stream)} calibration tokens; "
            f"{required} are required"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    manifest_temporary: Path | None = None
    manifest_path = Path(str(output) + ".manifest.json")
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for index in range(SAMPLES):
                ids = stream[
                    index * SEQUENCE_LENGTH : (index + 1) * SEQUENCE_LENGTH
                ]
                text = tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                roundtrip = tokenizer(
                    text, add_special_tokens=False, truncation=False
                )["input_ids"]
                if roundtrip != ids:
                    raise ValueError(
                        f"calibration row {index} is not token-exact after text round trip"
                    )
                handle.write(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n"
                )

        manifest = {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "source": {
                "path": str(train),
                "sha256": _sha256(train),
                "selected_episode_ids": source_ids,
            },
            "tokenizer": {
                "path": str(tokenizer_root),
                "files": _tokenizer_hashes(tokenizer_root),
            },
            "samples": SAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "tokens": required,
            "output": {
                "path": str(output),
                "sha256": _sha256(temporary),
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{manifest_path.name}.",
            delete=False,
        ) as handle:
            manifest_temporary = Path(handle.name)
            handle.write(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
        temporary.rename(output)
        temporary = None
        manifest_temporary.rename(manifest_path)
        manifest_temporary = None
    except BaseException:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if manifest_temporary is not None and manifest_temporary.exists():
            manifest_temporary.unlink()
        if output.exists() and not manifest_path.exists():
            output.unlink()
        raise
    return output, manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", default="curated/sft/train.jsonl", type=Path
    )
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)
    output, manifest = build(
        arguments.train, arguments.tokenizer, arguments.out
    )
    print(f"wrote {output} and {manifest}")


if __name__ == "__main__":
    main()
