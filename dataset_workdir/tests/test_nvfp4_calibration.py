from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_nvfp4_calibration as calibration  # noqa: E402


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if not tokenize or add_generation_prompt:
            raise AssertionError("unexpected chat-template options")
        return [int(value) for value in messages[0]["content"].split()]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("decode must preserve every token")
        return " ".join(str(value) for value in token_ids)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        if add_special_tokens or truncation:
            raise AssertionError("round trip must not alter the token stream")
        return {"input_ids": [int(value) for value in text.split()]}


class NonRoundTripTokenizer(FakeTokenizer):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        result = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )
        result["input_ids"][-1] += 1
        return result


def _transformers_module(tokenizer_type: type[FakeTokenizer]) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(
            tokenizer_root: Path,
            *,
            trust_remote_code: bool,
        ) -> FakeTokenizer:
            if not Path(tokenizer_root).is_dir() or trust_remote_code:
                raise AssertionError("builder did not use the pinned local tokenizer")
            return tokenizer_type()

    module.AutoTokenizer = AutoTokenizer
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_tokenizer(root: Path) -> dict[str, str]:
    root.mkdir()
    payloads = {
        "tokenizer.json": b'{"version":"fake-v1"}\n',
        "tokenizer_config.json": b'{"eos_token":"<eos>"}\n',
        "chat_template.jinja": b"{{ messages }}\n",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }


class Nvfp4CalibrationTests(unittest.TestCase):
    def test_packing_is_deterministic_and_manifest_binds_every_input(self) -> None:
        rows = [
            {"id": "train-z", "messages": [{"role": "user", "content": "31 32 33"}]},
            {"id": "train-a", "messages": [{"role": "user", "content": "11 12 13"}]},
            {"id": "train-m", "messages": [{"role": "user", "content": "21 22 23"}]},
        ]
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"nvfp4-calibration-v1\0{row['id']}".encode()
            ).digest(),
        )
        expected_ids = [str(row["id"]) for row in ordered[:2]]
        expected_tokens = [
            int(value)
            for row in ordered[:2]
            for value in (str(row["messages"][0]["content"]) + " 0").split()
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_root = root / "tokenizer"
            tokenizer_hashes = _write_tokenizer(tokenizer_root)
            train_a = root / "train-a.jsonl"
            train_b = root / "train-b.jsonl"
            _write_jsonl(train_a, rows)
            _write_jsonl(train_b, list(reversed(rows)))
            output_a = root / "calibration-a.jsonl"
            output_b = root / "calibration-b.jsonl"

            with (
                mock.patch.object(calibration, "SAMPLES", 2),
                mock.patch.object(calibration, "SEQUENCE_LENGTH", 4),
                mock.patch.dict(
                    sys.modules,
                    {"transformers": _transformers_module(FakeTokenizer)},
                ),
            ):
                _, manifest_a_path = calibration.build(
                    train_a, tokenizer_root, output_a
                )
                _, manifest_b_path = calibration.build(
                    train_b, tokenizer_root, output_b
                )

            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            output_rows = [
                json.loads(line)
                for line in output_a.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(output_rows), 2)
            tokenizer = FakeTokenizer()
            actual_chunks = [
                tokenizer(row["text"], add_special_tokens=False, truncation=False)[
                    "input_ids"
                ]
                for row in output_rows
            ]
            self.assertEqual(
                actual_chunks,
                [expected_tokens[:4], expected_tokens[4:]],
            )

            manifest_a = json.loads(manifest_a_path.read_text(encoding="utf-8"))
            manifest_b = json.loads(manifest_b_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_a["schema_version"], 1)
            self.assertEqual(manifest_a["profile_id"], calibration.PROFILE_ID)
            self.assertEqual(
                (manifest_a["samples"], manifest_a["sequence_length"], manifest_a["tokens"]),
                (2, 4, 8),
            )
            self.assertEqual(manifest_a["source"]["path"], str(train_a.resolve()))
            self.assertEqual(manifest_a["source"]["sha256"], _sha256(train_a))
            self.assertEqual(
                manifest_a["source"]["selected_episode_ids"], expected_ids
            )
            self.assertEqual(
                manifest_b["source"]["selected_episode_ids"], expected_ids
            )
            self.assertEqual(
                manifest_a["tokenizer"],
                {"path": str(tokenizer_root.resolve()), "files": tokenizer_hashes},
            )
            self.assertEqual(manifest_a["output"]["path"], str(output_a.resolve()))
            self.assertEqual(manifest_a["output"]["sha256"], _sha256(output_a))

    def test_insufficient_training_data_does_not_fall_back_to_other_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_root = root / "tokenizer"
            _write_tokenizer(tokenizer_root)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            _write_jsonl(
                train,
                [{"id": "train-only", "messages": [{"role": "user", "content": "1 2 3"}]}],
            )
            _write_jsonl(
                validation,
                [
                    {
                        "id": "must-not-be-used",
                        "messages": [
                            {"role": "user", "content": "4 5 6 7 8 9 10 11"}
                        ],
                    }
                ],
            )
            output = root / "calibration.jsonl"

            with (
                mock.patch.object(calibration, "SAMPLES", 2),
                mock.patch.object(calibration, "SEQUENCE_LENGTH", 4),
                mock.patch.dict(
                    sys.modules,
                    {"transformers": _transformers_module(FakeTokenizer)},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "training export supplies 4 calibration tokens; 8 are required",
                ),
            ):
                calibration.build(train, tokenizer_root, output)

            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".manifest.json").exists())

    def test_non_exact_text_round_trip_is_rejected_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_root = root / "tokenizer"
            _write_tokenizer(tokenizer_root)
            train = root / "train.jsonl"
            _write_jsonl(
                train,
                [{"id": "train", "messages": [{"role": "user", "content": "1 2 3 4 5 6 7"}]}],
            )
            output = root / "calibration.jsonl"

            with (
                mock.patch.object(calibration, "SAMPLES", 2),
                mock.patch.object(calibration, "SEQUENCE_LENGTH", 4),
                mock.patch.dict(
                    sys.modules,
                    {"transformers": _transformers_module(NonRoundTripTokenizer)},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "calibration row 0 is not token-exact after text round trip",
                ),
            ):
                calibration.build(train, tokenizer_root, output)

            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".manifest.json").exists())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["tokenizer", "train.jsonl"],
            )


if __name__ == "__main__":
    unittest.main()
