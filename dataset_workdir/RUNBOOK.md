# Collection and export runbook

All commands run from this directory. Collectors only read declared sources and write beneath this
workspace. They do not fetch, switch, build, test, benchmark, or profile a GInfer checkout.

```bash
python3 scripts/collect_ginfer.py
python3 scripts/fetch_public.py
python3 scripts/build_campaign_episodes.py --check
python3 scripts/build_normalized.py
python3 scripts/export_dataset.py
python3 scripts/validate.py
```

Use `python3 scripts/pipeline.py` for the same sequence. Re-running is idempotent. Existing raw
objects must match their recorded hash; conflicts fail closed. Network fetches preserve a new
content-addressed revision instead of replacing an older object.

The post-training NVFP4 calibration set is a separate, tokenizer-bound release product. Build it
only after freezing the train split and consolidated BF16 checkpoint:

```bash
python3 scripts/build_nvfp4_calibration.py \
  --train curated/sft/train.jsonl \
  --tokenizer /path/to/qwen3_8_27b_agent_bf16 \
  --out /path/to/qwen3_8_27b_nvfp4_calibration.jsonl
```

The builder hash-orders train episodes, applies the exact checkpoint chat template, packs exactly
512 x 4096 tokens, verifies every decoded row round-trips to the same token IDs, and writes a hash
manifest. It fails when the train corpus has fewer than 2,097,152 usable tokens; validation/test
rows are never fallback inputs. Feed that JSONL to GInfer's pinned
`tools.convert.qwen3_8_27b.quantize_modelopt_nvfp4` command together with its untouched sibling
manifest. The quantizer rejects a missing/mismatched manifest or tokenizer and verifies that its
Python environment imports ModelOpt from the clean, pinned checkout.

`build_normalized.py` runs the campaign builder again and merges its output with reviewed seed
episodes. The explicit `--check` command is the fast, non-writing preflight for source admission,
commit/revert resolution, schema validity, and secret scanning. Review
`reports/trajectory_builder.json` after the writing build; do not promote auto-normalized training
rows without sampling exact diffs, preimages, correction application, and outcome labels.

The default public collector reuses the exact collected registry revision. Run
`python3 scripts/fetch_public.py --refresh` only for an intentional web-source refresh; changed
bytes are appended as a new content-addressed object and never replace the prior snapshot.

## Closing and importing a live cycle

`config/ginfer_sources.json` may contain `live_reference` entries. These register where evidence is
expected but are never copied. After the campaign owner has closed a cycle and produced a committed
summary or explicit terminal marker:

1. add a new `closed_local_file` declaration with the immutable campaign/cycle and expected state;
2. pin any associated Git evidence to full commit IDs;
3. run the GInfer collector and inspect the new raw manifest rows;
4. add the closed ledger/history/context source IDs and split ownership to
   `config/trajectory_campaigns.json` (or author a seed annotation when it cannot be normalized
   safely); and
5. rebuild exports and run validation.

Do not infer closure from file age, process absence, or a partially populated CSV. Never collect
from the active SM86 Muse agent or another live candidate directory.

## Adding public NVIDIA material

Add an entry to `config/nvidia_sources.json` with exact URL, title, known version/date, MIME type,
SM applicability, source class, and license/use note. The fetcher uses conditional HTTP metadata
where available and stores bytes by SHA256. Update architecture cards only with claims actually
supported by a collected source revision.

NVIDIA AI Enterprise Suite material is excluded. Public NVIDIA manuals and developer guides are in
scope. This permission does not extend to third-party repositories.

## Third-party repositories

The third-party collector pins repository HEAD and retrieves only `LICENSE` and `README` metadata
until a reviewer confirms license compatibility. Code ingestion must be explicitly enabled per
path and revision. Unknown, missing, source-available-only, or incompatible licenses remain
metadata-only.

## Review gates

Before publishing an export:

- inspect `reports/validation.json` and require zero errors;
- verify all source and layer manifests contain SHA256 hashes;
- sample prompts for outcome leakage and secret/path disclosure;
- confirm related family IDs occupy one split;
- confirm candidate/revert pairs are linked;
- confirm eval prompts and private labels are separate; and
- confirm the campaign-builder report contains no unexplained omission class;
- tokenize the train split with the exact pinned Qwen3.8 tokenizer and inspect length/view/outcome
  balance; and
- record the BF16 base revision, Transformer Engine training configuration, distributed/offload
  layout, and post-training quantization recipe without changing canonical episodes.
