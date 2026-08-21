# Collection and export runbook

All commands run from this directory. Collectors only read declared sources and write beneath this
workspace. They do not fetch, switch, build, test, benchmark, or profile a GInfer checkout.

```bash
python3 scripts/collect_ginfer.py
python3 scripts/fetch_public.py
python3 scripts/build_normalized.py
python3 scripts/export_dataset.py
python3 scripts/validate.py
```

Use `python3 scripts/pipeline.py` for the same sequence. Re-running is idempotent. Existing raw
objects must match their recorded hash; conflicts fail closed. Network fetches preserve a new
content-addressed revision instead of replacing an older object.

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
4. author/review normalized episode annotations; and
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
- record the chosen base-model/training/quantization method without changing canonical episodes.
