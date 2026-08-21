# GInfer CUDA optimization dataset

This workspace builds a provenance-rich dataset for specializing a Qwen3.8-27B model in GInfer
handwritten C++/CUDA kernel optimization. It is intentionally separate from every GInfer checkout,
build tree, profile directory, branch, and remote host. All GInfer inputs are read-only snapshots.

Start with [`DESIGN.md`](DESIGN.md), then use [`RUNBOOK.md`](RUNBOOK.md). The canonical format is
provider-neutral. SFT, preference, and evaluation formats are derived products and may be rebuilt.

## Layout

```text
config/                 immutable-source declarations and split policy
schemas/                JSON Schemas for source, manifest, episode, and architecture records
scripts/                idempotent collectors, normalizer/exporter, and validator
sources/registry.jsonl  authoritative source registry
raw/                    immutable byte snapshots plus content hashes
normalized/             canonical events, architecture cards, and cross-SM deltas
curated/                SFT, preference, and evaluation exports
reports/                generated validation and collection reports
```

The raw layer is append-only: a collector accepts an already-present byte-identical object and
refuses a conflicting overwrite. Every file has a SHA256 record. Normalization retains source IDs,
immutable revision identifiers, licenses, evidence timing, and candidate/restoration links.

## Current seed

The first collection targets:

- the committed SM89 Muse/Qwen campaign summary and terminal ledgers;
- completed SM89 Muse cycle 7-9 ledgers, copied only from explicitly closed cycle directories;
- Git candidate/revert history and diffs at immutable commits;
- the two committed, accepted SM86 changes as implementation evidence, without inventing missing
  runtime measurements;
- the committed SM120a accepted change as implementation provenance while treating the following
  local campaign ledger as live and refresh-only;
- official public NVIDIA programming, tuning, PTX, occupancy, Tensor Core/data-movement, and Nsight
  material; and
- license/provenance records for selected third-party repositories. Third-party code is not
  imported unless the repository exposes a compatible license at the pinned revision.

This seed is deliberately small. Full terminal chatter, profiler databases, model outputs, build
products, credentials, and active campaign files are excluded.

## Training method boundary

The canonical dataset does not assume a trainer or weight representation. A later experiment must
choose between fine-tuning a higher-precision Qwen3.8-27B base and quantizing to NVFP4, or using an
NVFP4-aware adapter/quantization-aware path. That choice affects optimization stability, memory,
and export tooling, not the evidence schema or leakage policy, and therefore does not block corpus
construction.

