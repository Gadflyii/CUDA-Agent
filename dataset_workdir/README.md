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

## Current corpus builder

`config/trajectory_campaigns.json` admits only explicitly closed SM89, SM86, and SM120a campaigns.
`scripts/build_campaign_episodes.py` joins their ledgers to immutable candidate/revert histories,
applies later corrections, reconstructs exact implementation patches plus pre-change source hunks,
and derives leakage-safe diagnosis, implementation, judgment, and orchestration views. It refuses
to invent missing commands or restoration edges. Live cycles, full terminal chatter, profiler
databases, model outputs, build products, credentials, and unclosed campaign files remain excluded.

The generated report at `reports/trajectory_builder.json` gives the candidate/view/split counts and
every omission reason. Auto-normalized rows require sampled review before a release training run.
Official NVIDIA material and license/provenance records remain reference evidence; third-party code
is not imported unless its pinned license permits the intended use.

## Training and deployment boundary

The planned specialization is a full-parameter Qwen3.8-27B fine-tune on both 96 GB RTX PRO 6000
Blackwell GPUs in Server 1. Transformer Engine FP8 is the primary compute path, with full state
sharding, optimizer/master-state offload as required, activation recomputation, and a matched BF16
control. The durable training authority is a consolidated BF16 safetensors checkpoint.

After BF16 evaluation, NVIDIA ModelOpt is the production Blackwell NVFP4 authority; AutoRound is an
independent candidate. `scripts/build_nvfp4_calibration.py` packs exactly 512 training-only 4096-token
rows with the pinned tokenizer. GInfer then admits only its pinned MLP0-55 ModelOpt profile and
builds the complete `.ginfer` release with exact checkpoint provenance. Arbitrary ModelOpt, LLM
Compressor, and AutoRound exports remain rejected. See
[`DESIGN.md`](DESIGN.md#full-fine-tuning-on-server-1) for the memory model, bring-up gates, training
sequence, and export qualification.
