# Dataset design

## Objective and target behavior

The model should behave like a disciplined GInfer kernel engineer, not a generic CUDA answer bot.
Given a live semantic contract, exact workload, hardware image, source context, and measured
evidence, it should:

1. map the public Op or Engine route through dispatch, launcher, device code, and target schedule;
2. distinguish correctness requirements from implementation freedom;
3. diagnose a bottleneck from the smallest decisive profile or benchmark evidence;
4. state one falsifiable candidate with an expected effect and rejection observation;
5. produce narrow, ownership-correct C++/CUDA patches with explicit SM guards and dispatch seams;
6. select focused compilation, independent-oracle, public-Op, Program-round, and Engine checks;
7. judge candidates from pointwise and real-boundary evidence, including rejecting attractive
   microbenchmark wins that cannot clear the product gate;
8. preserve candidate accounting and restoration links; and
9. report claims with exact workload, GPU/SKU, SM image, toolchain, units, limitations, and pending
   physical qualification.

The specialization target is a **full-parameter fine-tune of Qwen3.8-27B**. Training starts from the
exact BF16 safetensors checkpoint and emits a resumable BF16 checkpoint. Transformer Engine FP8 is
the primary training-compute path, not the checkpoint or artifact authority: trainable/master
weights and optimizer state retain their required higher precision. A short BF16-compute control
must establish that FP8 has equivalent convergence and materially better step time on Server 1.

The accepted BF16 checkpoint is then quantized for deployment. ModelOpt NVFP4 and AutoRound are
separate candidates, each evaluated against the BF16 authority before conversion to a registered
`.ginfer` identity. Quantized results are never treated as the reference for training correctness.
The dataset remains provider-neutral and does not encode tokenizer-specific chat tokens; the
training input builder applies the exact pinned Qwen3.8 tokenizer and chat template.

## Canonical episode

`schemas/canonical_episode.schema.json` defines one evidence-grounded episode. Its major fields are:

- stable episode and family IDs;
- task view, target/model/Op, architecture scope, and split group;
- immutable source snapshots and licenses;
- ordered observable events with actor, event type, information class, payload, and evidence refs;
- a view declaration listing model-input, target, and withheld event IDs;
- structured outcome, correctness, performance, and restoration data; and
- quality, leakage, and review annotations.

Events are provider-neutral. Exporters may convert them to `messages`, prompt/completion, pairwise,
or evaluation formats. The corpus records concise technical rationales that were externally
observable in campaign diagnoses, commit diffs, test choices, measurements, and decisions. It must
never solicit, reconstruct, or label hidden chain-of-thought. A good rationale names facts,
hypotheses, alternatives considered, and evidence thresholds without pretending to expose private
reasoning traces.

Each evidence item has one information class:

- `task_context`: contract, source surface, workload, and constraints known at task start;
- `pre_candidate_evidence`: baseline/profile facts available before a candidate was written;
- `candidate_artifact`: a proposed patch or command;
- `post_candidate_result`: compile, oracle, benchmark, or runtime evidence produced afterward;
- `disposition_label`: accept/reject/inconclusive and its justification;
- `restoration`: revert linkage and restored-source verification; or
- `report`: a completed evidence-grounded summary.

This temporal classification is the primary leakage control.

## Source taxonomy and provenance

Sources are registered before use and pinned wherever possible:

1. `ginfer_committed`: immutable Git objects, committed runbooks, tests, results, and diffs;
2. `cuda_agent_committed`: immutable CUDA-Agent skill-policy Git objects that governed the
   collected GInfer campaign;
3. `ginfer_closed_local`: ignored local ledgers only when a committed result or explicit terminal
   record establishes that the cycle is closed;
4. `ginfer_live_reference`: active profile/campaign locations recorded for later refresh, never
   copied while open;
5. `nvidia_official`: public NVIDIA manuals, guides, technical articles, and tool documentation;
6. `third_party_licensed`: pinned repository material whose license permits the intended use;
7. `third_party_metadata_only`: URL, revision, and license status without code ingestion; and
8. `authored_structure`: schemas, architecture cards, cross-SM deltas, and task annotations, each
   pointing back to evidence sources.

Every raw object records the source ID, exact URL or repository/path/revision, retrieval timestamp,
local relative path, byte size, SHA256, MIME type, declared version/date when available, source
class, license/use status, and SM applicability. Git diffs use full commit IDs. A revert is linked
to its candidate; it is never normalized as an independent positive implementation.

Raw snapshots are immutable and content addressed. A changed online document creates a new object
and registry revision. Normalized and curated manifests record the collector/exporter version and
all input hashes, allowing exports to be reproduced without silently following `HEAD` or a mutable
web page.

## Closed-campaign trajectory builder

`config/trajectory_campaigns.json` is the explicit admission list for automatic trajectory
construction. Each entry names closed ledger snapshots, immutable Git histories, governing
context, later corrections, exact target/hardware/workload scope, and cycle-level split ownership.
The builder never scans a checkout, guesses that a cycle is closed, or consumes a live-reference
source.

`scripts/build_campaign_episodes.py` performs these deterministic transformations:

1. resolve every configured source through the registry and raw manifest;
2. parse the heterogeneous closed CSV ledgers while retaining the source row identity;
3. resolve candidate and restoration commit prefixes only when the declared immutable Git history
   makes them unique;
4. require an exact `reverts_commit` edge before attaching a rejected commit to an implementation
   outcome;
5. retain only repository code/test/build hunks and reconstruct pre-change hunk context without
   exposing the candidate's added lines;
6. apply recorded correction rows before generating an outcome;
7. emit diagnosis only when pre-candidate baseline and hypothesis evidence exist, implementation
   only when an exact patch and preimage exist, judgment for every usable ledger row, and
   orchestration for accepted or terminal no-win boundaries; and
8. structurally fingerprint patches, reconcile direct ports to the most restrictive split, run
   schema and secret checks, and merge the generated episodes with the reviewed seed.

Exact tool commands are emitted only when an immutable source actually records them. Current
ledgers normally record the selected checks and their results rather than byte-exact shell calls,
so the implementation target contains the exact code diff plus a focused verification decision;
it does not fabricate a tool transcript. Oversized patches or missing/revert-unproven history are
omitted from implementation views and retained, with limitations, only where they remain valid
judgment evidence. `reports/trajectory_builder.json` records every omission class and distribution.

## Leakage-safe task views

The same historical campaign can yield several views, but all views share one `family_id` and
therefore one split.

| View | Model input | Training target | Withheld |
|---|---|---|---|
| diagnosis | contract, source map, baseline/profile | bottleneck, falsifiable hypothesis, next decisive measurement | candidate diff, candidate results, decision |
| implementation | contract, diagnosed hypothesis, relevant source/tests | patch plus focused verification plan | compile/oracle/performance outcomes, disposition |
| judgment | candidate patch and already-observed gate results | accept/reject decision and evidence-based explanation | historical label until scoring |
| orchestration/reporting | completed candidate records and campaign controls | next gate/stop action or concise result report | future cycles and unrelated campaigns |

For `candidate_generation=true`, input events may contain only `task_context` and
`pre_candidate_evidence`; the validator rejects candidate results, labels, and restoration records.
Outcome words are not banned from all natural language—contracts may legitimately say what would
cause rejection—but historical result/label event IDs and evidence refs must be withheld.

Evaluation has a public prompt file and a separate private answer/label file. Training tooling must
never mount private eval labels into prompt construction.

## Derived products

- **SFT:** input events become a user/system prompt; target events become a concise assistant
  response containing technical rationale, tool choices, patches/commands, and a claim-calibrated
  report. Hidden reasoning is absent.
- **Preference:** chosen and rejected responses share byte-identical visible inputs. Rejected
  responses are real hard negatives where possible: correctness failures, unsupported SM transfer,
  microbenchmark-only wins, sub-threshold gains, seam regressions, or false all-SM claims.
- **Evaluation:** prompt, rubric, required evidence IDs, and private reference answer/decision are
  separate. Metrics combine contract correctness, patch applicability, oracle selection, evidence
  calibration, pointwise performance judgment, and real-device qualification.

Preference pairs are not manufactured by flipping an accepted label. A rejected patch can be a
valuable hard negative only if its semantic context and actual measured failure are preserved.
Near-duplicate variants are grouped, and their outcome evidence is hidden in generation views.

## Full fine-tuning on Server 1

### Hardware and memory contract

Server 1 supplies two RTX PRO 6000 Blackwell GPUs with 96 GB each. They are two independent memory
domains, not an automatically unified 192 GB allocation. Before selecting the distributed layout,
record the exact SKU, driver/CUDA/Transformer Engine versions, `nvidia-smi topo -m`, peer-access
result, host RAM, NUMA placement, and available local NVMe capacity. The topology measurement
decides whether tensor-parallel collectives or fully-sharded all-gathers are the faster two-GPU
layout; it is not inferred from the product name.

A conventional 27B AdamW full tune is approximately 432 GB before activations: about 54 GB BF16
parameters, 54 GB BF16 gradients, 108 GB FP32 master weights, and 216 GB FP32 first/second moments.
Perfect two-way sharding would still be about 216 GB per GPU. Transformer Engine FP8 reduces and
accelerates eligible linear compute and some working tensors, but does not remove the authoritative
weights or optimizer state. Therefore the run requires all of the following:

- two-way parameter and gradient sharding;
- CPU offload of FP32 master weights and Adam moments, unless a separately validated reduced-state
  optimizer proves both fit and equivalent convergence;
- transformer-block activation recomputation;
- packed variable-length sequences and microbatch one per GPU; and
- resumable distributed checkpoints that materialize a complete BF16 Hugging Face checkpoint.

The primary stack is NVIDIA Megatron Core/Bridge with Transformer Engine. Exact Qwen3.8 conversion
is an admission gate: load the pinned BF16 checkpoint, round-trip one distributed checkpoint, and
compare representative logits/state outputs before training. General Qwen support in a framework
does not prove this exact hybrid-attention/MTP checkpoint mapping. If the selected Megatron release
cannot express the required full sharding plus offload, use an equivalent supported full-shard
runtime with Transformer Engine rather than weakening the full-fine-tune requirement.

### Bring-up and training sequence

1. **Freeze data and base.** Pin the BF16 model revision, tokenizer/template hashes, normalized and
   curated manifest hashes, and split policy. Private validation/test answers are absent from the
   training mount. Auto-normalized training rows receive sampled review before release training.
2. **Build token statistics.** Tokenize with the pinned base tokenizer, preserve complete diffs,
   and report lengths by task view and split. Start with packed 8K sequences; admit 16K examples
   only after the memory probe. Long-context capability up to the base model limit is evaluated,
   not recreated by forcing every SFT step to 131K.
3. **Qualify execution.** Run a memory-only forward/backward probe, then matched BF16-compute and
   TE-FP8-compute steps. Require finite loss/gradients, checkpoint resume, BF16 logit parity before
   updates, and an FP8 throughput benefit after synchronization. Capture peak GPU/host memory and
   tokens/s; an out-of-memory fallback to LoRA is not allowed.
4. **Pilot the full tune.** Use AdamW, BF16 model/gradient authority, TE FP8 linear compute, gradient
   clipping, activation recomputation, and token-count-based accumulation. Begin the learning-rate
   selection with `1e-6`, `2e-6`, and `5e-6`; choose from held-out behavior and retention, not the
   lowest training loss. The initial effective batch target is 128K non-padding tokens/update,
   adjusted only for measured stability or memory.
5. **Run and checkpoint.** Save frequent resumable sharded checkpoints plus periodic consolidated
   BF16 safetensors. Evaluate diagnosis, exact-patch implementation, judgment, orchestration, and
   general tool/coding retention separately. Stop on held-out regression or overfit, not an
   arbitrary epoch count.
6. **Select in BF16.** The winning checkpoint must pass the frozen private evaluation and the
   sandboxed real GInfer task harness in BF16-capable reference execution before quantization.

The currently generated corpus is a valid builder output, not automatically a sufficient release
corpus. A production full tune is gated on reviewed positive and hard-negative coverage, token
volume, task-view balance, and a capability-retention set. The first run may be a memory/performance
and overfit-risk pilot; its success must not be reported as a trained production agent merely
because loss decreases.

### Quantization and `.ginfer` deployment

The selected BF16 checkpoint is immutable input to two downstream branches:

1. **NVIDIA ModelOpt NVFP4.** This is the Blackwell production authority. Build exactly 512 packed
   4096-token rows from the train split with the pinned Qwen3.8 tokenizer, then apply the checked-in
   GInfer MSE recipe to only Text MLP gate/up/down projections in layers `0..55`. ModelOpt emits
   W4A4 NVFP4: E2M1 packed values, one E4M3 scale per K16 block, and FP32 per-tensor weight/input
   multipliers. GInfer preserves the packed codes and block scales, maps the two multipliers to its
   reciprocal-divisor convention, and owns row-FP8 quantization for attention, GDN, layers `56..63`,
   embedding, and output head from the same BF16 checkpoint. Run converter verification, task
   evaluation, long-context sentinels, and real Engine performance on Blackwell.
2. **AutoRound.** Evaluate AutoRound NVFP4 and, where useful, its weight-only groupwise result from
   the same BF16 checkpoint. An AutoRound checkpoint is not relabeled as a current GInfer format:
   it must either match a registered tensor/scale contract exactly or receive an explicit new
   converter/identity with its own oracle and kernels.

This export boundary is material. `qwen3_8_27b_nvfp4-modelopt-v2` accepts only the pinned ModelOpt
revision and recipe, exactly 168 NVFP4 source matrices, exact fused gate/up multiplier equality,
and `ginfer-quantization.json` hashes binding the packed export to the selected BF16 checkpoint.
ModelOpt `weight_scale_2` and `input_scale` are multipliers; the adapter stores
`round_f32(1 / scale)` because the registered GInfer kernels consume divisors. It never decodes or
requantizes ModelOpt FP4 codes or E4M3 block scales. The calibration manifest and exact
tokenizer/template hashes are required inputs, and the selected Python environment must import
ModelOpt from the clean pinned checkout rather than an unrelated installed build. The groupwise
converter independently applies GInfer's registered max-absolute Q4/Q5/W8 recipes to BF16 tensors.
An arbitrary ModelOpt,
LLM Compressor, or AutoRound directory is not loadable.

LLM Compressor is an interoperability/reference source, not the production quantizer. Its
compressed-tensors layout remains useful for comparing represented values and published artifacts,
but ModelOpt owns the trained-checkpoint Blackwell path because NVIDIA defines the NVFP4 W4A4
recipe, scale hierarchy, calibration algorithms, and unified export. AutoRound remains a separate
quality candidate and must enter through a registered format-specific producer.

Selection is pointwise: BF16 quality is the reference, then each quantized branch must clear the
same private behavioral suite and real GInfer agent tasks. Report quality deltas, artifact size,
peak VRAM, prefill/decode throughput, and failures separately. NVFP4 is the preferred Blackwell
production result; the groupwise branch remains a portability/fallback candidate, not a reason to
lower the NVFP4 quality gate.

Framework behavior is taken from the pinned versions installed for the run, with these upstream
documents as discovery authorities:

- [Transformer Engine user guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html);
- [Megatron Bridge performance and parallelism guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html);
- [Megatron Bridge mixed-precision training](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/mixed-precision.html);
- [Megatron Bridge / ModelOpt quantization](https://docs.nvidia.com/nemo/megatron-bridge/nightly/modelopt/quantization.html);
- [NVIDIA ModelOpt PTQ recipe guide](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md);
- [Transformer Engine NVFP4 format](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html); and
- [AutoRound source and format documentation](https://github.com/intel/auto-round).

The run manifest records the resolved package/container revisions; a mutable documentation URL is
not sufficient experiment provenance.

## Architecture-aware splitting

The indivisible split key is the transitive closure of:

- candidate commit and restoration commit;
- all repaired commits belonging to the same mechanism;
- accepted baseline and immediately derived candidates;
- same Op/shape/dispatch mechanism within one campaign cycle;
- all task views derived from the same evidence; and
- substantially identical patches ported across SMs.

The current closed-campaign policy uses three evaluation tiers:

1. **in-architecture generalization:** unseen candidate families on an architecture present in
   training;
2. **cross-architecture transfer:** the complete SM86 campaign remains held out while SM89 and
   earlier SM120a cycles supply training evidence; and
3. **real-hardware qualification:** sealed workloads run on RTX 3090, RTX 4090, RTX 5090, and RTX
   PRO 6000, with the future RTX PRO 6000 results never inferred from RTX 5090 despite the shared
   SM120a image.

No split is produced by random rows. SM120a Muse cycle 7 and Qwen cycle 7 are in-architecture
validation; Qwen cycles 8-9, both consolidation campaigns, and all SM86 candidates remain
test-owned.
Structural patch fingerprints move direct ports to the most restrictive related split. Dataset
normalization retains every admitted ledger row. The eventual training loader applies the declared
near-duplicate/no-win sampling caps after exact-tokenizer length statistics are available, so one
24-candidate cycle cannot dominate; validation and test remain complete. Final test prompts and
device measurements are versioned and frozen before model selection.

## Quality filters

An episode is eligible only when it has:

- an authoritative semantic boundary and non-ambiguous model/Op scope;
- immutable source identifiers and compatible use rights;
- a temporal evidence cutoff;
- enough workload detail to interpret performance claims;
- an independent oracle for numerical/exact claims, or an explicit `not_observed` limitation;
- units and direction for measurements;
- SM image distinguished from GPU SKU and physical runtime host;
- a candidate/restoration link for rejected committed changes;
- no secrets, host credentials, private keys, tokens, artifact contents, or unredacted user data;
- no raw chain-of-thought request or purported reconstruction; and
- human or rule-based review status.

Low-information terminal chatter, repeated compiler noise, stochastic model prose, huge profiler
dumps, ungrounded optimization advice, and results with mismatched prompt/context/cache conditions
are excluded. Generated text is semantic-sentinel evidence only, never a kernel oracle.

## Deduplication and pruning

Files are byte-deduplicated by SHA256. Text receives a normalized fingerprint after line-ending,
boilerplate, and unstable-path normalization. Episodes receive structural fingerprints over
contract, Op family, mechanism, shape domain, SM, and patch hunks. Exact duplicates collapse to one
record with multiple provenance refs. Near-duplicates are clustered and capped by outcome and SM.

Keep evidence diversity: accepted changes, correctness failures, compile failures, real-boundary
misses, seam regressions, neutral results, and architecture-specific ports. Prune repeated variants
that teach no new mechanism or decision boundary. Never prune the restoration edge or the one row
that establishes a terminal no-win cycle.

## Fact classes

Facts must not be blended:

- **architecture-immutable facts** come from versioned NVIDIA architecture/ISA documentation, such
  as compute-capability-defined instruction support or documented resource limits;
- **compiler-derived facts** come from a named toolchain and exact build, such as register count,
  local memory, generated SASS, launch bounds, or compile success for `sm_86`, `sm_89`, or
  `sm_120a`; they can change with compiler flags or version; and
- **device-measured facts** come from a physical SKU/run, such as latency, achieved occupancy,
  bandwidth, clocks, memory availability, or a profiler counter. They never transfer merely
  because two devices accept the same image.

Architecture cards store these in separate arrays. Compiler- and device-derived records require a
toolchain/build or host/SKU/workload key. The cards prohibit turning a measured RTX 4090 crossover
or an RTX 5090 SM count into an immutable architecture rule.

## Real-hardware evaluation

The final model is evaluated by a sandboxed agent harness that can inspect a frozen GInfer tree,
write only to a disposable candidate worktree, and propose patches. Candidate patches are applied
and judged outside the model by native GInfer tooling:

1. exact-SM compile;
2. independent numerical/exact oracle at real shapes and seams;
3. public-Op matrix;
4. target Program or public Engine boundary with repeated measurements;
5. unaffected concurrency/context/dispatch controls; and
6. semantic Engine sentinels and registered real tests.

Runtime claims require physical hardware. `sm_86`, `sm_89`, and `sm_120a` compilation alone is not
runtime qualification. RTX 5090 and RTX PRO 6000 results remain separate despite sharing an
`sm_120a` image. Evaluation reports correctness pass rate, patch/build rate, accepted-candidate
rate under frozen gates, geomean speedup only across valid tasks, worst pointwise regression,
false-claim rate, and campaign efficiency (useful accepted candidates per expensive run).
