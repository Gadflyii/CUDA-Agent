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

The training target is Qwen3.8-27B delivered in NVFP4 form. The dataset remains provider-neutral
and does not encode tokenizer-specific chat tokens. It also does not assume that NVFP4 inference
weights are directly trainable. The training-method decision remains explicit and deferred:

- preferred baseline experiment: supervised/preference tuning from the supported higher-precision
  base, then calibrated NVFP4 quantization and regression evaluation;
- alternative experiment: an NVFP4-aware adapter or quantization-aware path if the selected stack
  can demonstrate stable updates and faithful adapter merging.

Both paths consume the same canonical events and must be compared on held-out technical behavior,
not just training loss.

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
2. `ginfer_closed_local`: ignored local ledgers only when a committed result or explicit terminal
   record establishes that the cycle is closed;
3. `ginfer_live_reference`: active profile/campaign locations recorded for later refresh, never
   copied while open;
4. `nvidia_official`: public NVIDIA manuals, guides, technical articles, and tool documentation;
5. `third_party_licensed`: pinned repository material whose license permits the intended use;
6. `third_party_metadata_only`: URL, revision, and license status without code ingestion; and
7. `authored_structure`: schemas, architecture cards, cross-SM deltas, and task annotations, each
   pointing back to evidence sources.

Every raw object records the source ID, exact URL or repository/path/revision, retrieval timestamp,
local relative path, byte size, SHA256, MIME type, declared version/date when available, source
class, license/use status, and SM applicability. Git diffs use full commit IDs. A revert is linked
to its candidate; it is never normalized as an independent positive implementation.

Raw snapshots are immutable and content addressed. A changed online document creates a new object
and registry revision. Normalized and curated manifests record the collector/exporter version and
all input hashes, allowing exports to be reproduced without silently following `HEAD` or a mutable
web page.

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

## Architecture-aware splitting

The indivisible split key is the transitive closure of:

- candidate commit and restoration commit;
- all repaired commits belonging to the same mechanism;
- accepted baseline and immediately derived candidates;
- same Op/shape/dispatch mechanism within one campaign cycle;
- all task views derived from the same evidence; and
- substantially identical patches ported across SMs.

The initial policy uses three evaluation tiers:

1. **in-architecture generalization:** unseen candidate families on an architecture present in
   training;
2. **cross-architecture transfer:** whole SM/route groups held out (initially SM86 and SM120a are
   evaluation-only until enough independent groups exist); and
3. **real-hardware qualification:** sealed workloads run on RTX 3090, RTX 4090, RTX 5090, and RTX
   PRO 6000, with the two SM120a SKUs kept as separate device targets.

No split is produced by random rows. Dataset balancing caps repeated no-op tile variations so one
24-candidate cycle cannot dominate. Final test prompts and device measurements are versioned and
frozen before model selection. If a mechanism appears on several architectures, all direct ports
remain together unless the evaluation is explicitly a documented cross-SM transfer test.

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

