---
name: ginfer-kernel-optimization
description: Optimize, port, and qualify handwritten C++ and CUDA kernels for real GInfer inference, including 120K-131K contexts, fixed concurrency from one to eight requests, Muse Glimmer, and every supported NVIDIA SM family. Use for native GInfer Op, CUDA Graph, whole-round, or end-to-end performance work; not for PyTorch extensions, generic CUDA examples, model conversion alone, or unrelated C++ features.
metadata:
  short-description: Optimize native GInfer CUDA kernels
---

# GInfer Kernel Optimization

Improve production C++/CUDA execution in the isolated GInfer optimization worktree. Preserve the
complete numerical and state-transition contracts while maximizing real inference performance for
the supported model, context, concurrency, and GPU matrix. There is no PyTorch extension,
`torch.compile` baseline, synthetic exercise harness, or universal percentage target.

## Use the isolated optimization worktree

The normal workspace is:

```text
/ai/ginfer/.worktrees/agentic-kernels/   branch perf/agentic-kernels
├── build-86/                           sm_86 image only
├── build-89/                           sm_89 image only
├── build-120a/                         sm_120a image only
└── profiles/
    ├── sm86/rtx3090/
    ├── sm89/rtx4090/
    └── sm120a/
        ├── rtx5090/
        └── rtx-pro-6000/
```

The branch starts from the integration point containing both Muse Glimmer and SM-family support.
Treat `/ai/ginfer` and its other feature worktrees as read-only sources of history and comparison
unless the user explicitly assigns work there.

The branch isolates history; the worktree isolates checked-out files and build state. Both are
required. One dedicated worktree per machine is sufficient for this optimization line—do not make
copied source folders for individual kernels. Create another worktree only when a genuinely
concurrent branch of source changes must be developed independently.

At the start of work:

1. Read this worktree's `AGENTS.md` in full.
2. Confirm the path, `perf/agentic-kernels` branch, base ancestry, and `git status`. Preserve every
   pre-existing change.
3. Keep source, CMake caches, generated code, benchmarks, and profiler output inside this worktree.
   Never reuse a build directory from `/ai/ginfer` or another worktree.
4. Use the three per-SM build roots above. A CMake build tree belongs to exactly one source
   worktree, toolchain, build type, and CUDA architecture.
5. Put disposable Nsight reports and raw measurement output under the matching ignored
   `profiles/sm*/` tree. Put only reviewed, durable results in the repository when the task calls
   for them.

Record the branch-point commit and measure the clean baseline before the first optimization. If a
later repeatable A/B comparison needs the old source, use a separate detached baseline worktree or
preserved baseline binary; do not switch this worktree or rebuild another developer's checkout.
Do not merge, rebase, publish, or delete the optimization branch unless the user asks.

## Product performance envelope

The ultimate target is one GPU and one resident model serving a startup-fixed compact batch of one
through eight active requests. Optimize real inference and agentic operations, not isolated
KernelBench-like shapes.

Treat these as first-class dimensions:

- effective context and KV position from approximately 120,000 tokens through the model's 131,072
  boundary, distinguishing configured capacity from tokens actually resident or attended;
- every active concurrency value `C=1..8`, including mixed per-request positions and valid lengths
  when the affected Op consumes them;
- latency at `C=1`, scaling and throughput at `C=2..8`, and the absence of pathological tail or
  route-boundary regressions;
- prefill, long-context decode, and the enabled speculative route (MTP or DFlash) when they reach
  the changed implementation;
- CUDA Graph capture/replay, prefix reuse, INT8/BF16 KV formats, and state transaction paths when
  they are part of the affected public contract;
- every registered target that can reach a shared Op, including Muse Glimmer where applicable.

Use INT8 KV for the 120K-131K product points where that is the registered capacity path. Do not
claim a long-context result from a large configured capacity with a short resident sequence. Do
not use only `C=1` and `C=8` to hide an interior batch-size dispatch cliff.

An Op benchmark remains necessary to tune a kernel, but it is not sufficient evidence for a claim
about full inference, speculative-round latency, long-context behavior, or concurrent throughput.
Measure the highest public boundary named by the task after the Op-level implementation stabilizes.

Before implementation, state the concrete claim being pursued:

- public Op, target round, or public Engine boundary;
- model/artifact, AR/MTP/DFlash mode, KV format, graph mode, and prefix state;
- exact token extents, context position/capacity, and `C=1..8` points in scope;
- GPU SKU, exact SM image, CUDA/toolchain, cache condition, and baseline command.

Infer these facts from the request and repository when possible. Ask only when a missing choice
would materially change the implementation.

## Read only the live authorities

After `AGENTS.md`, read the documents governing the affected boundary:

- `include/ginfer/ops/<family>.h` for the complete semantic contract;
- `docs/maintainer/op-development.md` for ownership, qualification, and performance evidence;
- `docs/maintainer/concurrent-inference-architecture.md` and `paged-kv-cache.md` when concurrency,
  long context, compact batching, prefix reuse, or KV state is involved;
- `tests/README.md` and the affected `tests/ops/` suite for correctness and oracle style;
- `bench/README.md` and the affected Op or target benchmark for measurement behavior;
- the relevant target/model reference for model mathematics or schedule composition;
- the active feature plan and latest comparable result when an unfinished target such as Muse
  Glimmer is not yet integrated into the stable model references;
- root/source CMake files and each selected build's `CMakeCache.txt` for architecture ownership.

Do not load every model document or create a parallel authority that can drift from GInfer.

## Load CUDA knowledge selectively

The references below are a compact decision aid, not a replacement for the live GInfer contract,
the installed CUDA documentation, or measurements from the target GPU. Read only what the current
decision needs:

- After the clean baseline and before the first candidate, read
  [references/hardware-feedback-loop.md](references/hardware-feedback-loop.md). It defines the
  evidence packet, single-hypothesis refinement loop, and acceptance memory used throughout a run.
- For a multi-cycle campaign or a handoff between GPUs or model targets, also read
  [references/campaign-orchestration.md](references/campaign-orchestration.md). It defines the
  worktree preparation, candidate accounting, current 24-candidate cycle policy, durable ledger,
  restoration, and terminal stop condition.
- Before changing device code, read
  [references/cuda-kernel-design.md](references/cuda-kernel-design.md) for execution, memory,
  resource, and inference-pattern guidance.
- When a profile is needed to choose the next change, read
  [references/nsight-compute-triage.md](references/nsight-compute-triage.md). Collect the smallest
  metric subset that distinguishes the live hypotheses; do not dump every profiler metric into the
  reasoning context.
- Before adding an SM-specific path, changing a tile/resource policy across architectures, or
  interpreting cross-GPU results, read
  [references/sm86-sm89-sm120a.md](references/sm86-sm89-sm120a.md).
- Read [references/ptx-and-sass.md](references/ptx-and-sass.md) only when disassembly and profiling
  support an instruction-level change. Handwritten PTX is a late, narrow tool, never the default
  response to a slow kernel.

These references intentionally summarize and route to authoritative sources rather than copying
whole manuals. If an installed toolkit, current NVIDIA document, or GInfer invariant conflicts
with a summary, verify the live authority and update the summary before relying on it.

## Trace the production path

Map the complete affected route before changing a kernel:

```text
public semantic contract
        -> wrapper validation and finite dispatch
        -> launcher and launch policy
        -> CUDA kernel/private device primitives
        -> target schedule, when the claim includes one
        -> public Engine route, when the claim is end-to-end
```

Find every caller and dispatch predicate. Confirm dtype, format, logical and padded shapes,
alignment, aliasing, workspace, stream ordering, graph capture, context/KV indexing, batch/slot
mapping, and supported extents. For stateful or fused work, include every output, mutation, and
observable cast boundary.

Keep ownership intact:

- contracts remain under `include/ginfer/ops/`;
- wrappers validate and select from semantic, geometry, and device facts;
- launchers own grids, blocks, shared memory, and template instantiation;
- kernels own device computation;
- target packages own model schedules and persistent state, not reusable Ops;
- targets include public Op contracts, never private launcher or kernel headers.

Do not dispatch an Op on model identity, artifact object name, or Program phase. A genuinely
Muse-only scheduling step stays in `src/targets/muse_glimmer_30b`; a semantically closed device
transformation belongs in the central Op layer even if Muse is its first caller.

## Establish correctness and baseline evidence

Use GInfer's native CMake, CTest, and benchmark infrastructure. Configure Release builds with one
exact `CMAKE_CUDA_ARCHITECTURES` image each. The current campaign covers `86`, `89`, and `120a`;
also verify the repository allowlist in case the supported set changes.

Before editing:

1. Build the smallest production target, focused test, and benchmark that exercise the route in
   the local GPU's exact-SM tree.
2. Run the focused correctness test through the public contract.
3. Measure a stable baseline at the claim boundary and relevant context/concurrency points.
4. Profile only enough to identify a decision-changing bottleneck. Use Nsight Systems for launch,
   synchronization, graph, or CPU/GPU timeline questions; use Nsight Compute for a specific kernel
   question.

If no suitable benchmark exists and performance at that boundary is part of the deliverable, add
a repository-native benchmark that calls the public contract. Do not use generated output quality,
another production route, or another kernel as the correctness oracle.

## Optimize from evidence

Form one concrete, falsifiable hypothesis tied to the measured bottleneck. Record the evidence,
predicted effect on the scoped matrix, narrow candidate change, and result that would disprove the
hypothesis before editing. Keep diagnosis and implementation as distinct phases; when the user has
authorized parallel agents, a read-only judge may produce the diagnosis packet while the coding
agent implements it. The benchmark and correctness suite remain the judges of fact.

Then implement the strongest coherent solution for that hypothesis. Relevant techniques can
include fusion, fewer memory passes, architecture-appropriate layouts and vector movement,
asynchronous copies, warp specialization, Tensor Core MMA, persistent state, occupancy-aware
launches, graph capture, and removal of avoidable host synchronization. Select techniques from the
real arithmetic, long-context access pattern, compact batch geometry, and GPU rather than a fixed
checklist.

Do not force one lowest-common-denominator kernel or schedule across SM families. Shared semantic
code and primitives are useful when performance remains strong, but independent compile-time
implementations, tile shapes, or dispatch thresholds are preferred when Ampere, Ada, and Blackwell
need different execution strategies.

For several plausible candidates, compile the small decision set together and measure one
candidate-by-extent-by-concurrency matrix under identical conditions. Temporary sweeps may call
private launchers only as permitted by `op-development.md`. Encode the measured winner or
crossover in production dispatch, qualify it again through the public contract, and delete losing
candidates and comparison-only entry points.

Do not accept a speedup caused by skipped work, stale output, weaker semantics, unrequested
precision loss, different cache state, reduced resident context, changed prompt/token count,
disabled validation, or a narrower timing boundary. Do not widen a tolerance to make a candidate
pass. Evaluate both the pointwise matrix and aggregate latency/throughput; an average must not hide
a material regression at one concurrency, context, or dispatch seam.

## Qualify every supported SM

For every changed CUDA or dispatch path:

- build `build-86`, `build-89`, and `build-120a` as separate compile images;
- gate architecture-specific instructions and translation units at compile time and retain a
  deliberate implementation or rejection path for every supported image;
- select runtime routes from explicit capability and geometry facts, never marketing names;
- derive occupancy from the live device policy rather than a hard-coded SM count;
- keep artifact-format restrictions explicit at admission;
- run correctness and the performance matrix on physical hardware for each SM before making an
  all-SM claim.

Compilation on one host proves only that a non-local image builds. Runtime results from the RTX
5090 (`sm_120a`) do not validate the RTX 4090 (`sm_89`) or RTX 3090 (`sm_86`). A task scoped to one
SM may finish with the other images compile-checked and clearly reported as pending runtime work;
the overall optimization campaign is not complete until all four physical host/SKU matrices
exist. The RTX 5090 and RTX PRO 6000 use the same `sm_120a` image but remain separate performance
targets because their live SM count, memory capacity, and bandwidth differ.

The physical validation hosts are the local RTX 5090, SSH-accessible Server 1 RTX PRO 6000
Blackwell, WSL RTX 4090, and Server 2 RTX 3090. When running non-local validation, read
[references/remote-sm-validation.md](references/remote-sm-validation.md) before changing any
remote checkout. The remote machines may pull, build, test, benchmark, and profile this branch;
their existing active GInfer checkout remains untouched.

For shared Ops, qualify every affected route and ensure a Muse- or SM-specific win does not
silently regress other registered callers. Restrict expensive end-to-end runs to models that
actually reach the changed path.

## Correctness requirements

Qualify floating-point work against the suite's independent FP32/FP64 mathematical oracle and
exact transforms against an independent exact oracle. Cover real registered shapes, all affected
dispatch boundaries, concurrency-sensitive indexing, long-context boundaries, and graph
capture/replay when applicable. Stateful Ops must verify returned values and the full state
transition; fused Ops need an oracle for the complete fused formula.

Use the suite-owned criterion appropriate to the arithmetic profile. Pairwise parity between two
production implementations is supplementary evidence only. Add cases for every new alignment,
tail, route seam, device capability, context boundary, or compact-batch assumption.

## Iterate and finish

After each material candidate:

1. build the focused local-SM target;
2. run the focused conformance suite;
3. remeasure identical context/concurrency points and inspect the full curve or round breakdown;
4. retain the change only when the result is repeatable and improves the requested objective
   without a material unsupported regression;
5. compile all three SM images and run broader affected tests after the design stabilizes;
6. use the remote validation workflow to run the same matrix on the other physical GPUs.

Finish a scoped task when its requested boundary is correct, its performance claim is supported,
and no known in-scope issue prevents use. Remove scratch binaries, logs, forced selectors,
temporary controls, and losing implementations. Keep legitimate per-SM routes and permanent
public benchmarks.

Report:

- the bottleneck and implementation decision;
- before/after results with units and relative change at each scoped context/concurrency point;
- exact model, hardware, SM image, toolchain, KV/graph/cache mode, and commands;
- focused correctness plus all-SM compile results;
- physical host/SKU runtime matrices completed and still pending;
- any unsupported route or material regression.
