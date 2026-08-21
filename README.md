# GInfer Kernel Optimization Skill

This repository packages a Codex skill for improving the handwritten C++ and CUDA execution paths
in the GInfer inference engine at `/ai/ginfer`. It replaces the original PyTorch-extension exercise
and works against GInfer's real Op contracts, native tests, benchmarks, model artifacts, CUDA Graph
paths, and architecture-specific build images.

The skill is designed for work such as:

- optimizing an existing GInfer Op or fused CUDA path;
- removing a bottleneck from Muse Glimmer AR or DFlash2 inference;
- tuning long-context inference at approximately 120K through 131,072 tokens;
- optimizing the complete startup-fixed concurrency range from one through eight requests;
- porting and tuning groupwise-int kernels for every supported GPU family;
- adding a capability-specific implementation without regressing existing SM images; and
- proving an Op-level, whole-round, or end-to-end speedup with native correctness evidence.

## Layout

```text
agent_workdir/
├── SKILL.md                         # Agent workflow, routing, and GInfer constraints
├── agents/openai.yaml               # Skill display metadata
└── references/
    ├── campaign-orchestration.md    # Worktree preparation, cycle budget, and stop policy
    ├── hardware-feedback-loop.md     # Evidence packet and bounded refinement loop
    ├── nsight-compute-triage.md      # Small-metric profiling and diagnosis map
    ├── cuda-kernel-design.md         # Memory, execution, resources, inference patterns
    ├── sm86-sm89-sm120a.md           # Architecture-specific design constraints
    ├── ptx-and-sass.md               # PTX entry criteria, hazards, and qualification
    └── remote-sm-validation.md       # Isolated physical-GPU validation workflow
```

The reference files are curated decision aids, not local copies of CUDA manuals. `SKILL.md` routes
the agent to only the reference needed for the current decision and each reference links back to
the live NVIDIA or CudaForge source. Current toolkit documentation, device queries, GInfer
contracts, and measurements override a summary if they differ. The older SC09 optimization deck is
used only for durable concepts such as useful versus actual bytes and latency hiding; none of its
generation-specific constants are treated as current.

The implementation workspace is intentionally not copied into this repository. Kernel work has a
dedicated Git branch and worktree so it cannot modify the GInfer root checkout or the existing Muse
and SM-family feature worktrees:

```text
/ai/ginfer/.worktrees/agentic-kernels/   branch perf/agentic-kernels
├── build-86/                           isolated sm_86 CMake tree
├── build-89/                           isolated sm_89 CMake tree
├── build-120a/                         isolated sm_120a CMake tree
└── profiles/                           ignored raw results and profiler captures
    ├── sm86/rtx3090/
    ├── sm89/rtx4090/
    └── sm120a/
        ├── rtx5090/
        └── rtx-pro-6000/
```

`perf/agentic-kernels` was created from `feat/muse-glimmer`; that base already contains
`feat/sm-family-citizenship`, so the optimization branch starts with both Muse Glimmer and the
`sm_86`/`sm_89`/`sm_120a` architecture work. The build trees are configured on demand and must never
be shared across worktrees or SM images. Large model artifacts remain in their authoritative
location and are consumed read-only rather than duplicated into each worktree.

The branch separates optimization history, while the worktree separates the actual checked-out
files and build state. No additional copied per-kernel source hierarchy is needed. The remote WSL
4090, Server 1 RTX PRO 6000, and Server 2 3090 machines use their own local worktree of this same
branch, leaving each machine's existing GInfer checkout untouched.

## Qualification matrix

The project target is real single-GPU inference with one resident model, long contexts from about
120,000 tokens through the 131,072-token boundary, and every startup-fixed concurrency value from
one through eight active requests. Kernel work must account for:

- latency at concurrency one and throughput/scaling at every concurrency from two through eight;
- prefill, long-context decode, and enabled MTP or DFlash paths;
- compact batches with different request positions and valid lengths;
- CUDA Graph, prefix reuse, and relevant BF16/INT8 KV routes; and
- separate runtime qualification on `sm_86`, `sm_89`, and `sm_120a` hardware, including both
  Blackwell SKUs.

An Op microbenchmark is used for kernel decisions, but any whole-round or inference claim is also
remeasured through the corresponding target or public Engine boundary. Compiling all three images
on one machine is required but does not replace runtime measurements on the physical GPUs. The
local RTX 5090 runs `sm_120a`; the SSH-accessible WSL RTX 4090 runs `sm_89`; and the SSH-accessible
Server 2 RTX 3090 runs `sm_86`. SSH-accessible Server 1 runs the RTX PRO 6000 Blackwell
Workstation with the same `sm_120a` image as the 5090, but it owns an independent performance
matrix because its SM count, VRAM, and bandwidth differ. Every host pulls the same committed
optimization branch.

## Campaign controls

Multi-cycle searches use a committed accepted baseline, a short rejection ledger, and one
falsifiable candidate at a time. The current default campaign policy allows at most 24 candidates
per cycle, ends a cycle at the first candidate that clears correctness and the repeated real-round
performance gate, and starts the next cycle from that accepted commit with a fresh profile. A
target campaign stops after one complete 24-candidate cycle produces no accepted candidate. The
budget and terminal count are explicit campaign inputs and can be changed by the user.

See [`campaign-orchestration.md`](agent_workdir/references/campaign-orchestration.md) for the exact
candidate accounting, acceptance gates, file-tree preparation, evidence layout, restoration, and
target handoff rules used by the agent.

## Install locally

The skill has no host-local Python or CUDA dependency; it is pulled separately from GInfer. On a
new machine, clone both published repositories, then create the GInfer worktree without changing
that machine's primary checkout:

```bash
cuda_agent_checkout=/absolute/path/to/cuda-agent
ginfer_repo=/absolute/path/to/ginfer
ginfer_worktree="$ginfer_repo/.worktrees/agentic-kernels"

git clone https://github.com/Gadflyii/CUDA-Agent.git "$cuda_agent_checkout"
git -C "$ginfer_repo" fetch origin perf/agentic-kernels
git -C "$ginfer_repo" worktree add -b perf/agentic-kernels \
  "$ginfer_worktree" origin/perf/agentic-kernels
```

If the branch or worktree already exists, require it to be clean and update it with a
fast-forward-only fetch/merge instead of recreating it:

```bash
git -C "$cuda_agent_checkout" pull --ff-only origin main
git -C "$ginfer_repo" fetch origin perf/agentic-kernels
git -C "$ginfer_worktree" merge --ff-only origin/perf/agentic-kernels
```

Make the pulled skill discoverable by linking the package into the host's Codex skills directory:

```bash
codex_skills_dir="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$codex_skills_dir"
ln -s "$cuda_agent_checkout/agent_workdir" "$codex_skills_dir/ginfer-kernel-optimization"
```

Build products, profiler captures, and model artifacts are deliberately not transported by Git.
Each host configures the native `build-86`, `build-89`, or `build-120a` tree and points the Engine
at an existing local read-only `.ninfer` artifact. The committed runbook records how to verify the
host, artifact, exact branch commit, and native-SM cache before running.

Then invoke it explicitly or let Codex select it for matching work:

```text
Use $ginfer-kernel-optimization to optimize Muse Glimmer DFlash2 round latency at 120K context and concurrency 1-8.
```

```text
Use $ginfer-kernel-optimization to tune the groupwise-int decode path across sm_86, sm_89, and sm_120a.
```

The skill first reads the selected worktree's `AGENTS.md` and relevant repository authorities. It
uses GInfer's CMake/CTest and native benchmark paths, not PyTorch, `torch.compile`, or a fixed global
speedup threshold.
