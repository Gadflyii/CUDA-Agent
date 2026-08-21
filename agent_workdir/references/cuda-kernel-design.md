# CUDA Kernel Design for Real Inference

Read this reference before changing device code. Use it to generate hypotheses from the GInfer
contract and profile; it is not a checklist that every kernel must implement.

## Start with work and traffic

Account for the minimum semantic work, bytes read, bytes written, reductions, conversions, and
launches. Compare that lower bound with measured traffic and time. The largest avoidable pass or
launch often offers more value than instruction-level tuning.

Prefer, when the contract permits:

- eliminating redundant computation or memory passes;
- fusing producer/consumer stages while preserving reusable Op ownership;
- retaining intermediates in registers or shared memory only when reuse repays the resource cost;
- moving compact metadata once and keeping inference state resident on the GPU;
- specializing a finite set of important shapes/formats at compile time while retaining a correct
  fallback for other admitted inputs.

Do not fuse across a public semantic or state-transaction boundary merely to remove a launch.

## Map work to warps deliberately

A warp has 32 lanes. Long data-dependent paths taken by only some lanes serialize the paths for
that warp. Assign lanes so adjacent lanes perform adjacent, similarly shaped work where possible.
For reductions and exchanges, use explicit active masks and synchronization; independent thread
scheduling means implicit warp lockstep is not a correctness mechanism.

For tails, distinguish a short predicated load/store from a divergent long algorithm. Finite
compile-time specializations can remove hot-path branches, but every dispatch seam and tail must
remain tested.

## Treat memory movement as a transaction problem

On the supported architectures, a warp's global accesses are serviced in the required 32-byte
segments. Useful/requested bytes can be far below actual bytes when lanes are strided, scattered,
misaligned, or inactive. Use vector movement only when pointer alignment, element count, aliasing,
and tails prove it legal; a vector type is not automatically a faster transaction.

Use shared memory when it creates measured reuse, converts an uncoalesced global pattern into a
coalesced one, or supports an efficient cross-thread exchange. Shared memory has 32 banks for
32-bit words on these devices. Inspect bank conflicts for transposes and strided tiles; padding can
help, but added shared memory can lower residency.

Hardware-accelerated global-to-shared asynchronous copies are available from Ampere onward and can
overlap movement with computation while avoiding an intermediate register. They help only when
there is enough independent computation, stages fit resource limits, alignment is valid, and the
pipeline/barrier protocol is correct. A synchronous load can remain best for tiny or poorly reused
tiles.

For long-context KV and attention paths, model traffic at the actual resident position. Focus on
eliminating rereads, combining dequantization with consumption, preserving coalesced head/token
layout, and balancing the online reduction. A large configured capacity with a short resident
sequence is not representative evidence.

## Balance resources; do not maximize occupancy blindly

Registers, shared memory, threads, and architectural block limits jointly determine residency.
Occupancy exists to hide latency; it is not a performance score. Higher occupancy can lose when it
forces spills, reduces instruction-level parallelism, or adds synchronization.

For every material tile/block change, record:

- registers per thread and spill/local-memory traffic;
- static and dynamic shared memory per block;
- active blocks/warps and the limiting resource on the target SM;
- grid waves at `C=1..8` and relevant token/head extents;
- elapsed time, not occupancy alone.

Use `-Xptxas=-v`, Nsight Compute's occupancy data, and the CUDA occupancy APIs as diagnostic tools.
Apply `__launch_bounds__` or register caps only after a measured resource hypothesis; arbitrary
caps commonly trade lower register count for slower local-memory spills.

## Match the pattern to inference geometry

### Reductions, selection, top-k, and softmax

Keep lane ownership, tie-breaking, NaN/Inf behavior, sampling probabilities, and deterministic
paths identical to the public contract. Compare warp shuffles, shared reductions, multi-stage
reductions, and small fixed-size register networks by total work and register footprint. A method
that helps greedy selection can still regress stochastic sampling, so measure every route.

### GEMV, small-M GEMM, and fused projections

At compact batch, enough CTAs and useful work per CTA may matter more than a large square-GEMM
tile. Compare Tensor Core setup and padding against SIMT paths at the exact M/N/K and quantized
formats. Fuse dequantization, bias, activation, residual, or paired projections only when the
complete formula and output precision remain testable.

### Attention and long-context decode

At 120K-131K, KV traffic, dequantization, reduction order, and partition/merge overhead can dominate.
Measure INT8 and BF16 KV paths separately. Partition enough work to use the GPU without making
merge traffic or launch count dominate at `C=1`; re-evaluate all interior concurrency points.

### Tiny state and agentic operations

When kernels are only a few microseconds, inspect the CPU/GPU timeline and CUDA Graph route before
rewriting arithmetic. Fusion, device-resident state, fewer validations on already-admitted paths,
or graph-safe composition may beat a faster standalone kernel. Preserve stream ordering and full
state transitions.

## Use asynchronous features with explicit semantics

Choose the narrowest correct synchronization scope. Split arrive/wait barriers and pipelines can
overlap independent work, but only after identifying the producer, consumer, memory visibility,
and lifetime of every stage. Operations using an asynchronous proxy may require a proxy fence
before ordinary loads/stores observe the data. Never replace a block barrier with a warp barrier
unless every consumer is in the writing warp.

## Use historical guidance as concepts, not constants

The 2009 NVIDIA SC09 optimization deck is still useful for several durable mental models:

- distinguish useful bytes from actual memory transactions;
- arrange adjacent threads around contiguous data rather than optimizing only one thread's local
  sequence;
- hide latency with enough independent work and resident warps;
- use shared memory for reuse, communication, or access reordering;
- treat divergence, block geometry, host/device transfers, and overlap as measured costs.

Its numeric bandwidths, half-warp transaction rules, cache assumptions, bank count, occupancy
targets, block-count limits, and recommended block sizes describe pre-Fermi hardware. Do not copy
those constants or its Visual Profiler workflow into an Ampere, Ada, or Blackwell decision. Use the
current per-architecture tuning guides, Nsight tools, and live device attributes for all concrete
policy.

## Sources

- [CUDA Programming Guide: Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [CUDA Optimization at SC09 (historical)](https://www.nvidia.com/content/GTC/documents/SC09_Optimization_Micikevicius.pdf)
