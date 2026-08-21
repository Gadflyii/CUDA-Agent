# Nsight Triage for GInfer Kernels

Read this reference only when profiling will choose between plausible next changes. Stable elapsed
time at the public boundary remains the performance result; profiler replay measurements are
diagnostic evidence.

## Choose the profiler from the question

Use Nsight Systems first when the unknown is launch overhead, CPU dispatch, synchronization,
memcpy, CUDA Graph capture/replay, stream overlap, or a multi-kernel round. This is especially
important for compact `C=1..8` inference, where a microsecond-scale launch gap can matter more than
device instruction throughput.

Use Nsight Compute only for a named kernel and a representative production launch when the
unknown is memory behavior, instruction throughput, occupancy/resource pressure, divergence,
latency hiding, or Tensor Core utilization. Filter the launch precisely; profiling an entire
inference trace can be extremely slow and can select the wrong dynamic instance.

## Start with a small metric packet

Metric names and availability vary by GPU and Nsight Compute version. Query the installed tool
instead of assuming a fixed list:

```bash
ncu --list-sets
ncu --list-sections
ncu --query-metrics
```

Begin with a high-level Speed-of-Light/launch/occupancy view, then add only the family that tests
the current hypothesis. A useful packet usually includes:

- kernel duration or active cycles and the exact dynamic launch identity;
- grid/block shape, waves per SM, registers per thread, static/dynamic shared memory, achieved
  occupancy, and the limiting launch resource;
- scheduler issue/eligible/active-warp evidence;
- compute-pipeline and memory-pipeline utilization;
- requested versus actual global traffic, DRAM bytes/throughput, L2/L1 behavior, local-memory
  traffic, and shared-bank conflicts when memory is implicated;
- the one or two dominant warp-stall families only when scheduler issue is actually low;
- Tensor Core or relevant arithmetic-pipeline utilization when compute is implicated.

Avoid `--set full` as the first collection. Nsight Compute can require multiple replay passes, and
an exhaustive dump adds overhead and ambiguous signals. Save the report under the correct
`profiles/sm*/<sku>/` directory and extract a concise comparison packet for reasoning.

## Interpret symptoms as combinations

No single percentage proves a bottleneck. Use elapsed time plus corroborating signals.

| Observed combination | Plausible next hypothesis | Check before changing code |
|---|---|---|
| High DRAM utilization, low arithmetic intensity, expected byte count is large | Remove a pass, fuse an epilogue, compress/reuse data, or improve locality | Confirm bytes required by the semantic algorithm and that context/cache state is real |
| Actual memory traffic greatly exceeds useful/requested bytes | Coalesce accesses, align/vectorize valid regions, reorganize layout, or eliminate overfetch | Check tail alignment, lane participation, cache-line reuse, and every supported format |
| High local-memory traffic with high registers/thread | Reduce live ranges, split a phase, retile, or remove unproductive unrolling | Confirm spills in `ptxas`/SASS; do not cap registers blindly and create more spills |
| Low eligible warps, long-scoreboard pressure, memory pipelines below saturation | Increase independent work, prefetch/pipeline, improve locality, or raise useful residency | Separate dependency latency from insufficient grid size and from uncoalesced traffic |
| High barrier pressure and poor issue rate | Narrow synchronization scope, restructure producer/consumer work, or reduce stages | Preserve cross-warp visibility; a warp-only barrier is valid only for warp-owned data |
| Very few waves or blocks for compact batch | Expose parallelism within requests, use a persistent/cooperative schedule, fuse launches, or use graphs | Account for block resource limits, tails, C=1 latency, and C=8 fairness |
| Compute pipeline near its relevant roof with low memory pressure | Reduce instructions, use a suitable intrinsic/Tensor Core path, specialize constants, or fuse arithmetic | Preserve numerical order/tolerance and confirm the compiler did not already emit the instruction |
| High nominal occupancy but unchanged issue/latency | Occupancy is not the limiting objective | Look for dependencies, redundant work, instruction mix, or launch overhead |
| Low occupancy with high ILP and strong throughput | Do not optimize occupancy alone | Test whether more residency improves elapsed time without spills or extra synchronization |
| Branch divergence plus high executed-instruction cost | Reassign work by lane/warp, specialize finite cases, or use predication | Distinguish unavoidable tail masks from data-dependent long paths |

Only focus on stall reasons when schedulers are failing to issue; stalls can be harmless or
unavoidable. Compare requested/effective bandwidth with actual traffic to distinguish true
bandwidth saturation from wasted transactions.

## Measurement discipline

- Profile the same accepted/candidate binaries used in the benchmark; record commit and kernel
  name.
- Use a production-representative launch, not an easier shape that changes the route.
- Be cautious with replay on kernels that mutate state. Verify replay mode preserves valid inputs,
  or profile a controlled benchmark that reconstructs state for each launch.
- Do not compare NCU-reported replay duration directly with ordinary benchmark latency. Compare
  benchmark to benchmark and profile to profile.
- Recheck the public timing boundary after acting on a kernel metric; a faster kernel can leave the
  round unchanged or make another stage dominant.

## Sources

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute user documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA C++ Best Practices: performance metrics and effective bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics)
- [CudaForge paper: selected hardware-feedback metrics](https://arxiv.org/html/2511.01884)
