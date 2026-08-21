# Handwritten PTX and SASS Inspection

Read this reference only after profiling identifies a tiny instruction-level hot path or a required
architecture feature is not adequately exposed by the current C++ implementation.

## Gate PTX behind evidence

NVIDIA describes direct PTX as an advanced, last-resort technique. Use it only when all of these
are true:

1. The kernel and instruction sequence materially affect the requested public boundary.
2. Algorithm, fusion, traffic, launch, and layout opportunities have been considered first.
3. PTX/SASS inspection shows a specific compiler-generated sequence or missing operation to
   improve.
4. The desired instruction is supported by the exact toolkit and architecture image.
5. A high-level implementation remains as the semantic reference and fallback where required.
6. The expected gain is large enough to justify additional correctness and portability burden.

Prefer, in order, a normal CUDA C++ expression or intrinsic, an appropriate libcu++/CCCL
`cuda::ptx` wrapper, and finally inline `asm`. The `cuda::ptx` namespace provides one-to-one PTX
instruction wrappers and often avoids manual operand-constraint errors.

## Inspect what the compiler actually emitted

PTX is a virtual ISA; the shipped cubin contains architecture-specific SASS. Compare both when it
changes the decision:

```bash
nvcc ... -Xptxas=-v
cuobjdump --dump-ptx <binary-or-library>
cuobjdump --dump-sass <binary-or-library>
nvdisasm <cubin>
```

Record registers, spills/local memory, instruction count/mix around the hot primitive, control
flow, and any new synchronization or conversion. A shorter PTX snippet can still produce worse
SASS or raise register pressure.

## Inline-assembly correctness rules

- Match operand constraints to the C++ and PTX register widths. Use a read/write output constraint
  when the old value is also consumed.
- Use `asm volatile` when the statement has side effects that must not be deleted or moved. Add a
  `"memory"` clobber for hidden user-memory access or when compiler memory motion across the
  statement would violate semantics.
- Scope temporary PTX registers and labels so repeated inlining cannot collide.
- Treat pointer address spaces deliberately; inline assembly receives generic pointer values and
  the author is responsible for using valid state-space operations.
- The CUDA front end does not parse the assembly string. Constraint/type mistakes can surface only
  at `ptxas`, so compile every supported exact image.
- PTX `volatile`, CUDA C++ `volatile`, atomics, barriers, and memory fences have different
  semantics. Do not use volatility as inter-thread synchronization.
- Preserve active masks, convergence assumptions, memory order/scope, async-proxy fences, NaN and
  signed-zero behavior, tie-breaking, saturation, rounding, and conversion boundaries.

## Architecture containment

Guard architecture-specific instructions at compile time and keep a deliberate fallback or route
rejection. An `a` target such as `sm_120a` enables architecture-specific features that must not be
treated as a portable implementation for `sm_89` or `sm_86`. Verify both the PTX ISA requirement
and the target-specific instruction restriction in the current toolkit documentation.

After adding PTX:

1. build all exact SM images;
2. run focused edge-case and public conformance tests on every reachable path;
3. compare resources and SASS against the high-level implementation;
4. benchmark the entire scoped matrix, including fallback routes and dispatch seams;
5. retain it only for a repeatable public-boundary win.

## What the NVIDIA top-k example proves

NVIDIA's handwritten-PTX article demonstrates a narrow Hopper CUTLASS fusion where specialized
top-2/top-4 and masked-softmax PTX beat the generic C++ fallback for the tested shapes. Its useful
pattern is conditional specialization plus a C++ fallback and direct A/B measurement. Its reported
percentage is not a transferable expectation for GInfer, another SM, or another selection route.

## Sources

- [CUDA Programming Guide: Using PTX](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#using-ptx)
- [NVIDIA handwritten PTX optimization article](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/)
- [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [`cuda::ptx` API](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
