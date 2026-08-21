# Hardware-Feedback Optimization Loop

Read this reference after the clean baseline and before proposing the first candidate. It adapts
the useful control-loop ideas from CudaForge to production GInfer. It does not import CudaForge's
PyTorch extension harness, loose numerical tolerance, synthetic KernelBench score, or assumption
that one latency scalar represents the workload.

## Keep one compact evidence packet

Carry the current accepted implementation plus a compact evidence packet between iterations. Do
not repeatedly inject the entire conversation, raw profiler dump, or all losing source versions.
The packet should contain:

```text
claim boundary: public Op / target round / public Engine route
workload: exact shapes, dtype/format, resident context, C=1..8, graph/cache/speculative mode
binary: commit, compiler, flags, build type, exact SM image
device: live GPU, compute capability, driver, clocks/thermal/contention caveats
baseline: repeated samples and summary for every scoped matrix point
profile: only metrics needed to distinguish the current hypotheses
diagnosis: one dominant bottleneck and the evidence supporting it
candidate: one coherent change and its predicted matrix effect
disproof: observation that would reject the hypothesis
result: correctness plus paired performance matrix
decision: accept, reject, or gather one named missing measurement
```

Retain a short rejection ledger containing the candidate ID, changed idea, and measured reason for
rejection. This prevents cycling back to known losing ideas without burdening the next iteration
with their full implementations.

## Separate diagnosis from implementation

Treat diagnosis as a read-only step. Its output should be specific enough that an implementer can
act without inventing a different optimization:

```json
{
  "bottleneck": "one dominant measured limitation",
  "evidence": ["two or three decisive observations"],
  "hypothesis": "why one change should improve the scoped workload",
  "candidate_scope": "the narrow code and dispatch surface to change",
  "expected_matrix_effect": "which points should improve or remain neutral",
  "disproof": "the result that rejects this explanation"
}
```

If collaboration is available and the user has authorized parallel agents, an independent judge
can create this packet from source, baseline, and profiler evidence. The judge must not edit code.
Otherwise the same agent performs the two phases sequentially. Either way, correctness tests and
physical measurements decide acceptance; an agent's confidence does not.

## Run one controlled iteration

1. Confirm the current accepted commit and baseline are still comparable.
2. If the bottleneck is already clear from the timeline or algorithm, avoid profiling. Otherwise
   collect a small decision-specific Nsight packet.
3. Produce one diagnosis packet. Do not combine unrelated changes that prevent attribution.
4. Implement the candidate without weakening the public contract.
5. Build and run the focused conformance test before timing it.
6. Run paired measurements using the same binary mode, workload ordering, warmup, repetition,
   cache state, stream, and timing boundary as the baseline.
7. Inspect every scoped point and the relevant aggregate objective.
8. Accept only a correct, repeatable win without an unsupported material regression. Otherwise
   restore the accepted implementation and record the rejection.

A correctness failure changes the next step to repair, not performance diagnosis. After repair,
rerun correctness before profiling or timing. A performance regression changes the next step to
reject or explain with new evidence; do not keep a slower candidate because its implementation is
more sophisticated.

## Bound exploration

Use the iteration or time bound supplied by the user. With no explicit bound, stop a scoped pass
when the measured claim is achieved, the best supported candidates are exhausted, or additional
progress requires a different public boundary or new authorization. More iterations can improve
search quality, but they are not evidence by themselves.

## Sources and adaptation notes

- [CudaForge repository](https://github.com/OptimAI-Lab/CudaForge) and
  [paper](https://arxiv.org/html/2511.01884): separate Coder/Judge roles, correctness before
  optimization, hardware feedback, focused per-round context, and retention of the fastest correct
  candidate. Its authors report that a selected metric subset guided decisions better than an
  exhaustive metric dump.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  iterative assess/optimize/test/deploy practice and realistic workload selection.

CudaForge's reported speedups are evidence about its own KernelBench protocol, not expected GInfer
gains. GInfer uses its native independent oracle and complete product matrix instead.
