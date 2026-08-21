# Kernel Campaign Orchestration

Read this reference for a multi-cycle kernel search or a handoff between model targets or physical
GPUs. The single-candidate evidence loop remains defined in
[`hardware-feedback-loop.md`](hardware-feedback-loop.md); this document controls the campaign
around that loop.

## Freeze the campaign contract

Before building, record a compact campaign header:

```text
target and order: Muse Glimmer, then Qwen3.8-27B
host/GPU/image: exact hostname, GPU SKU, compute capability, sm_XX image
accepted start: full commit ID
artifact: exact path, registered identity, size, and hash when practical
workload: mode, draft width, KV format/capacity, graph/prefix mode, context, C values
correctness: independent Op oracle plus named real Engine semantic sentinels
real boundary: target Program round or public Engine measurement
candidate budget: maximum candidates per cycle
acceptance gate: repeatability threshold and allowed pointwise regressions
terminal rule: number of complete no-win cycles before stopping the target
```

The current campaign defaults are:

- at most 24 candidates in one cycle;
- stop the current cycle at the first accepted candidate;
- accept only after the independent oracle passes, affected public-Op points have no material
  unsupported regression, and two real-boundary repetitions each improve the fresh baseline by at
  least 2%;
- recheck unaffected concurrency points and route seams for neutrality;
- run the named short- and long-context Engine semantic sentinels and registered real test before
  final acceptance;
- begin a new cycle from the accepted commit and collect a fresh baseline/profile; and
- stop the target after one complete 24-candidate cycle has no accepted candidate.

These are campaign controls, not timeless GInfer semantics. A user-supplied budget, threshold, or
no-win count replaces the corresponding default and must be written into the campaign header.
Never silently extend a candidate budget or weaken the correctness/performance gate.

## Prepare an isolated tree on every host

The primary GInfer checkout on each machine is read-only. Use the same published branch in a
dedicated worktree and a native-SM build directory:

```text
<host-ginfer>/.worktrees/agentic-kernels/  branch perf/agentic-kernels
├── build-86/                              Server 2 RTX 3090 only
├── build-89/                              WSL RTX 4090 only
├── build-120a/                            RTX 5090 or PRO 6000 only
└── profiles/
    ├── sm86/rtx3090/<target>/<cycle>/
    ├── sm89/rtx4090/<target>/<cycle>/
    └── sm120a/<sku>/<target>/<cycle>/
```

For an existing worktree, require a clean status, fetch the named branch, and advance only with a
fast-forward. For a missing worktree, create it from the fetched branch without switching the
primary checkout. Never use `reset --hard`, force checkout, automatic stash, merge, or rebase to
make a remote tree match.

Before mutation, verify and record:

1. repository/worktree path, branch, full HEAD, remote, and clean status;
2. hostname, GPU name, compute capability, driver, live free memory, and CUDA compiler;
3. Release build type and exactly one `CMAKE_CUDA_ARCHITECTURES` value;
4. required artifact and fixture availability; and
5. exact equality between local, remote, and origin campaign commits.

A branch separates history; a worktree separates checked-out files, build caches, and generated
state. Both are required. Do not copy individual GInfer source files into the skill repository or
another ad-hoc directory. Artifacts remain read-only at their authoritative path.

## Count candidates consistently

A candidate is one falsifiable implementation hypothesis whose source edit has begun. It consumes
one slot when the first candidate source commit is created, even if it later fails to compile,
fails correctness, or is rejected before the real-boundary benchmark. A mechanical repair that
does not change the hypothesis remains part of that candidate; a changed mechanism, dispatch
domain, or resource strategy is a new candidate.

Read-only source inspection, baseline measurement, profiler collection, and benchmark-enabling work
do not consume candidate slots. Enabling work must be production-relevant, independently reviewed,
and retained only if it remains useful without the candidate.

For every candidate, append one durable ledger row containing:

```text
cycle candidate | commit | changed mechanism | predicted effect | oracle result |
public-Op result | real-boundary result | accept/reject reason | restoration commit
```

Commit candidate source before remote execution. Reject through a normal revert commit so the
accepted tree and experimental history remain inspectable. Do not leave a rejected implementation
in the final source merely because one microbenchmark point improved.

## Gate from narrow evidence to real inference

Use the cheapest decisive gate first:

1. compile the focused target;
2. pass the independent numerical or exact oracle at changed shapes and dispatch seams;
3. measure the affected public-Op matrix, including the production-weighted extent;
4. run the target Program round at the scoped concurrency and repeat the decisive point;
5. check latency/throughput at the unaffected concurrency points and route seams; and
6. run the short- and long-context public Engine semantic sentinels plus the registered real test.

Reject immediately when a lower gate disproves the hypothesis. Do not spend a model load or
long-context run on a candidate that already fails its oracle or public-Op matrix.

Generated text must be coherent and satisfy the fixture's independent retrieval/task oracle.
Stochastic token streams, completion lengths, and acceptance ratios may differ after a numerically
valid kernel change; plausible text alone is never the Op oracle. Record prompt and decode
throughput separately and do not compare runs with different prompt tokens, output accounting,
resident context, or admission state as paired performance evidence.

## Accept, restart, or stop

On acceptance:

1. retain the candidate commit and any legitimate benchmark/test enabling commits;
2. push the exact accepted branch and verify every participating worktree is synchronized;
3. close the cycle ledger with the before/after matrix and correctness results;
4. collect a fresh baseline and decision-specific profile from the accepted binary; and
5. start the next numbered cycle with the candidate counter reset to zero.

On rejection, restore the accepted source through a normal revert, verify the oracle as needed, and
continue until the cycle budget is exhausted. When a full cycle has no accepted candidate, verify
that the final source tree matches the cycle-start source, rebuild the accepted binary, close and
commit the ledger summary, and apply the terminal no-win rule.

After a target stops, hand off in the declared order. Preserve the same branch and accepted source,
but freeze a new target-specific artifact/workload baseline and start candidate numbering at cycle
one for that target. The current order for each GPU family is Muse Glimmer first, then Qwen3.8-27B.

## Commit durable evidence, leave bulk data local

Source, retained benchmarks/tests, campaign controls, accepted results, and concise rejection
ledgers are committed. Raw `.nsys-rep`, Nsight Compute reports, SQLite exports, full model logs,
and build products stay under the ignored host-specific `profiles/` or build tree. A committed
result document records where those raw files live and the facts needed to interpret them.

Before handoff, require:

- clean local and remote worktrees at one pushed commit;
- no untracked source snapshots, forced selectors, scratch executables, or losing variants;
- a focused oracle and registered real test passing from the final binary;
- a durable result summary and rejection ledger; and
- explicit pending qualification for every physical GPU not yet measured.
