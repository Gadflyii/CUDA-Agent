# Remote SM Validation

Read this reference only when a candidate needs runtime correctness, benchmarking, or profiling on
Server 1 RTX PRO 6000 Blackwell (`sm_120a`), WSL RTX 4090 (`sm_89`), or Server 2 RTX 3090
(`sm_86`). The local RTX 5090 is the other `sm_120a` target.

## Isolation model

Each host has its own local GInfer repository and must check out `perf/agentic-kernels` in a
dedicated Git worktree. A branch alone does not separate the files in an existing checkout.

Use the remote repository's `.worktrees/agentic-kernels` path when its local layout supports that
convention; otherwise discover and record an equally isolated path on that host. Inside it, use
only the native build tree:

| Host | GPU | Image | Native build tree |
|---|---|---|---|
| local | RTX 5090 | `120a` | `build-120a` |
| Server 1 | RTX PRO 6000 Blackwell Workstation | `120a` | `build-120a` |
| WSL 4090 | RTX 4090 | `89` | `build-89` |
| Server 2 | RTX 3090 | `86` | `build-86` |

The 5090 and PRO 6000 execute the same `sm_120a` image, but never merge their measurements. The
current GInfer hardware authority records the Workstation card as 188 SMs with 96 GB VRAM, but
always query and record the live device rather than hard-coding those facts into launch policy.
The PRO 6000 also has a different hardware roofline. Run applicable groupwise-int and NVFP4 paths
there when the changed implementation can reach them.

Do not edit, switch, clean, or build in the remote repository's current primary checkout. Do not
reuse another branch's CMake cache. Large model artifacts already present on a host are inputs;
do not replace, reconvert, or copy them unless the task explicitly requires it.

## Resolve the host once per session

Use the SSH endpoint supplied by the environment or user for the human-readable host label. Do not
guess an address, username, or repository path from the label. Before mutation, query only:

- hostname, user, GPU name/compute capability, driver, and available memory;
- GInfer repository and candidate-worktree paths;
- branch, HEAD, remotes, and concise status;
- CMake/CUDA compiler paths and the native build cache, if present;
- required artifact availability.

If the endpoint cannot be resolved or authenticated, report that setup problem. Do not probe other
accounts or network addresses.

## Publish and synchronize a candidate

Remote validation uses one exact commit so results from all four machines are comparable.

1. Complete local focused correctness and create a coherent commit on
   `perf/agentic-kernels` when the candidate is ready for cross-host testing.
2. Push that branch to its configured origin only as part of the authorized cross-host validation
   task. Record the full candidate commit ID.
3. On each remote host, fetch the named branch in the repository that the user placed in scope.
4. If the dedicated worktree does not exist, add it from the fetched branch without changing the
   primary checkout. If it exists, require a clean status and advance it with a fast-forward-only
   update.
5. Verify that remote `HEAD` exactly equals the recorded candidate before configuring or running.

Never use `reset --hard`, force checkout, force push, automatic stash, merge, or rebase to make a
remote tree match. Stop on dirty state, divergence, unexpected branch ownership, or a mismatched
remote. Do not publish unrelated local commits.

## Build and run

Configure a Release build for only the host's native SM with the tests and benchmarks needed by
the task. Resolve the host's installed CMake and CUDA toolchain instead of assuming the local
paths. Build the smallest relevant production, test, and benchmark targets first; broaden only
after the candidate passes.

Run the same semantic cases and exact performance matrix used locally, including context resident
position/capacity, every scoped concurrency value, KV format, graph/prefix/speculative mode,
warmup, repetition count, corpus or token inputs, and artifact identity. Never compare different
prompts, resident context lengths, cache conditions, or Engine options as if only the GPU changed.

Passwordless `sudo` is available on the remote validation hosts where configured. It authorizes
commands genuinely required to build, test, benchmark, or profile GInfer on those machines.
Ordinary Git, CMake, test, and benchmark commands should run as the repository owner. Do not use
`sudo` to change drivers,
toolkits, packages, services, permissions, clocks, power limits, or other system configuration
unless the user separately requests that change.

## Capture results

Store disposable output under the matching host/SKU directory:

- local 5090: `profiles/sm120a/rtx5090/`;
- Server 1 PRO 6000: `profiles/sm120a/rtx-pro-6000/`;
- WSL 4090: `profiles/sm89/rtx4090/`;
- Server 2 3090: `profiles/sm86/rtx3090/`.

For every result retain:

- exact commit ID, remote hostname, GPU SKU, compute capability, driver, and CUDA compiler;
- build type and exact CUDA architecture;
- model/artifact identity and full workload parameters;
- correctness command and outcome;
- benchmark samples or repository summary, with units;
- profiler command and report path only when profiling answered a live question;
- thermal, contention, unavailable-artifact, or other material limitations.

Pull back only the small reports needed for analysis. Leave build products and large profiler
captures on their owning host unless the task requires transfer. An all-SM claim requires runtime
evidence for `sm_86`, `sm_89`, and `sm_120a`; campaign completion additionally requires separate
5090 and PRO 6000 matrices at the same candidate commit. A failed or unavailable host remains
explicitly pending.
