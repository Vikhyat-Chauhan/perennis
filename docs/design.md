# Agent-Native Inference Engine

*A multi-year project targeting an open problem in LLM inference: session-aware serving for agentic workloads.*

**Status:** design draft
**Owner:** Vikhyat Chauhan
**Working name:** TBD (candidates: SessionServe, Continuum, Perennis)

---

## 1. Vision

Existing production inference engines (vLLM, SGLang, TensorRT-LLM, LMDeploy) were designed for a chatbot-shaped workload: one prompt in, one completion out, requests are largely independent, and the "session" (if it exists) is synthesized by replaying history on every turn. This assumption is quietly breaking as workloads shift from chat to agents.

This project is a ground-up inference engine that treats **the agent session as the primary object of scheduling and caching**, not the request. It is a personal project designed to (a) rebuild first-principles C++ fluency to interview level, (b) provide deep hands-on experience with the full modern LLM inference stack, and (c) make a defensible contribution to an open, active problem in ML systems.

The project is scoped in phases so that each stopping point produces an independently valuable artifact: a small tensor library, a working single-request CPU engine, a minimal serving stack, a session-native prototype, and eventually a distributed system.

---

## 2. Problem Statement

Recent research (KAIROS 2026, Akashic 2026, MemForest 2026, PLENA 2025, "Combating the Memory Walls" 2025) documents that agentic workloads differ from chatbot workloads in ways that current engines handle poorly:

1. **Stateful, long-lived sessions.** An agent may live across dozens of tool calls and hundreds of thousands of tokens, with per-session state that must persist beyond a single request. Chatbot engines discard KV state at request boundary or hash-cache it opportunistically.

2. **Long-tailed, unpredictable growth.** Turn counts, lifetimes, and context-growth rates across concurrent agents follow long-tail distributions. Memory pressure becomes hard to predict, causing thrashing between GPU and slower tiers.

3. **Cross-session structural similarity.** Hundreds or thousands of concurrent agents often share system prompts, tool definitions, and few-shot exemplars. Prefix caching helps but is a shallow solution when the "session" is the natural unit.

4. **Heterogeneous per-turn cost.** A single agent trajectory mixes cheap routing calls, mid-cost retrievals, and expensive reasoning steps. Uniform scheduling policies leave throughput on the table.

5. **Trajectory-shaped access patterns.** Agent tool calls have statistical structure (e.g. `search → summarize → answer`), which is exploitable for prefetching but ignored by current systems.

vLLM's own Q2 2026 roadmap calls out "KV cache manager rethink for complex KV cache layout," CPU/disk offloading, and unresolved scheduler issues (excessive preemption, prefill head-of-line blocking, PP/DP balancing) as active work. Multiple recent papers (Mooncake, DistServe, FastSwitch, LayerKV, LoongServe, BanaServe, KAIROS) are attacking pieces of this from different angles. The problem is real, current, and open.

---

## 3. Design Thesis

**The session, not the request, is the primary abstraction.**

Concretely, this means:

- A session has a persistent identity that outlives any single request or process.
- KV cache is bound to sessions, not requests, and can migrate across memory tiers (GPU → CPU → NVMe → object store) transparently.
- The scheduler reasons about *sessions competing for GPU residency*, not requests competing for a slot.
- Prefix sharing across sessions is a structural feature, not an opportunistic hash lookup.
- Session-level metadata (tool-call trajectory patterns, historical growth rates, priority) informs eviction, prefetch, and placement decisions.

This is a bet that the right abstraction unlocks a set of optimizations that are awkward or impossible when sessions are second-class.

---

## 4. Non-Goals

To keep scope honest:

- **Not a training framework.** Inference only.
- **Not a distributed training system.** Multi-node serving only.
- **Not aiming to beat vLLM at chatbot throughput.** The bet is that agentic workloads are a distinct regime; if we match vLLM there, that is fine.
- **Not building custom hardware or a compiler.** Uses existing GPUs; kernels either hand-written CUDA or via cuBLAS/CUTLASS.
- **Not initially targeting production reliability.** Correctness, then performance, then hardening, in that order.

---

## 5. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  HTTP / gRPC API                                            │
│  POST /session/{id}/turn                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Session Manager    │  ← the novel core
              │  (identity, state,  │
              │   trajectory model) │
              └──────────┬──────────┘
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
│  Scheduler  │   │  KV Cache   │   │  Prefix     │
│  (session-  │◄──┤  Tier Mgr   │   │  Dedup      │
│   aware)    │   │  GPU/CPU/   │   │  (radix     │
│             │   │  NVMe/S3    │   │   tree)     │
└──────┬──────┘   └─────────────┘   └─────────────┘
       │
┌──────▼─────────────────────────────────────────┐
│  Inference Executor                            │
│  Tensor library • KV manager • Sampling        │
│  CPU backend  •  CUDA backend                  │
└────────────────────────────────────────────────┘
```

Each box is roughly one phase of work.

---

## 6. Phased Roadmap

Each phase produces a shippable artifact and can be a stopping point. Estimated durations assume ~10 hrs/week of focused effort alongside a full-time role.

### Phase 1 — C++ Foundations via Tensor Library (Months 1–3)

**Goal:** rebuild production C++ fluency to interview level, using a real target (a tensor library) rather than toy exercises.

**Scope:**
- Custom memory arena / allocator with alignment and RAII lifetime
- Templated `Tensor<T>` supporting `float`, `half`, `int8`
- Rule of 5 implemented explicitly (copy/move ctors, assignment, dtor)
- Operator overloading (`+`, `*`, `[]`, `<<`)
- Custom iterator + `std::algorithm` interop
- Polymorphic op dispatch (virtual `Op` base)
- Exception hierarchy with strong exception safety on ops
- Const-correctness pass
- CMake, GoogleTest, ASan, UBSan, clang-tidy in CI

**Success criteria:**
- Library builds clean on gcc + clang with `-Wall -Wextra -Wpedantic`
- Full test coverage, zero sanitizer warnings
- README with an API tour

**Interview value:** covers nearly every C++ fundamental in one artifact you can walk through. This alone materially strengthens the C++ side of your interview surface.

---

### Phase 2 — Correct Single-Request Inference (Months 4–6)

**Goal:** end-to-end forward pass of a real (small) model on CPU, generating identical tokens to a reference implementation.

**Scope:**
- BPE / SentencePiece tokenizer
- Model loader (start with GGUF format; leverage llama.cpp's format spec)
- Forward pass: matmul, RMSNorm, RoPE, scaled dot-product attention, SwiGLU MLP
- Sampling: greedy, top-k, top-p, temperature
- Reference model: Qwen 2.5 0.5B or Llama 3.2 1B
- Token-level equivalence test vs llama.cpp

**Success criteria:**
- Given a fixed seed and prompt, output matches llama.cpp for at least 500 tokens across 20 diverse prompts
- Perplexity on a small held-out set matches within 1% of reference

**Interview value:** transformer internals stop being conceptual. You can whiteboard attention, explain RoPE from memory, and speak concretely about numerical precision issues.

---

### Phase 3 — Serving Infrastructure (Months 7–9)

**Goal:** turn the engine into a small serving system with the standard modern-serving stack.

**Scope:**
- HTTP endpoint (start with cpp-httplib or Crow)
- Thread pool + thread-safe request queue with condition variables
- Paged KV cache (fixed-size blocks, block table per sequence)
- Continuous batching (dynamic addition of sequences to running batch)
- Block-level prefix caching (hash-indexed)
- Prometheus-style `/metrics` endpoint (TTFT, TPOT, throughput, cache hit rate)
- Benchmark harness modeled on `benchmark_serving.py`

**Success criteria:**
- Serves N concurrent requests without correctness regressions
- Prefix cache hit rate ≥ 90% on prompts with shared 500-token prefix
- Continuous batching demonstrably improves throughput vs static batching in benchmark

**Interview value:** you have built vLLM's core mechanics. You can whiteboard PagedAttention and continuous batching from memory and defend the design choices.

---

### Phase 4 — Session-Native Architecture (Months 10–18)

**Goal:** the novel contribution. Rebuild the serving layer around session as first-class.

**Scope:**
- `Session` abstraction with durable identity (UUID) and metadata store (RocksDB or LMDB)
- Hierarchical KV cache tiers:
  - Tier 0: GPU HBM (active batch)
  - Tier 1: CPU DRAM (warm sessions)
  - Tier 2: local NVMe (cold sessions)
  - Tier 3: object storage (archived / cross-node)
- Async migration between tiers with correctness invariants (a session cannot be observed in an inconsistent state during migration)
- Cross-session prefix deduplication (shared system prompts, tool defs) via radix-tree structure
- Trajectory model: statistical model of tool-call sequences per agent class, used for speculative prefetch
- Session-aware scheduler that prevents head-of-line blocking by long-running agent sessions
- Fairness policy: bound worst-case latency for short interactive requests when co-resident with long agents

**Success criteria:**
- Reproducible benchmark against vLLM on an agentic trace (WebArena, SWE-bench, or synthetic tool-call traces)
- Measurable win on either TTFT or cost-per-agent-completion on trajectories >32k tokens
- Documented failure modes and honest analysis of where vLLM is still ahead

**Interview value:** this is where the project becomes a *conversation starter*. It's an original systems contribution defended by benchmarks, not just a re-implementation. A workshop-paper writeup becomes realistic at this stage.

---

### Phase 5 — Disaggregation and Multi-Node (Months 18–24)

**Goal:** distributed serving with session-aware routing.

**Scope:**
- Prefill and decode as separately schedulable workers (Splitwise / DistServe style)
- KV cache transfer protocol (start with gRPC over TCP; NCCL or RDMA if hardware available)
- Session-affinity router: route requests to nodes with the session's cache warm; migrate on eviction
- Heterogeneous GPU support: assign prefill to compute-heavy nodes, decode to memory-heavy nodes (Cronus direction)

**Success criteria:**
- Two-node deployment with correct KV transfer
- Session affinity demonstrably reduces cold-start cost vs random routing

**Interview value:** you can now credibly interview for distributed inference roles at any of the frontier labs or serving-focused startups.

---

### Phase 6 — CUDA Path (Year 2–3, gated on GPU access)

**Goal:** become someone who can be hired specifically for kernel-level work.

**Scope:**
- Hand-written CUDA attention kernel with IO-aware tiling (FlashAttention pattern)
- INT8 / FP8 quantized matmul path
- CUTLASS integration for GEMM
- Nsight Compute profiling; occupancy and memory-bound vs compute-bound reasoning
- Comparison of naive vs optimized kernels with clear ablations

**Prerequisites:** real GPU access (H100 ideally, A100 or 4090 workable). Without this, phase becomes "study CUDA and CUTLASS deeply" but not "ship kernels."

**Interview value:** kernel-level fluency is one of the highest-paid skills in ML systems right now. This phase directly qualifies you for TensorRT-LLM, vLLM kernel team, DeepSpeed, and inference-startup roles that don't exist without it.

---

### Phase 7 — Research and Positioning (Year 3+)

**Goal:** convert the artifact into professional credit.

**Options (not all required):**
- Write up the session-native architecture as a paper. Realistic targets: MLSys workshop, EuroSys workshop, SOSP short paper, or ArXiv preprint that gets cited.
- Publish head-to-head benchmarks vs vLLM/SGLang on agentic traces as a blog post series.
- Upstream a subset of the work to vLLM or SGLang (their public roadmaps identify aligned work; contributing is often the fastest way to get hired).
- Give talks (Ray Summit, PyTorch Conference, local meetups).

---

## 7. Working Cadence

- **Weekly:** ~10 hours minimum. Commits every session, no exceptions. Draft PRs against your own repo to force review-style discipline.
- **Monthly:** written status update in the repo (`docs/status/YYYY-MM.md`). One paragraph on what shipped, what's blocked, and what changed in the plan. This becomes your own record and, later, blog content.
- **Quarterly:** reread this design doc and update it. The frontier moves; the plan should move with it.
- **Every phase end:** tag a release, write a public post, update the resume line. Even if you never finish the project, each phase is a defensible artifact.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Multi-year solo projects rarely finish as scoped | Each phase is independently valuable; degrade gracefully |
| Job search interrupts progress | The project is a means, not an end. Take offers that put you inside real inference stacks even if you shelve this |
| Frontier moves faster than the plan | Reread and re-scope quarterly. If vLLM ships session-native serving, pivot the novel contribution |
| No GPU access limits Phase 6 | Phases 1–5 have real value CPU-only. Phase 6 becomes "study" not "ship" |
| Complexity trap: over-engineering Phase 1 tensor lib | Time-box each phase; ship rough, iterate |
| Motivation cliff mid-Phase-4 | Write the paper-style abstract up front; work backwards from it as a north star |

---

## 9. Prior Art and References

Systems to study deeply before Phase 3:
- **vLLM** (Kwon et al. 2023, PagedAttention) — the reference implementation
- **SGLang** (Zheng et al. 2024, RadixAttention) — the prefix-caching state of the art
- **DistServe** (Zhong et al. 2024) — prefill/decode disaggregation
- **Splitwise** (Patel et al. 2024) — phase separation on heterogeneous hardware
- **Mooncake** (Qin et al. 2024) — KVCache-centric disaggregated architecture, closest philosophically to this project

Agentic-workload-specific:
- **KAIROS** (2026) — power-aware agentic serving, documents the thrashing regime
- **Akashic / MemAttention** (2026) — long-context agentic serving
- **PLENA** (2025) — memory-wall analysis for agentic inference
- **MemForest** (2026) — hierarchical agent memory
- **BanaServe** (2025) — unified KV cache with dynamic module migration

Read the vLLM and SGLang GitHub issue trackers monthly. They telegraph what's about to become important.

---

## 10. Immediate Next Steps

1. Pick a name and register the repo (public from day one).
2. Scaffold Phase 1: CMake layout, tensor class skeleton, memory arena stub, GoogleTest wiring.
3. Write the first phase-end abstract: one paragraph describing what "Phase 1 complete" looks like as a blog post you'd link on your resume.
4. Block a recurring calendar hold for the weekly work session. This is the single strongest predictor of whether the project happens.
