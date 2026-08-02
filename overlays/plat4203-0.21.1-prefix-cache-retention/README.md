# PLAT-4203 Path B overlay — DSV4 sm121 (0.21.1) + prefix-cache retention + MTP hitfix

Fixes broken partial prefix-cache reuse on `DeepSeek-V4-Flash` (an append to a long
session re-prefills the entire prompt). Two independent halves:

1. **Retention / block-survival** (`#43447` free-queue eviction ordering) — always
   on. **BENCH-confirmed working**: an exact-repeat is warm (0.4s).
2. **Hit-length under MTP** (`#40624` "Option A" single coordinator-level EAGLE
   drop) — **env-gated `VLLM_DSV4_SINGLE_EAGLE_DROP`, default OFF**. Fixes the
   DSV4 multi-group append-under-MTP 0-hit spiral (`#42948`), where an +N-token
   append still re-prefilled the whole session (9.5s) even with half 1.

Latest image: `gx10-1:30500/vllm-ds4-sm121:0.21.1-retention-hitfix-<sha>`
(retention-only image: `...:0.21.1-retention-<sha>`).

## The MTP hitfix (half 2) — why the append returned 0 hits

DeepSeek-V4-Flash builds **4 attention groups** (1 MLA-full + 3 SWA-MLA with
different `(block_size, sliding_window)`), one of which is the MTP/EAGLE group
(tagged by `_annotate_eagle_groups_deepseek_v4`). `HybridKVCacheCoordinator.
find_longest_cache_hit` runs a fixed-point over the groups. The EAGLE group gets a
per-group "match one block past the boundary then drop it" step inside the loop,
guarded by `eagle_verified` (issue `#32802`, drop-once-per-candidate-length). But
`is_simple_hybrid` only short-circuits the **2-group** case; with DSV4's **4**
groups the loop iterates, and every time a non-eagle SWA group shrinks the
candidate length, `eagle_verified.clear()` **re-fires the EAGLE drop** — cascading
the hit down to 0 for an append (`#42948`, "DSv4 variant of #32802"). An exact
repeat lands cleanly on the cached boundary and escapes the cascade (warm).

**Key finding:** the lookup path here is already **functionally identical** to
vLLM 0.23.0 (verified by diff), and `is_simple_hybrid` is 2-group in 0.23.0 too,
and `#42948` is **unresolved upstream** — so building 0.23.0 (Path A) would NOT
fix this half either. The real fix is fresh coordinator surgery, authored here as
`#40624` **Option A**.

### The fix (`vllm/v1/core/kv_cache_coordinator.py`)
New method `HybridKVCacheCoordinator._find_longest_cache_hit_single_eagle_drop`,
reached from `find_longest_cache_hit` only when `VLLM_DSV4_SINGLE_EAGLE_DROP=1`
**and** there are eagle groups **and** it is not a simple 2-group hybrid. It:
1. Runs the same fixed-point with **no per-group EAGLE drop** (`use_eagle=False`
   everywhere) → the cascade cannot happen; it finds the true longest common
   prefix hit across all groups.
2. Reduces by a **single** EAGLE drop of one `lcm_block_size`:
   `reduced = max(0, common - lcm)` (recompute the final lcm-region so the MTP
   draft head still gets fresh hidden states).
3. **RE-DERIVES** every group at `reduced` (Pass 2) by re-running the same
   convergence — each manager's `find_longest_cache_hit(max_length=reduced)`.

**P0 fix (why Pass 2 re-derives instead of truncating):** a
`SlidingWindowManager.find_longest_cache_hit` result of length H is
`[NULL...NULL, R...R]` — only the last `ceil((window-1)/block_size)` entries are
real KV blocks (covering `[H-window, H)`); the rest are `null_block`. The first
version did `del blks[num_blocks:]` (post-hoc truncation), which **deleted the
real tail blocks and exposed leading NULLs as the new window** -> attention read
null/uninitialized KV -> **silent garbage output** (confirmed live on a TP4 quad:
empty/garbage responses while TTFT looked "warm"). Post-hoc truncation is only
valid for downward-closed full attention. Re-deriving rebuilds each SWA list with
a correctly-covered window at the reduced boundary.

**Correctness:** length can only be **shorter** than the true common prefix
(never longer), and each group's blocks come from the same
`find_longest_cache_hit` primitive the stock path trusts, so the window is always
covered. Default OFF preserves exact current behavior.

### Correctness verification (`verify_hitfix.py`, CPU-only, real vLLM classes)
A unit harness builds a real DSV4-like `HybridKVCacheCoordinator` (1 MLA-full + 3
SWA groups, eagle group tagged), caches a growing session, issues an append, and
certifies (all PASS): append **hits** (non-zero) with the gate on; each group's
returned blocks **equal a fresh `find_longest_cache_hit`** at the final length;
every **SWA group's window tail is real cached KV** (non-null); and the **OLD
`del`-truncate is shown to expose a NULL window tail** on the same state (proving
the harness distinguishes fix from bug).

### To BENCH
Set `VLLM_DSV4_SINGLE_EAGLE_DROP=1` on the scratch DSV4 pod. Run BOTH (a) the
append-TTFT test AND (b) an OUTPUT-CORRECTNESS test (factual question over a large
cached context — verify the answer is correct + non-empty; the earlier corruption
was silent). Leave `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset.

---

## RFC-37003 priority retention (additive, default off)

This overlay also ports the request-scoped KV retention mechanism from the
RFC-37003 candidate implementation. Cortex may send validated
`retention_directives` plus an opaque `retention_scope`; `covers_prompt` is
resolved to the exact prompt token interval in the engine. Protected blocks
remain allocatable soft pins: ordinary LRU is consumed first, then the lowest
priority and oldest protected block.

`VLLM_RETENTION_BUDGET_FRAC` is the hard protected-pool ceiling and defaults to
`0`, which is an exact operational opt-out. A canary must set a positive value
explicitly. TTL expiry demotes blocks to LRU in a batch; requests beyond the
budget also degrade to LRU and increment a drop counter. Metrics are exported
as `vllm:retention_*` gauges/counters through the normal vLLM stats path.

Origin CI runs sidecar tests, then the real request/protocol/BlockPool harness
and the DSV4 hitfix harness inside the exact ARM64 vendor runtime before Kaniko
publishes the immutable content-addressed image. No hand-built image is part of
this path.

## Baseline retention — #43447 free-queue eviction ordering

Root cause of half 1: sliding-window KV prefix blocks are evicted between turns,
so the next turn finds no SWA cache hit and re-prefills the whole prompt.

## What this is

A **pure-Python overlay** on top of the production sm121 image
`aidendle94/sparkrun-vllm-ds4-gb10:production-ready` (vLLM `0.21.1rc1.dev339`,
DeepSeek-V4 sparse-MLA sm121 kernels **compiled and unchanged**). No CUDA rebuild.
It backports the core mechanism of upstream vLLM PR
[#43447](https://github.com/vllm-project/vllm/pull/43447)
("selective prefix-cache retention for sliding-window KV cache", shipped in
vLLM 0.23.0), **adapted to the diverged 0.21.1 sparkrun tree**.

Image: `gx10-1:30500/vllm-ds4-sm121:0.21.1-retention-<patch_sha>`
(= `localhost:30500/...`, same registry).

## Why it's a port, not a clean apply

The sparkrun 0.21.1 tree has already **diverged** from stock and carries its own
bespoke SWA-retention machinery that #43447 does not assume:

- `SlidingWindowMLAManager._protect_prompt_blocks()` pins the prompt-boundary
  blocks with an **extra ref** (via `block_pool.touch`) so they survive after a
  request releases its normal ref (`_protected_prompt_block_ids` / `_queue`,
  `release_protected_prompt_blocks`).
- `SlidingWindowManager._cache_block_mask()` already exists and **disables the
  sparse mask under MTP/EAGLE** (`if getattr(self, "eagle_extra_cache_blocks", 0):
  return None`), i.e. with MTP on it caches every SWA block densely.
- Naming differs from 0.23.0: this tree uses `cache_alignment_tokens` /
  `lcm_block_size` (coordinator), not `scheduler_block_size`; the SWA mask hook
  is the instance method `_cache_block_mask`, not the classmethod
  `reachable_block_mask`.

So #43447 cannot be `git apply`-ed. The **safe, composable subset** ported here is
#43447's **always-on free-queue eviction ordering**, which is additive to the
existing protection mechanism (it only reorders eviction priority; it never frees
a still-referenced/protected block, since `free_blocks` only enqueues blocks whose
`ref_cnt` reaches 0).

## Exactly what changed (4 files, ~130 changed lines; see `diffs/`)

1. **`vllm/v1/core/kv_cache_utils.py`** — add `FreeKVCacheBlockQueue.prepend_n()`
   (verbatim from #43447; splices blocks at the front of the free list).
2. **`vllm/v1/core/block_pool.py`** — `free_blocks(..., prepend=False)`: prepend
   → `prepend_n`, else `append_n` (verbatim from #43447).
3. **`vllm/v1/core/single_type_kv_cache_manager.py`**
   - `SingleTypeKVCacheManager.remove_skipped_blocks`: split the removed
     sliding-window blocks — **uncached scratch blocks `prepend=True` (recycled
     first)**, **cached blocks appended (retained last)**.
   - **New `SlidingWindowManager.free()` override**: same cached-last / scratch-first
     split when a request frees, so cached SWA prefix blocks survive across turns.
     Inherited by `SlidingWindowMLAManager` (the DSV4 class). Composes with
     `_protect_prompt_blocks` (protected blocks keep `ref_cnt > 0` and are skipped).
   - `_cache_block_mask`: **additive, default-off** knob honoring
     `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` to widen the retained-tail segment
     (sparser retention → less KV memory) for the **non-MTP** path only. The
     existing MTP-guard (`eagle_extra_cache_blocks` → `return None`) is preserved.
4. **`vllm/envs.py`** — register `VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None`.

## The env var, and what to set

`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` only gates the sparse-retention **mask**,
which on this build is **bypassed whenever MTP/EAGLE is active**. Production DSV4
runs MTP (`num_speculative_tokens=2`), so **with MTP this var is inert** — leave it
**UNSET**. The actual PLAT-4203 fix is the **always-on free-queue eviction ordering**
(active by default, no env needed). Only set a positive value (a multiple of the
hybrid `lcm_block_size`) if you disable MTP and want to trade prefix-hit density for
KV memory.

## Residual risk (review before BENCH)

- Ported from source-reading only; **not runtime-tested** (no GPU window). Syntax
  verified with `py_compile` on Python 3.12 (matches the image).
- The free-ordering fix addresses **block survival across turns**. If the residual
  re-prefill is instead driven by the coordinator's global-LCM **hit-length**
  computation or the MTP hit truncation (the other half of #43447 / issue #40624),
  this overlay reduces but may not fully eliminate it — that half lives in the
  hybrid coordinator and was intentionally **not** grafted onto the diverged tree.
  The clean path for that is the 0.23.0 build (Path A), where #43447 is native.
- `.pyc` invalidation: COPY updates source mtime, so CPython's default
  timestamp-based invalidation recompiles the modules; no stale bytecode.

## Rebuild

```
docker build --platform linux/arm64 -t gx10-1:30500/vllm-ds4-sm121:0.21.1-retention-<sha> .
docker push gx10-1:30500/vllm-ds4-sm121:0.21.1-retention-<sha>
```
COPY-only (no RUN) ⇒ no qemu needed to build the arm64 image on an amd64 host.
In-cluster kaniko equivalent: mirror `cortex/.forgejo/scripts/run-ci-build-jobs.sh`
(arm64 nodeSelector, `--insecure --skip-tls-verify --insecure-pull`,
dockerhub mirror for the base pull).
