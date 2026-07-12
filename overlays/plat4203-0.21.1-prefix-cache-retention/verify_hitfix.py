"""PLAT-4203 hitfix correctness verification (CPU-only, real vLLM classes).

Proves the P0 fix: `_find_longest_cache_hit_single_eagle_drop` must NOT truncate
SWA group block lists in place (which exposed leading NULLs as the window ->
silent garbage). Instead it re-derives every group at the reduced length.

Invariant asserted: for the env-gated Option-A path, each group's returned block
list EQUALS a fresh `find_longest_cache_hit(max_length=final_hit_len)` for that
group. The OLD buggy `del blks[num_blocks:]` would NOT satisfy this for SWA
groups (it would leave a null-tailed window). We also directly assert the SWA
window tail is real (non-null) at the final length.
"""
import os
os.environ["VLLM_DSV4_SINGLE_EAGLE_DROP"] = "1"

import vllm.envs as E
assert E.VLLM_DSV4_SINGLE_EAGLE_DROP is True, "env not honored"

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, SlidingWindowSpec, KVCacheGroupSpec, KVCacheConfig, KVCacheTensor,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher, init_none_hash, BlockHashListWithBlockSize,
)
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams

BS = 16          # one block size for all groups -> lcm = 16, eagle drop = 1 block
HBS = 16         # hash block size
NUM_BLOCKS = 4000
MAX_LEN = 8192

# hashing fn (consistent between requests so shared prefixes match)
try:
    from vllm.utils.hashing import sha256_cbor as HASHFN
except Exception:
    from vllm.v1.core.kv_cache_utils import xxhash_cbor as HASHFN
init_none_hash(HASHFN)
block_hasher = get_request_block_hasher(HBS, HASHFN)

def spec_full():
    return FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float16)

def spec_swa(window):
    return SlidingWindowSpec(block_size=BS, num_kv_heads=1, head_size=1,
                             dtype=torch.float16, sliding_window=window)

# DSV4-like: 1 full (MLA) + 3 SWA groups w/ different windows; last is the MTP/eagle group.
groups = [
    KVCacheGroupSpec(["full"],  spec_full()),
    KVCacheGroupSpec(["swa_a"], spec_swa(window=6 * BS)),
    KVCacheGroupSpec(["swa_b"], spec_swa(window=4 * BS)),
    KVCacheGroupSpec(["swa_mtp"], spec_swa(window=3 * BS), is_eagle_group=True),
]
cfg = KVCacheConfig(num_blocks=NUM_BLOCKS,
                    kv_cache_tensors=[KVCacheTensor(size=NUM_BLOCKS * 16, shared_by=[g.layer_names[0]]) for g in groups],
                    kv_cache_groups=groups)

mgr = KVCacheManager(kv_cache_config=cfg, max_model_len=MAX_LEN, hash_block_size=HBS,
                     enable_caching=True, use_eagle=True)
coord = mgr.coordinator
print("coordinator:", type(coord).__name__,
      "| attn_groups:", len(coord.attention_groups),
      "| lcm:", coord.lcm_block_size,
      "| eagle_group_ids:", coord.eagle_group_ids)
assert type(coord).__name__ == "HybridKVCacheCoordinator"
assert coord.eagle_group_ids, "expected an eagle group tagged"

def make_req(rid, token_ids):
    return Request(request_id=rid, prompt_token_ids=list(token_ids),
                   sampling_params=SamplingParams(), pooling_params=None,
                   block_hasher=block_hasher)

# Build a long "session" prompt and cache it chunk-by-chunk (so SWA windows are
# cached at MANY boundaries, like a growing session — makes an append hittable).
N_TOK = 220 * BS
base = [i % 50000 for i in range(N_TOK)]
req = make_req("sess", base)
computed, n = mgr.get_computed_blocks(req)
CHUNK = 8 * BS
done = 0
while done < N_TOK:
    step = min(CHUNK, N_TOK - done)
    blks = mgr.allocate_slots(req, step, n if done == 0 else 0, computed if done == 0 else None,
                              num_lookahead_tokens=2)
    assert blks is not None, f"alloc failed at {done}"
    done += step
    req.num_computed_tokens = done
    mgr.cache_blocks(req, done)
mgr.free(req)
print(f"cached session of {N_TOK} tokens in {N_TOK//BS} blocks")

# APPEND: same prefix + a few new tokens (not a full new block).
append = base + [777, 778, 779, 780, 781]
areq = make_req("append", append)

# --- the gated Option-A hit ---
computed_blocks, hit_len = coord.find_longest_cache_hit(areq.block_hashes, areq.num_tokens - 1)
print("Option-A hit_length =", hit_len, "(tokens); ", hit_len // BS, "blocks")
assert hit_len > 0, "FAIL: append returned 0 hit (cascade not fixed in harness)"

def group_block_hashes(spec):
    if spec.block_size == coord.hash_block_size:
        return areq.block_hashes
    return BlockHashListWithBlockSize(areq.block_hashes, coord.hash_block_size, spec.block_size)

null_block = coord.block_pool.null_block

# INVARIANT 1: each group's returned list EQUALS a fresh find_longest_cache_hit at hit_len.
# (The OLD del-truncate would violate this for SWA groups.)
for spec, group_ids, manager_cls in coord.attention_groups:
    fresh = manager_cls.find_longest_cache_hit(
        block_hashes=group_block_hashes(spec), max_length=hit_len,
        kv_cache_group_ids=group_ids, block_pool=coord.block_pool,
        kv_cache_spec=spec, use_eagle=False, alignment_tokens=coord.lcm_block_size)
    for k, gid in enumerate(group_ids):
        got = list(computed_blocks[gid])
        exp = list(fresh[k])
        got_ids = [b.block_id for b in got]
        exp_ids = [b.block_id for b in exp]
        assert got_ids == exp_ids, (
            f"FAIL invariant-1 group {gid} ({type(spec).__name__}): "
            f"returned list != fresh re-derive.\n got={got_ids}\n exp={exp_ids}")
    print(f"  group {group_ids} {type(spec).__name__}: re-derive MATCHES fresh lookup (len={len(fresh[0])})")

# INVARIANT 2: for each SWA group, the window tail at hit_len is REAL (non-null).
# This is exactly what the P0 bug corrupted (exposed leading NULLs as the window).
from math import ceil
for spec, group_ids, manager_cls in coord.attention_groups:
    if not isinstance(spec, SlidingWindowSpec):
        continue
    swcb = ceil((spec.sliding_window - 1) / spec.block_size)
    for gid in group_ids:
        blks = list(computed_blocks[gid])
        if not blks:
            continue
        tail = blks[-swcb:]
        real = [b for b in tail if b is not null_block and getattr(b, "block_hash", None) is not None]
        assert len(real) == len(tail), (
            f"FAIL invariant-2 SWA group {gid}: window tail has NULL/uninit blocks "
            f"(the P0 corruption). tail_ids={[b.block_id for b in tail]}")
    print(f"  SWA group {group_ids} window tail ({swcb} blocks) all REAL cached KV ✓")

# INVARIANT 3: demonstrate the OLD buggy del-truncate WOULD have broken invariant-2,
# proving this test actually distinguishes the fix from the bug.
# Reconstruct the pre-drop common hit, then apply the OLD del-truncate, and show a
# SWA window tail becomes null.
def converge_no_eagle(max_len):
    hbg = [None] * len(cfg.kv_cache_groups)
    hl = max_len
    while True:
        curr = hl
        for spec, group_ids, manager_cls in coord.attention_groups:
            if isinstance(spec, FullAttentionSpec) and hbg[group_ids[0]] is not None:
                curr = curr // spec.block_size * spec.block_size; continue
            hb = manager_cls.find_longest_cache_hit(
                block_hashes=group_block_hashes(spec), max_length=curr,
                kv_cache_group_ids=group_ids, block_pool=coord.block_pool,
                kv_cache_spec=spec, use_eagle=False, alignment_tokens=coord.lcm_block_size)
            curr = len(hb[0]) * spec.block_size
            for gid, bl in zip(group_ids, hb): hbg[gid] = list(bl)
        if curr >= hl: break
        hl = curr
    return hbg, hl

hbg, common = converge_no_eagle(areq.num_tokens - 1)
reduced = max(0, common - coord.lcm_block_size)
old_bug_exposed_null = False
for spec, group_ids, manager_cls in coord.attention_groups:
    if not isinstance(spec, SlidingWindowSpec):
        continue
    swcb = ceil((spec.sliding_window - 1) / spec.block_size)
    for gid in group_ids:
        blks = list(hbg[gid])
        nb = reduced // spec.block_size
        old = blks[:nb]  # OLD: del blks[nb:]
        if old:
            tail = old[-swcb:]
            if any(b is null_block for b in tail):
                old_bug_exposed_null = True
print(f"OLD del-truncate would expose NULL window tail? {old_bug_exposed_null} "
      f"(common={common}, reduced={reduced})")
assert old_bug_exposed_null, (
    "FAIL invariant-3 negative control: the reconstructed old del-truncate "
    "path did not expose a NULL SWA window tail; this harness no longer "
    "discriminates the P0 regression"
)

print("\nALL INVARIANTS PASS — Option-A re-derive returns window-correct blocks; "
      "P0 (SWA null-window corruption) is fixed.")
