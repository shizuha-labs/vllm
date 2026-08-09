# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).parent


@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: bytes | None = None
    is_null: bool = False


kv_utils_stub = ModuleType("vllm.v1.core.kv_cache_utils")
kv_utils_stub.KVCacheBlock = KVCacheBlock
sys.modules["vllm.v1.core.kv_cache_utils"] = kv_utils_stub


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


priority_module = _load_module(
    "vllm.v1.core.priority_eviction_queue",
    HERE / "vllm/v1/core/priority_eviction_queue.py",
)
PriorityEvictionQueue = priority_module.PriorityEvictionQueue
RetentionMeta = priority_module.RetentionMeta


def _hashed_block(block_id: int) -> KVCacheBlock:
    block = KVCacheBlock(block_id)
    block.block_hash = b"hash" + block_id.to_bytes(4, "big")
    return block


def _set_meta(queue, block, priority, *, expiry=None, scope=None, freed=0.0):
    queue._meta[block.block_id] = RetentionMeta(
        priority=priority,
        expiry=expiry,
        scope=scope,
        last_freed_time=freed,
    )


def test_priority_and_generation_ordering():
    queue = PriorityEvictionQueue()
    low = _hashed_block(1)
    high = _hashed_block(2)
    _set_meta(queue, low, 20)
    _set_meta(queue, high, 80)
    queue.try_insert(low, 100.0)
    queue.try_insert(high, 200.0)
    assert queue.pop_lowest() is low

    queue.suspend(high)
    _set_meta(queue, high, 90)
    queue.try_insert(high, 300.0)
    middle = _hashed_block(3)
    _set_meta(queue, middle, 70)
    queue.try_insert(middle, 250.0)
    assert queue.pop_lowest() is middle
    assert queue.pop_lowest() is high
    assert queue.metrics(4)["priority_evictions_total"] == 3


def test_heap_tombstones_are_bounded_across_long_reuse_cycle():
    queue = PriorityEvictionQueue()
    block = _hashed_block(1)
    _set_meta(queue, block, 90)
    for freed_at in range(10_000):
        assert queue.try_insert(block, float(freed_at))
        queue.suspend(block)
    assert len(queue._heap) <= queue._COMPACTION_FLOOR
    assert queue.try_insert(block, 10_001.0)
    assert queue.pop_lowest() is block


def test_budget_admission_displaces_only_lower_priority():
    queue = PriorityEvictionQueue()
    incumbent = _hashed_block(1)
    higher = _hashed_block(2)
    equal = _hashed_block(3)
    _set_meta(queue, incumbent, 20)
    _set_meta(queue, higher, 90)
    _set_meta(queue, equal, 90)

    assert queue.admit(incumbent, 1, 100.0) == (True, None)
    admitted, displaced = queue.admit(higher, 1, 200.0)
    assert admitted is True
    assert displaced is incumbent
    assert higher in queue
    assert queue.metrics(1)["budget_drops_total"] == 1

    admitted, displaced = queue.admit(equal, 1, 300.0)
    assert admitted is False
    assert displaced is None
    assert higher in queue
    assert queue.metrics(1)["budget_drops_total"] == 2


def test_ttl_batch_demotion_and_metrics(monkeypatch):
    queue = PriorityEvictionQueue()
    expired = _hashed_block(1)
    live = _hashed_block(2)
    monkeypatch.setattr(priority_module.time, "monotonic", lambda: 100.0)
    _set_meta(queue, expired, 50, expiry=120.0)
    _set_meta(queue, live, 50, expiry=180.0)
    queue.try_insert(expired)
    queue.try_insert(live)
    monkeypatch.setattr(priority_module.time, "monotonic", lambda: 150.0)
    assert queue.release_expired() == [expired.block_id]
    assert queue.pop_lowest() is live
    assert queue.metrics(2)["ttl_expiries_total"] == 1


def test_ttl_clears_suspended_metadata_without_freeing_active_block(monkeypatch):
    queue = PriorityEvictionQueue()
    queued = _hashed_block(1)
    suspended = _hashed_block(2)
    live = _hashed_block(3)
    monkeypatch.setattr(priority_module.time, "monotonic", lambda: 100.0)
    _set_meta(queue, queued, 50, expiry=120.0, freed=20.0)
    _set_meta(queue, suspended, 50, expiry=120.0, freed=10.0)
    _set_meta(queue, live, 50, expiry=180.0, freed=30.0)
    assert queue.try_insert(queued)
    assert queue.try_insert(suspended)
    assert queue.try_insert(live)
    queue.suspend(suspended)

    monkeypatch.setattr(priority_module.time, "monotonic", lambda: 150.0)
    # Only the queued candidate is safe for BlockPool.get_new_blocks() to put
    # back on the free LRU. The suspended block merely loses stale metadata.
    assert queue.release_expired() == [queued.block_id]
    assert suspended.block_id not in queue._meta
    assert suspended.block_id not in queue._in_queue
    assert live.block_id in queue._meta
    assert live in queue
    assert queue.metrics(3) == {
        "protected_blocks": 1,
        "queued_protected_blocks": 1,
        "budget_blocks": 3,
        "priority_evictions_total": 0,
        "ttl_expiries_total": 2,
        "budget_drops_total": 0,
    }


def test_scope_ownership_and_range_offset():
    queue = PriorityEvictionQueue()
    blocks = [_hashed_block(10), _hashed_block(11)]
    queue.apply_directives(
        blocks,
        [{"start": 32, "end": 48, "priority": 70}],
        "alice",
        block_size=16,
        start_block_index=2,
    )
    assert queue._meta[10].priority == 70
    assert 11 not in queue._meta
    queue.apply_directives(
        blocks[:1],
        [{"start": 32, "end": 48, "priority": 20}],
        "bob",
        block_size=16,
        start_block_index=2,
    )
    assert queue._meta[10].priority == 70
    queue.apply_directives(
        blocks[:1],
        [{"start": 32, "end": 48, "priority": 20}],
        "alice",
        block_size=16,
        start_block_index=2,
    )
    assert queue._meta[10].priority == 20


def test_block_pool_wires_all_retention_lifecycle_paths():
    source = (HERE / "vllm/v1/core/block_pool.py").read_text()
    assert "if num_cached_blocks >= num_full_blocks:" in source
    assert source.count("self._apply_retention_hook(") == 2
    assert 'directive.get("covers_prompt")' in source
    assert "release_expired()" in source
    assert "admitted, displaced = pq.admit(" in source
    assert "self.priority_eviction_queue.suspend(block)" in source
    assert "self.priority_eviction_queue.drain()" in source
    assert "+ self.priority_eviction_queue.num_blocks" in source


def test_scheduler_wires_bounded_prefix_replay():
    source = (HERE / "vllm/v1/core/sched/scheduler_retention.py").read_text()
    assert "VLLM_PREFIX_CACHE_MIN_RECOMPUTE_TOKENS" in source
    assert "request.num_tokens - min_recompute_tokens" in source
    assert "manager.coordinator.find_longest_cache_hit(" in source
    assert "manager.get_computed_blocks = get_computed_blocks" in source


def test_protocol_fields_validation_and_sampling_round_trip(tmp_path):
    try:
        import vllm._C  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("full protocol validation runs in the vendor-image CI gate")
    protocol_dir = tmp_path / "chat_completion"
    protocol_dir.mkdir()
    shutil.copyfile(
        Path("vllm/entrypoints/openai/chat_completion/protocol.py"),
        protocol_dir / "protocol_base.py",
    )
    shutil.copyfile(
        HERE / "vllm/entrypoints/openai/chat_completion/protocol_retention.py",
        protocol_dir / "protocol.py",
    )
    module = _load_module("retention_protocol_test", protocol_dir / "protocol.py")
    request = module.ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        retention_directives=[{"covers_prompt": True, "priority": 90, "duration": 300}],
        retention_scope="cortex:v1:" + "b" * 64,
    )
    params = request.to_sampling_params(16, {})
    assert params.extra_args["retention_directives"][0]["covers_prompt"] is True
    assert params.extra_args["retention_scope"].startswith("cortex:v1:")

    with pytest.raises(ValueError, match="non-increasing"):
        module.ChatCompletionRequest(
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            retention_directives=[
                {"start": 0, "end": 16, "priority": 20},
                {"start": 16, "end": 32, "priority": 90},
            ],
        )


def test_block_pool_skips_cache_registration_for_kv_ephemeral():
    """kv_ephemeral requests (operator 2026-08-07): one-shot prompts never
    register prefix-cache hashes, so their blocks stay scratch and the #43447
    free path prepends them (recycled first) instead of displacing live
    sessions' reusable cache."""
    source = (HERE / "vllm/v1/core/block_pool.py").read_text()
    guard = source.index('_extra.get("kv_ephemeral")')
    hashing = source.index("cached_block_hash_to_block.insert")
    assert guard < hashing, "ephemeral guard must run before hash registration"
    # The guard returns before BOTH the hash insertion and the retention hook.
    guard_block = source[guard: source.index("new_full_blocks = blocks", guard)]
    assert "return" in guard_block


def test_protocol_kv_ephemeral_round_trip(tmp_path):
    try:
        import vllm._C  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("full protocol validation runs in the vendor-image CI gate")
    protocol_dir = tmp_path / "chat_completion_eph"
    protocol_dir.mkdir()
    shutil.copyfile(
        Path("vllm/entrypoints/openai/chat_completion/protocol.py"),
        protocol_dir / "protocol_base.py",
    )
    shutil.copyfile(
        HERE / "vllm/entrypoints/openai/chat_completion/protocol_retention.py",
        protocol_dir / "protocol.py",
    )
    module = _load_module("retention_protocol_eph_test", protocol_dir / "protocol.py")
    request = module.ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        kv_ephemeral=True,
    )
    params = request.to_sampling_params(16, {})
    assert params.extra_args["kv_ephemeral"] is True
    plain = module.ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
    )
    plain_params = plain.to_sampling_params(16, {})
    assert not (plain_params.extra_args or {}).get("kv_ephemeral")


def test_coordinator_release_maintains_free_queue_watermark(monkeypatch):
    """2026-08-08: releasing protection only to the immediate allocation's
    need kept the free queue permanently near-empty — parked prompts were
    recycled within minutes of protection release (agent-nami 179K in 2m21s
    on an 11.3M pool). The coordinator must raise the release target to a
    pool-fraction watermark so released prompt blocks retire through a DEEP
    LRU free queue instead of being recycled immediately."""
    source = (HERE / "vllm/v1/core/kv_cache_coordinator.py").read_text()
    assert "_protected_free_watermark_blocks" in source
    assert "VLLM_PROTECTED_FREE_WATERMARK_FRAC" in source
    assert "target_free_blocks = max(" in source

    # Exercise the watermark math on a stub (method is self-contained).
    import types

    ns: dict = {"os": __import__("os")}
    method_src = source[source.index("    def _protected_free_watermark_blocks") :]
    method_src = method_src[: method_src.index("\n    def release_protected_prompt_blocks")]
    exec("import os\n" + "def _wm(self):" + method_src.split(") -> int:", 1)[1], ns)

    class _Pool:
        num_gpu_blocks = 10_000

    stub = types.SimpleNamespace(block_pool=_Pool())
    monkeypatch.delenv("VLLM_PROTECTED_FREE_WATERMARK_FRAC", raising=False)
    assert ns["_wm"](stub) == 3_000  # default 0.30
    stub2 = types.SimpleNamespace(block_pool=_Pool())
    monkeypatch.setenv("VLLM_PROTECTED_FREE_WATERMARK_FRAC", "0")
    assert ns["_wm"](stub2) == 0  # hatch: restore old behaviour
    stub3 = types.SimpleNamespace(block_pool=_Pool())
    monkeypatch.setenv("VLLM_PROTECTED_FREE_WATERMARK_FRAC", "2.0")
    assert ns["_wm"](stub3) == 9_500  # clamped to 0.95


def _protection_stub_cls():
    """Bind the real protection/release methods onto a minimal stub manager."""
    import textwrap
    from collections import deque

    source = (HERE / "vllm/v1/core/single_type_kv_cache_manager.py").read_text()

    def extract(name: str) -> str:
        start = source.index(f"    def {name}")
        nxt = source.index("\n    def ", start + 10)
        return textwrap.dedent(source[start:nxt])

    ns: dict = {"deque": deque, "Sequence": list, "KVCacheBlock": object}
    for name in (
        "_protect_prompt_blocks",
        "_release_one_protected_prompt_block",
        "_compact_protected_prompt_queue",
        "release_protected_prompt_blocks",
    ):
        exec(extract(name), ns)  # noqa: S102 — sources under test

    class _Blk:
        def __init__(self, bid):
            self.block_id = bid
            self.block_hash = ("h", bid)
            self.is_null = False
            self.ref_cnt = 1

    class _Pool:
        def __init__(self):
            self.blocks = {}
            self.freed = []

        def touch(self, blocks):
            pass

        def free_blocks(self, blocks):
            self.freed.extend(b.block_id for b in blocks)

        def get_num_free_blocks(self):
            return len(self.freed)

    class _Mgr:
        enable_caching = True
        _protect_prompt_blocks = ns["_protect_prompt_blocks"]
        _release_one_protected_prompt_block = ns["_release_one_protected_prompt_block"]
        _compact_protected_prompt_queue = ns["_compact_protected_prompt_queue"]
        release_protected_prompt_blocks = ns["release_protected_prompt_blocks"]

        def __init__(self):
            self.block_pool = _Pool()
            self._protected_prompt_block_ids = set()
            self._protected_prompt_block_queue = deque()
            self._protected_prompt_block_seq = {}
            self._protected_prompt_seq_counter = 0

        def _trim_protected_prompt_blocks(self):
            pass

        def protect(self, bids):
            blocks = []
            for bid in bids:
                blk = self.block_pool.blocks.get(bid) or _Blk(bid)
                self.block_pool.blocks[bid] = blk
                blocks.append(blk)
            self._protect_prompt_blocks(blocks)

    return _Mgr


def test_protection_release_frees_prompt_tail_before_head():
    """agent-rui 2026-08-08: releasing a prompt's HEAD blocks first breaks the
    find_longest_cache_hit chain at block 0 — a 272K prefix read 0% cached
    while most of its blocks were still resident. Release must consume each
    prompt tail-first (the same convention free() documents), so a partially
    released prefix still yields partial hits."""
    mgr = _protection_stub_cls()()
    mgr.protect([1, 2, 3, 4, 5, 6])
    for _ in range(3):
        assert mgr._release_one_protected_prompt_block()
    assert mgr.block_pool.freed == [6, 5, 4]
    assert {1, 2, 3} <= mgr._protected_prompt_block_ids


def test_protection_release_is_lru_across_prompts_on_retouch():
    """A re-touched (still warm) prompt must move to the release-queue back;
    a burst release consumes least-recently-used prompts first, not
    protection-birth order (rui was re-touched 3min before the wave and
    still died first under FIFO)."""
    mgr = _protection_stub_cls()()
    mgr.protect([1, 2, 3])      # prompt A (older)
    mgr.protect([11, 12, 13])   # prompt B
    mgr.protect([1, 2, 3])      # A re-touched — now the warmest
    for _ in range(3):
        assert mgr._release_one_protected_prompt_block()
    assert mgr.block_pool.freed == [13, 12, 11]  # B (LRU) went first, tail-first
    assert {1, 2, 3} <= mgr._protected_prompt_block_ids
    for _ in range(3):
        assert mgr._release_one_protected_prompt_block()
    assert mgr.block_pool.freed[3:] == [3, 2, 1]  # then A, tail-first


def test_protection_queue_compacts_stale_retouch_entries():
    """Lazy LRU dedup leaves superseded entries behind; the queue must stay
    bounded (≤2× live) across chatty re-registrations of the same prompt."""
    mgr = _protection_stub_cls()()
    for _ in range(50):
        mgr.protect([1, 2, 3, 4])
    assert len(mgr._protected_prompt_block_queue) <= 2 * len(
        mgr._protected_prompt_block_ids
    )
    for _ in range(4):
        assert mgr._release_one_protected_prompt_block()
    assert mgr.block_pool.freed == [4, 3, 2, 1]
    assert not mgr._release_one_protected_prompt_block()


def test_tool_parser_progress_overlay_is_bounded_and_metadata_only():
    """A parser-suppressed delta can stay externally silent for minutes.

    The overlay emits a standards-compliant SSE comment at bounded cadence,
    but leaves the parser and its atomic tool-JSON emission untouched.
    """
    patch = (HERE / "diffs/tool_parser_progress.patch").read_text()
    assert patch.count("diff --git ") == 1
    assert (
        "diff --git a/vllm/entrypoints/openai/chat_completion/serving.py "
        "b/vllm/entrypoints/openai/chat_completion/serving.py"
    ) in patch
    assert "deepseekv4_tool_parser.py" not in patch
    assert "deepseekv32_tool_parser.py" not in patch

    assert '_TOOL_PARSER_PROGRESS_INTERVAL_SECONDS: Final = 5.0' in patch
    assert '"stage": "tool_parser"' in patch
    assert '"request_id": request_id' in patch
    assert '"sequence": sequence' in patch
    assert 'return f": vllm-progress ' in patch
    assert "data:" not in patch

    progress_branch = patch.index("# A tool parser can intentionally buffer")
    continue_after_progress = patch.index("continue", progress_branch)
    for proof in (
        "output.token_ids",
        "tool_parser is not None",
        "request.tools",
        'request.tool_choice != "none"',
        "time.monotonic()",
        "_TOOL_PARSER_PROGRESS_INTERVAL_SECONDS",
        "yield _tool_parser_progress_comment",
    ):
        assert patch.index(proof, progress_branch) < continue_after_progress

    dockerfile = (HERE / "Dockerfile").read_text()
    timing_apply = dockerfile.index("--patch /tmp/frontend_timing.patch")
    progress_apply = dockerfile.index("--patch /tmp/tool_parser_progress.patch")
    compile_gate = dockerfile.index("python -m py_compile")
    assert timing_apply < progress_apply < compile_gate
