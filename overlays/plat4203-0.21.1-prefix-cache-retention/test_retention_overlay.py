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
