# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority eviction queue for request-scoped KV retention directives.

Adapted from hyeongyun0916/vllm#1 (the implementation linked from upstream
RFC #37003).  State remains a sidecar keyed by block id so the hot-path
``KVCacheBlock`` and ``Request`` layouts stay unchanged.
"""

import heapq
import time
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(slots=True)
class RetentionMeta:
    priority: int
    expiry: float | None
    scope: str | None
    last_freed_time: float


class PriorityEvictionQueue:
    _COMPACTION_FLOOR = 64

    def __init__(self) -> None:
        self._meta: dict[int, RetentionMeta] = {}
        self._heap: list[tuple[int, float, int, int, KVCacheBlock]] = []
        self._in_queue: set[int] = set()
        self._generation: dict[int, int] = {}
        self.priority_evictions_total = 0
        self.ttl_expiries_total = 0
        self.budget_drops_total = 0

    @property
    def num_blocks(self) -> int:
        return len(self._in_queue)

    @property
    def num_protected_blocks(self) -> int:
        return len(self._meta)

    def __contains__(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._in_queue

    def has_metadata(self, block_id: int) -> bool:
        return block_id in self._meta

    def try_insert(
        self,
        block: KVCacheBlock,
        last_freed_time: float | None = None,
    ) -> bool:
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False
        if meta.expiry is not None and meta.expiry <= time.monotonic():
            self._meta.pop(block.block_id, None)
            self.ttl_expiries_total += 1
            return False
        if last_freed_time is not None:
            meta.last_freed_time = last_freed_time
        generation = self._generation.get(block.block_id, 0) + 1
        self._generation[block.block_id] = generation
        heapq.heappush(
            self._heap,
            (
                meta.priority,
                meta.last_freed_time,
                generation,
                block.block_id,
                block,
            ),
        )
        self._in_queue.add(block.block_id)
        self._maybe_compact()
        return True

    def admit(
        self,
        block: KVCacheBlock,
        budget_blocks: int,
        last_freed_time: float,
    ) -> tuple[bool, KVCacheBlock | None]:
        """Admit a protected candidate, displacing only lower priority.

        Returns ``(admitted, displaced_block)``. At equal priority the queued
        incumbent wins, preserving its established LRU position.
        """
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False, None
        if meta.expiry is not None and meta.expiry <= time.monotonic():
            self._meta.pop(block.block_id, None)
            self.ttl_expiries_total += 1
            return False, None
        if budget_blocks <= 0:
            self.record_budget_drop(block.block_id)
            return False, None
        if self.num_blocks < budget_blocks:
            return self.try_insert(block, last_freed_time), None

        lowest = self._peek_lowest()
        assert lowest is not None, "non-empty priority queue has no live heap entry"
        lowest_priority, incumbent = lowest
        if meta.priority <= lowest_priority:
            self.record_budget_drop(block.block_id)
            return False, None

        displaced = self._pop_lowest(count_eviction=False)
        assert displaced is incumbent
        self.budget_drops_total += 1
        admitted = self.try_insert(block, last_freed_time)
        assert admitted
        return True, displaced

    def suspend(self, block: KVCacheBlock) -> None:
        """Remove an active eviction candidate but retain its protection."""
        self._in_queue.discard(block.block_id)
        self._maybe_compact()

    def pop_lowest(self) -> KVCacheBlock | None:
        """Return the lowest-priority, least-recently-freed protected block."""
        return self._pop_lowest(count_eviction=True)

    def _pop_lowest(self, count_eviction: bool) -> KVCacheBlock | None:
        while self._heap:
            _, _, generation, block_id, block = heapq.heappop(self._heap)
            if block_id not in self._in_queue:
                continue
            if generation != self._generation.get(block_id):
                continue
            self._in_queue.discard(block_id)
            self._meta.pop(block_id, None)
            if count_eviction:
                self.priority_evictions_total += 1
            self._maybe_compact()
            return block
        return None

    def _peek_lowest(self) -> tuple[int, KVCacheBlock] | None:
        while self._heap:
            priority, _, generation, block_id, block = self._heap[0]
            if block_id in self._in_queue and generation == self._generation.get(
                block_id
            ):
                return priority, block
            heapq.heappop(self._heap)
        return None

    def _maybe_compact(self) -> None:
        """Bound lazy-deletion tombstones to O(live candidates + floor)."""
        limit = max(self._COMPACTION_FLOOR, self.num_blocks * 2 + 16)
        if len(self._heap) <= limit:
            return
        self._heap = [
            entry
            for entry in self._heap
            if entry[3] in self._in_queue and entry[2] == self._generation.get(entry[3])
        ]
        heapq.heapify(self._heap)

    def release_expired(self) -> list[int]:
        """Demote all expired candidates to ordinary LRU in one batch."""
        now = time.monotonic()
        expired: list[tuple[float, int]] = []
        for block_id in list(self._in_queue):
            meta = self._meta.get(block_id)
            if meta is not None and meta.expiry is not None and meta.expiry <= now:
                self._in_queue.discard(block_id)
                self._meta.pop(block_id, None)
                expired.append((meta.last_freed_time, block_id))
        self.ttl_expiries_total += len(expired)
        self._maybe_compact()
        expired.sort()
        return [block_id for _, block_id in expired]

    def unprotect(self, block_id: int) -> None:
        self._meta.pop(block_id, None)
        self._in_queue.discard(block_id)

    def record_budget_drop(self, block_id: int) -> None:
        if block_id in self._meta:
            self.budget_drops_total += 1
        self.unprotect(block_id)

    def clear(self) -> None:
        self._meta.clear()
        self._heap.clear()
        self._in_queue.clear()
        self._generation.clear()

    def drain(self) -> list[KVCacheBlock]:
        """Remove all queued candidates without recording real evictions."""
        blocks: list[KVCacheBlock] = []
        while self._heap:
            _, _, generation, block_id, block = heapq.heappop(self._heap)
            if block_id not in self._in_queue:
                continue
            if generation != self._generation.get(block_id):
                continue
            self._in_queue.discard(block_id)
            blocks.append(block)
        return blocks

    def metrics(self, budget_blocks: int) -> dict[str, int]:
        return {
            "protected_blocks": self.num_protected_blocks,
            "queued_protected_blocks": self.num_blocks,
            "budget_blocks": max(0, int(budget_blocks)),
            "priority_evictions_total": self.priority_evictions_total,
            "ttl_expiries_total": self.ttl_expiries_total,
            "budget_drops_total": self.budget_drops_total,
        }

    def apply_directives(
        self,
        blocks: list[KVCacheBlock],
        directives: list[dict],
        scope: str | None,
        block_size: int,
        start_block_index: int = 0,
    ) -> None:
        """Apply the highest-priority overlapping directive to each block.

        Any scope may escalate protection.  Only the owning scope may refresh,
        downgrade, or clear it.  This prevents one session from weakening
        shared prefix blocks protected by another session.
        """
        now = time.monotonic()
        for local_index, block in enumerate(blocks):
            if block.is_null or block.block_hash is None:
                continue
            token_start = (start_block_index + local_index) * block_size
            token_end = token_start + block_size
            best_priority = -1
            best_duration: float | None = None
            for directive in directives:
                start = directive.get("start", 0)
                end = directive.get("end")
                if end is not None and end <= token_start:
                    continue
                if start >= token_end:
                    continue
                priority = directive.get("priority", 0)
                if priority > best_priority:
                    best_priority = priority
                    best_duration = directive.get("duration")

            current = self._meta.get(block.block_id)
            if best_priority < 0:
                if scope is not None and current is not None and current.scope == scope:
                    self._meta.pop(block.block_id, None)
                continue

            expiry = now + best_duration if best_duration is not None else None
            current_priority = current.priority if current is not None else -1
            if best_priority > current_priority:
                self._meta[block.block_id] = RetentionMeta(
                    priority=best_priority,
                    expiry=expiry,
                    scope=scope,
                    last_freed_time=current.last_freed_time if current else 0.0,
                )
            elif current is not None and scope is not None and current.scope == scope:
                self._meta[block.block_id] = RetentionMeta(
                    priority=best_priority,
                    expiry=expiry,
                    scope=scope,
                    last_freed_time=current.last_freed_time,
                )
