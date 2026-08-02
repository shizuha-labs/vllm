"""Vendor-image gate for RFC-37003 retention and DeepSeek overlay wiring."""

import os
from types import SimpleNamespace

os.environ["VLLM_RETENTION_BUDGET_FRAC"] = "0.5"
os.environ["VLLM_DSV4_SINGLE_EAGLE_DROP"] = "1"

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.metrics.stats import SchedulerStats


assert envs.VLLM_RETENTION_BUDGET_FRAC == 0.5
assert envs.VLLM_DSV4_SINGLE_EAGLE_DROP is True

pool = BlockPool(8, True, 16)
blocks = pool.get_new_blocks(3)
for block in blocks:
    block.block_hash = b"hash" + block.block_id.to_bytes(4, "big")

request = SimpleNamespace(
    num_prompt_tokens=24,
    sampling_params=SimpleNamespace(
        extra_args={
            "retention_directives": [
                {"covers_prompt": True, "priority": 90, "duration": 300}
            ],
            "retention_scope": "cortex:v1:" + "a" * 64,
        }
    ),
)

# This is deliberately the fully-cached branch: directives must refresh even
# when no new hash is created during this call.
pool.cache_full_blocks(request, blocks, 3, 3, 16, 0)
assert pool.priority_eviction_queue.has_metadata(blocks[0].block_id)
assert pool.priority_eviction_queue.has_metadata(blocks[1].block_id)
assert not pool.priority_eviction_queue.has_metadata(blocks[2].block_id)

pool.free_blocks(blocks)
assert pool.priority_eviction_queue.num_blocks == 2
assert pool.get_num_free_blocks() == 7
assert pool.reset_prefix_cache() is True
assert pool.get_num_free_blocks() == 7
assert pool.get_retention_metrics()["priority_evictions_total"] == 0

os.environ["VLLM_RETENTION_BUDGET_FRAC"] = "0"
budget_zero_pool = BlockPool(8, True, 16)
block = budget_zero_pool.get_new_blocks(1)[0]
block.block_hash = b"budget00" + block.block_id.to_bytes(4, "big")
budget_zero_pool.priority_eviction_queue.apply_directives(
    [block], [{"start": 0, "end": 16, "priority": 90}], "scope", 16
)
budget_zero_pool.free_blocks([block])
assert budget_zero_pool.priority_eviction_queue.num_blocks == 0
assert budget_zero_pool.get_retention_metrics()["budget_drops_total"] == 1

os.environ["VLLM_RETENTION_BUDGET_FRAC"] = "0.25"
budget_pool = BlockPool(5, True, 16)
low, high = budget_pool.get_new_blocks(2)
for item in (low, high):
    item.block_hash = b"budget01" + item.block_id.to_bytes(4, "big")
budget_pool.priority_eviction_queue.apply_directives(
    [low], [{"start": 0, "end": 16, "priority": 20}], "low", 16
)
budget_pool.free_blocks([low])
budget_pool.priority_eviction_queue.apply_directives(
    [high], [{"start": 0, "end": 16, "priority": 90}], "high", 16
)
budget_pool.free_blocks([high])
assert high in budget_pool.priority_eviction_queue
assert low not in budget_pool.priority_eviction_queue
assert budget_pool.get_retention_metrics()["budget_drops_total"] == 1

chat_request = ChatCompletionRequest(
    model="dummy",
    messages=[{"role": "user", "content": "hi"}],
    retention_directives=[{"covers_prompt": True, "priority": 90, "duration": 300}],
    retention_scope="cortex:v1:" + "b" * 64,
)
sampling_params = chat_request.to_sampling_params(16, {})
assert sampling_params.extra_args["retention_directives"][0]["covers_prompt"]
assert sampling_params.extra_args["retention_scope"].startswith("cortex:v1:")

assert SchedulerStats().retention_metrics == {}
print("RETENTION VENDOR GATE PASS")
