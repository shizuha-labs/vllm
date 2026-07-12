# PLAT-4203 Path B overlay (retention + hitfix): sparkrun DSV4 sm121 image
# (0.21.1rc1.dev339) + backport of upstream vLLM #43447 free-queue eviction
# ordering (block survival) + #40624 "Option A" single coordinator-level EAGLE
# drop (env-gated VLLM_DSV4_SINGLE_EAGLE_DROP, default OFF) to fix the DSV4
# multi-group append-under-MTP 0-hit spiral (#42948). Pure Python overlay —
# NO CUDA rebuild; sm121 sparse-MLA kernels untouched.
FROM aidendle94/sparkrun-vllm-ds4-gb10:production-ready
COPY vllm/envs.py /opt/env/lib/python3.12/site-packages/vllm/envs.py
COPY vllm/v1/core/block_pool.py /opt/env/lib/python3.12/site-packages/vllm/v1/core/block_pool.py
COPY vllm/v1/core/kv_cache_utils.py /opt/env/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
COPY vllm/v1/core/single_type_kv_cache_manager.py /opt/env/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py
COPY vllm/v1/core/kv_cache_coordinator.py /opt/env/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py
