# Shizuha GLM-5.1 Production vLLM Provenance

This branch records the source-like contents of the vLLM package installed in
the GLM-5.1 production image used on the GB10/sm_121 cluster.

Live deployment:

- Kubernetes workload: `ai-models/glm51-fp8-tp16`
- Image: `registry.local:30500/vllm-mla-sm121@sha256:abc9c8ef4707f1bd9256015d6cf10ccc1905d6429da10da90414a0b5316438e5`
- Base image before the `iproute2` overlay: `registry.local:30500/vllm-mla-sm121@sha256:c02d00b8d52b8b9a230a8b70b6438bdb0192643c7ff431a65d8e43ccc4e63ee0`
- Installed vLLM version: `0.21.1rc1.dev201+g1fe330398`
- Version parent commit: `1fe3303983e1829fae25edfb0b93e8cbcfad96e6`

The production wheel was built with source-like files that are not fully
represented by the version parent alone. This branch therefore keeps the parent
commit and adds the recovered live wheel delta:

- 109 generated/vendor source files from the installed wheel
- 3 source patches used by the production image

Validation performed on 2026-06-18:

- Extracted `.py`, `.pyi`, and `py.typed` files from
  `/usr/local/lib/python3.12/dist-packages/vllm` in the live ready pod.
- Compared SHA-256 manifests against this branch.
- Result: `1841` source-like package files on both sides, zero hash diff.

Compiled objects and CUDA artifacts are pinned by the image digest above. Do
not infer compiled binary reproducibility from this source branch alone; rebuild
and validate a new digest before replacing production.
