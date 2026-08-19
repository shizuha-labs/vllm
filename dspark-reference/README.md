# DSpark lane parser source (reference)

Extracted from the running DSpark lane image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
(StatefulSets `deepseek-v4-flash-dspark-tp4b/tp4c/tp4d`, container path `/usr/local/lib/python3.12/dist-packages/vllm/parser/`).

Reference for CTX-645 backport: the DSpark lane's own `deepseek_v4.py` has `_dsml_arg_converter` + `_unwrap_wrapper_args`
(param-repair equivalents) but the streaming split-marker hold behavior lives in `streaming_parser_engine.py` (`_args_buffer`).
`_repair_param_dict` (upstream #41801) is absent by name — functionally covered by `_dsml_arg_converter`+`_unwrap_wrapper_args`.

Extracted 2026-08-19 by san (devops, CTX-739).
