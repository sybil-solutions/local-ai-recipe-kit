# Target: Qwen3.5-9B (GGUF Q4_K_M) on one 11 GB GPU, 128K context, MTP

- Weights: `bartowski/Qwen_Qwen3.5-9B-GGUF`, revision `182be2fd6c7bc44887d88a91cb03ff009cc9f549`
  (`Qwen_Qwen3.5-9B-Q4_K_M.gguf`, 6.17 GB).
- Engine: llamADALL v1-27 (llama.cpp b11243 fork), sm_75 build of
  `dalisoft/local-ai-recipe-kit@6bb85eba87be2baccbb98ff54d73a1a67f4fcf78` `llamadall/Dockerfile`.
- How it fits: all layers and 131,072 tokens of q8_0 KV in VRAM.

## Needs (measured on the reference machine — your probe must meet these)
| | RTX 2080 Ti 11 GB |
|---|---|
| VRAM | 10,217 MiB at ready, 10,265 MiB peak |
| Disk | 6.2 GB weights, 2.8 GB image |

## Known-good config — `rtx-2080-ti-11gb`
- Image: `docker build https://github.com/dalisoft/local-ai-recipe-kit.git#6bb85eba87be2baccbb98ff54d73a1a67f4fcf78:llamadall`
- Launch (`-v <models dir>:/models:ro`): `--model /models/Qwen_Qwen3.5-9B-Q4_K_M.gguf --alias qwen3.5-9b --jinja
  --n-gpu-layers 99 --fit off --ctx-size 131072 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
  --threads 12 --batch-size 2048 --ubatch-size 512 --split-mode none --spec-type draft-mtp --spec-draft-n-max 2
  --host 0.0.0.0 --port 8080`
- Reference result (protocol sweep): prefill 8k 1,611 / 16k 1,570 / 32k 1,478 / 64k 1,317 tok/s; decode 1 user
  135.4 tok/s, 1 user at 32k 108.6 tok/s; one request at a time.

## Quality
No reference panel exists for this model. Needle found at 118,313 tokens.
