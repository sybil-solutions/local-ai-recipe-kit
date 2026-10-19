# Target: Qwen3.5-9B (GGUF UD-Q4_K_XL with MTP head) on one 11 GB GPU, 128K context, MTP

- Weights: `unsloth/Qwen3.5-9B-MTP-GGUF`, revision `9716a636ee4bddc3fed678220b7a33dd2a4160ae`
  (`Qwen3.5-9B-UD-Q4_K_XL.gguf`, 6.14 GB).
- Engine: llamADALL v1-27 (llama.cpp b11243 fork), sm_75 build of
  `dalisoft/local-ai-recipe-kit@6bb85eba87be2baccbb98ff54d73a1a67f4fcf78` `llamadall/Dockerfile`.
- How it fits: all layers and 131,072 tokens of q8_0 KV in VRAM.

## Needs (measured on the reference machine — your probe must meet these)
| | RTX 2080 Ti 11 GB |
|---|---|
| VRAM | 10,185 MiB at ready, 10,223 MiB peak |
| Disk | 6.1 GB weights, 2.8 GB image |

## Known-good config — `rtx-2080-ti-11gb`
- Image: `docker build https://github.com/dalisoft/local-ai-recipe-kit.git#6bb85eba87be2baccbb98ff54d73a1a67f4fcf78:llamadall`
- Launch (`-v <models dir>:/models:ro`): `--model /models/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen3.5-9b --jinja
  --n-gpu-layers 99 --fit off --ctx-size 131072 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
  --threads 12 --batch-size 2048 --ubatch-size 512 --split-mode none --spec-type draft-mtp --spec-draft-n-max 2
  --host 0.0.0.0 --port 8080`
- Reference result (protocol sweep): prefill 8k 1,733 / 16k 1,682 / 32k 1,575 / 64k 1,395 tok/s; decode 1 user
  133.6 tok/s, 1 user at 32k 111.1 tok/s; one request at a time.

## Quality
No reference panel exists for this model. Needle found at 118,313 tokens.
