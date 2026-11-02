# Target: DeepSeek-V4.1-Flash-1017 on 4x H200 NVL (TP4, 256K), vLLM main + vllm#64893, Engram rows read from local SSD

- Weights: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`
  (510 GB: the checkpoint's own FP8 dense / MXFP4 expert formats; no conversion).
- Engine: vLLM main `51d2a855` (the published wheel) + vllm#64893 at `382b4f475f` (the `latency_specialized` decode
  executor for DeepSeek-V4.1-Flash on Hopper TP4) + lossless Hopper patches, built from
  sybil-solutions/local-ai-images `vllm-dsv41-h200/` (its README lists every layer and why each is exact):
  a P2P-store fallback for the executor's all-reduce on NVLink-bridged cards without NVSwitch, exact BF16 operands for
  the executor's dense kernels, hybrid adaptive verification (every draft verified up to 16 requests, adaptive above),
  MXFP4 Marlin padding 576 (vllm#56697), small-head split-KV sparse decode (vllm#59418), and an Engram reader that
  reads the 2 x 98 GB tables from the checkpoint at decode time with a 2M-row GPU cache, so host RAM stays small.
- How it fits: TP4, routed experts on Marlin W4A16 (the executor's own W4A8 MoE is not used: lossy), FP8 dense as
  W8A16 (`--linear-backend humming`), stock `fp8_ds_mla` KV: 7,587,334 KV tokens on the GPUs (28.9x concurrency at
  262,144), DSpark k = 5.

## Needs (measured on the reference machine with `tools/memwatch.sh` from before launch — your probe must meet these)
| | 4x H200 NVL 141 GB |
|---|---|
| VRAM | all four cards: 142,583 MiB each at ready (`--gpu-memory-utilization 0.985`, `--kv-cache-memory-bytes 41725000000`) |
| Free system RAM | 36.3 GB (MemAvailable drop at peak over the whole validation run; the measured box also ran another workload on its other four GPUs). Every run of this launch also passed under a hard 64 GB container memory+swap limit |
| /dev/shm | `--shm-size 16g` (no KV offload in this launch) |
| Disk | 510 GB for the weights on local SSD (Engram rows are read at random on every step); the reference machine reads them from a hardware RAID volume, not a bare NVMe drive |
| Interconnect | NVLink between all four cards (NV6 bridges here, no NVSwitch). The executor's all-reduce uses NVLink P2P stores when multicast is unavailable |

## Known-good config
### 4x H200 NVL (sm_90) — `4x-h200-nvl-141gb`
- Recipe: local-ai-registry launch `vllm-dsv41-h200-deepseek-v4.1-flash-1017-256k-tp4` (image, argv, needs, lab proof).
- Launch, spelled out (`--gpus '"device=0,1,2,3"' --shm-size 16g -p 8000:8000
  -v <models dir>/DeepSeek-V4.1-Flash-1017:/models:ro -v <cache dir>:/root/.cache`; the image's entrypoint is
  `vllm serve` and every environment switch is baked into the image):
  `/models --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port 8000 --tensor-parallel-size 4
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --default-chat-template-kwargs '{"thinking":true}' --engram-config '{"cpu_offload":true}'
  --max-model-len 262144 --max-num-seqs 256 --max-num-batched-tokens 16384
  --enable-prefix-caching --enable-chunked-prefill --enable-prompt-tokens-details
  --gpu-memory-utilization 0.985 --kv-cache-memory-bytes 41725000000
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
  --safetensors-load-strategy lazy --no-disable-custom-all-reduce --linear-backend humming --moe-backend marlin
  --additional-config '{"decode_backend":"latency_specialized"}' --max-logprobs 128`
- First start 575 s (FlashInfer and CuTe JIT into the cache dir); with a populated cache dir, 455-484 s to ready.
- Clients: ask for usage once at the end (`stream_options: {"include_usage": true}`, the OpenAI default). With a
  reasoning parser, `continuous_usage_stats` makes vLLM recount the reasoning tokens of the whole output on every
  chunk; on this launch it cut 8k decode at 160 users from 6,810 to 1,290 tok/s (API server pinned at one core).
- Reference result (AMD EPYC 9135, 256 GB RAM, 450 W per GPU), `tools/sweep.py` (random-token prefill, prose answers
  with thinking off, `--vocab-hi 128000` because the vocabulary is 129,280): prefill 8k 3,309 / 16k 3,489 / 32k 3,616 /
  64k 4,151 tok/s; decode 1 user 275.7, 2 users 423.8, 3 users 493.1, 4 users 518.3, 8 users 690.0, 16 users 956.3,
  32 users 1,548, 64 users 2,269, 128 users 3,368, 256 users 5,069 tok/s aggregate; 1 user at 32k 297.8 tok/s.
  Two things shape these numbers:
  - Random-token prompts make every Engram lookup miss the GPU row cache, so prefill is bound by reading rows from
    the SSD. Real text (a source-code corpus trimmed to 8k tokens, a fresh `cache_salt` per stream so no prefix
    hits) prefills at 29-33k tok/s on the same server once the row cache is warm, and 4.6k tok/s for the very first
    request after start.
  - DSpark accepts 2.2-2.6 tokens per step on the sweep's prose answers vs 4.3 when asked to summarise source code,
    and decode scales with that. On those code prompts (thinking off, output uncapped), the same server decodes 8k at
    485-489 tok/s for 1 user, 2,448 for 16, 4,692 for 64, 6,810 for 160 and 6,662 for 256.
  - At 128 and 256 users the sweep's per-stream budget (context / C) ends some answers by length.

## Quality band (reference panel `reference/deepseek-v4.1-flash-1017-ref-panel.json`, `tools/score_ref_panel.py --api vllm`)
This config: **top-1 agreement 0.976, mean KL 0.0153** (1,953 positions), needle PASS at 233,947 tokens (depth 0.8).
It sits inside the band of the 4x RTX PRO 6000 target (top-1 ≥ 0.97, mean KL ≤ 0.025). Every change in the image was
also checked against the image without it: greedy completions with per-token logprobs on 12 prompts differ only at
near-ties, inside the run-to-run spread of one build (mean |Δ logprob| 0.031 vs 0.026-0.029 between two runs).

## Safe knobs to tune
`VLLM_DSPARK_ADAPTIVE_MIN_REQS` (requests up to which every draft is verified; 16 measured best, 32 the alternative),
`num_speculative_tokens`, `DSV41_ENGRAM_GPU_CACHE_ROWS` (2M vs 4M measured equal), `--kv-cache-memory-bytes`,
`--max-num-seqs`. `DSV41_MHC_P2P=0` turns the P2P all-reduce fallback off on NVSwitch machines. Anything else: describe
it in the PR.
