# Target: DeepSeek-V4.1-Flash-1017 on 4x RTX PRO 6000 (TP4, 512K), vLLM-Moet r13, Engram rows read from NVMe

- Weights: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`
  (476 GB: 48 safetensors shards in the checkpoint's own FP8 / MXFP4 formats; no conversion).
- Engine: vLLM nightly `df3bd094` + the sm_120 patches of `github.com/kacper-daftcode/vLLM-Moet` at `5d25cff`
  (`Dockerfile.sm120-dsv41-nightly`, the "-r13" image: b12x W4A8 MoE with the fused persistent decode kernel,
  sparse-MLA gather loads, MXFP8 target lm_head, PCIe P2P all-reduce in NCCL's summation order, DSpark k = 7), plus one
  layer on top (`engram_disk`): the Engram embedding tables are read row by row from the checkpoint on NVMe at every
  step instead of being held in host RAM (bit-exact; 5.7 GiB of scales stay resident). Upstream keeps the 183 GiB table
  in shared memory (~330 GB host shm per server); this variant runs on a 128 GB host.
- How it fits: TP4, 73.4 GiB of weights per GPU, FP4 compressed KV record (`nvfp4_ds_mla`) + MXFP4 indexer cache:
  9.69M KV tokens on the GPUs (18x concurrency at 512K). Evicted KV goes to a 32 GiB pinned host region in /dev/shm and
  a disk tier (one file per 128-token block; vLLM never deletes them, run vLLM-Moet's `docker/sm120/kvcache-ttl.sh`).

## Needs (measured on the reference machine with `tools/memwatch.sh` from before launch — your probe must meet these)
| | 4x RTX PRO 6000 Workstation 96 GB |
|---|---|
| VRAM | all four cards: 368,132 MiB total at ready (0.94 utilisation), 387,916 MiB peak |
| Free system RAM | 47 GB (MemAvailable drop at peak; 32 GiB of it is the pinned KV offload region) |
| /dev/shm | ≥ 32 GiB free (`--ipc host`; tmpfs defaults to half of RAM, so 64 GiB offload does **not** fit a 128 GB host) |
| Disk | 476 GB for the weights on local NVMe (Engram rows are read randomly every decode step), plus room for the KV disk tier |
| PCIe | 5.0 x16 with GPU peer-to-peer (P2P all-reduce); measured 56.4 GB/s pinned host→device (`tools/hw_bw.py`) |

## Known-good config
### 4x RTX PRO 6000 Blackwell Workstation (sm_120) — `4x-rtx-pro-6000-96gb`
- Recipe: local-ai-registry launch `vllm-moet-deepseek-v4.1-flash-1017-512k-tp4` (image, argv, env, needs, lab proof).
  Image source: sybil-solutions/local-ai-images `vllm-moet-dsv41/` (vLLM-Moet `5d25cff` -r13 + the `engram_disk` layer,
  built by its release workflow). The submission below was measured on the same build, tagged locally
  (`ghcr.io/sybil-solutions/vllm-moet-dsv41-engramdisk@sha256:f3d7a175…`).
- Launch, spelled out (`--gpus all --ipc host`; upstream also passes `--ulimit memlock=-1:-1 --ulimit stack=67108864`,
  the measured container ran without them,
  `-v <models dir>/DeepSeek-V4.1-Flash-1017:/models:ro -v <cache dir>/dot-cache:/root/.cache
  -v <cache dir>/deep_gemm:/root/.deep_gemm -v <kv cache dir>:/srv/kvcache`; the image's entrypoint is `vllm serve`):
  `/models --served-model-name deepseek-v4.1-flash-1017 --host 0.0.0.0 --port 8000 --tensor-parallel-size 4
  --max-model-len 524288 --gpu-memory-utilization 0.94 --max-num-seqs 8 --kv-cache-dtype nvfp4_ds_mla
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 --tokenizer-mode deepseek_v41
  --max-num-batched-tokens 4096 --max-cudagraph-capture-size 64
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}'
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_tp_shard_min_rows":2048}' --moe-backend b12x
  --kv-offloading-size 32
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/srv/kvcache/v41-interleaved","n_read_threads":16,"n_write_threads":16,"locality":"LOCAL"}],"offload_prompt_only":false}}'
  --max-logprobs 128`
- Env: `VLLM_MOET_B12X_MOE_FUSED=1 VLLM_MOET_SPARSE_MLA_LDG=1 VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 VLLM_MOET_P2P_AR=1
  VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=16 VLLM_MOET_ENGRAM_DISK=1 VLLM_MOET_ENGRAM_DISK_THREADS=16 NCCL_P2P_DISABLE=0
  NCCL_P2P_LEVEL=SYS VLLM_ENGINE_READY_TIMEOUT_S=3600 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- First start ~22 min upstream (FlashInfer autotune + DeepGEMM JIT into the cache dir; use `--gpu-memory-utilization 0.92`
  the very first time); with a populated cache dir, starts took 7 min here (410-420 s to ready). Before a restart, delete stale `/dev/shm/vllm_offload_*.mmap` regions (root-owned,
  left behind by a stopped container): with one left over, the next start fails "Insufficient space in /dev/shm".
- Reference result (Threadripper PRO 9965WX, 128 GB RAM, PCIe 5.0, 300 W per GPU; **measured on a serving box with other agent
  traffic**, see the submission): prefill 8k 8,944 / 16k 9,937 / 32k 10,695 / 64k 11,119 tok/s; decode 1 user 137.6,
  2 users 191.7, 3 users 202.5, 4 users 263.2, 8 users 271.7 tok/s aggregate (prose answers); 1 user at 32k 73.1 tok/s.
  The single-stream code bench in vLLM-Moet (`tools/sm120_perf/fraser_bench.py`) gave 557-573 tok/s on its undisturbed
  runs (upstream: 641 median at 325 W with +5000 memory clock and the Engram table in RAM).

## Quality band (reference panel `reference/deepseek-v4.1-flash-1017-ref-panel.json`, `tools/score_ref_panel.py --api vllm`)
The reference is the same image with every vLLM-Moet runtime switch off (`VLLM_MOET_*=0`: stock vLLM kernels), stock
`fp8_ds_mla` KV + fp8 indexer, the default MoE backend and no speculative decoding (8 prompts, greedy, thinking off,
1,953 positions). That server scored against its own panel: top-1 0.9995, mean KL 0.00019 (the run-to-run floor).
This config: **top-1 agreement 0.974, mean KL 0.0195**, needle PASS at 245K and 510K tokens (depth 0.8).
Proposed band (this one run with a margin; tighten as more submissions land): top-1 ≥ 0.97 and mean KL ≤ 0.025. The gap to the reference comes from the config's lossy formats (FP4 KV record,
MXFP4 indexer keys, 8-bit MoE activations in b12x); which of them contributes how much was not measured.

## Safe knobs to tune
`--kv-offloading-size` (host RAM tier; must fit the free /dev/shm), `num_speculative_tokens` (5 vs 7: k = 7 wins on code,
loses on low-acceptance traffic), `--max-num-seqs` / `--max-model-len` (KV is shared), `VLLM_MOET_ENGRAM_DISK_THREADS`,
GPU power limit (record it — decode on this box sat at the 300 W cap). Each `VLLM_MOET_*=1` switch can be set to 0 to
fall back to the stock path. Anything else: describe it in the PR.
