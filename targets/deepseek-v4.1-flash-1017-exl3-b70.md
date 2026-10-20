# Target: DeepSeek-V4.1-Flash-1017 (EXL3 2.7 bpw) on one Intel Arc Pro B70, experts in DDR4, 262,144-token context, vision

- Weights: `0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw`, revision `e34ef55bda7c37d17d8dcc86a8a64e7cb875481e` (199 GB, KLD 0.0304 /
  top-1 94.95 % vs the native model on the model card's panel) **plus** the two engram shards `model-00047-of-00048.safetensors`,
  `model-00048-of-00048.safetensors` of `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170` (203 GB).
  `recipe/fetch_model.py` inside the image downloads both into one directory (375 GB on disk).
- Model: 284B-class MoE (40 layers, 384 routed experts top-6, engram n-gram tables on layers 1 and 14, VL tower), routed experts 2.7 bpw
  (K2-K5 per expert), dense linears + LM head EXL3.
- Engine: upstream SGLang `a6cf058` (V4.1) + branch `b70-xpu` of the port, in the image (see Launch).
- How it fits: 187 GB of expert blobs live in pinned host RAM (USM host chunks); a 17 GB two-pool LRU expert cache in VRAM
  (K3/K4 slots 17.7 MB, K2 slots 8.9 MB) is filled by write-through from the GEMV kernels; the two 96 GiB engram tables are pinned host
  chunks read by a Triton gather; KV fp8, 262,144-token session, XPU decode graph (bs 1).

## Needs (measured)
| | Arc Pro B70 32 GB |
|---|---|
| VRAM | the whole card (cache 17 GB + weights 6.4 GB; peak prefill ~30 GiB, `X3_CACHE_GB` trades decode speed for prefill/long-context headroom) |
| Free system RAM | ~400 GB: MemAvailable fell 496 -> 105 GB (peak drop 391 GB, ~113 GB left at steady state); swap peaks at ~200 GB during the ~15 min load |
| Disk | 375 GB free for the weights |
| PCIe | 4.0 x16, `hw_bw.py xpu`: host->device pinned 28.3 GB/s (decode speed is bound by this: cache misses read experts over PCIe) |

## Known-good config
### Arc Pro B70 (Battlemage) — `intel-arc-pro-b70-32gb`
- Image: `ghcr.io/0xsero/deepseek-v41-flash-b70@sha256:bf211db0f510024800b2ed31b72ecd8915ca81f53f150d3fa50c50012ce60cc6`
  (the package is private until its owner flips the visibility in the GitHub package settings).
- Fetch the weights (needs `HF_TOKEN` only if rate limited): `docker run --rm -v <models dir>/dsv41-1017-exl3:/model --entrypoint python3 <image> /opt/recipe/fetch_model.py /model`.
- Launch: `docker run -d --name dsv41-serve --security-opt no-new-privileges --device /dev/dri/renderD128 --shm-size 16g --memory 470g
  --memory-swap 470g -p 127.0.0.1:30281:30281 -e ZE_AFFINITY_MASK=0 -e API_KEY_FILE=/run/api_key -v <models dir>/dsv41-1017-exl3:/model:ro
  -v <0600 key file>:/run/api_key:ro <image>` (the image entrypoint `/opt/recipe/serve.sh` runs `python3 -m sglang.launch_server --model-path
  /model --device xpu --context-length 262144 --max-total-tokens 270000 --mem-fraction-static 0.84 --chunked-prefill-size 8192
  --max-running-requests 1 --disable-shared-experts-fusion --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 --served-model-name
  deepseek-v4.1-flash-1017 --api-key <key>` and sends one ~9k-token warm-up request after /health; ~15 min to ready). Exact argv/env in
  the submission's `launch`.
- Env (defaults in serve.sh): `X3_CACHE_GB=17 X3_IDX_BUDGET_MB=128 X3_DEEP_CHUNK_BUDGET=1100000000 X3_ROW_CHUNK=4096
  PYTORCH_ALLOC_CONF=expandable_segments:True`, engram host table + shared system USM env as in the submission.
- Result (kit protocol, random-id prefill): prefill 8k 413 / 16k 411 / 32k 399 / 64k 377 tok/s; decode 1 user 11.8 tok/s
  (2-4 users are served one at a time: 11.7-11.8 tok/s aggregate); 1 user at 32k context 11.8 tok/s; needle 250,724 tokens depth 0.8 PASS
  (777 s). Images work (encoder in the same process).
- The model has no reference panel in this kit and no other machine was used as a reference: `quality.panel` is null, so `validate.py`
  rejects the submission on that single point (recorded in the PR).

## DSpark (bundled mtp.0-2 drafter) — measured, not part of this config
The checkpoint's draft (3 stages: fp8 attention, router, 128 MXFP4 experts top-3, markov + confidence heads) was ported far enough to run
on XPU end to end (draft load, graphs, M=6 verify through the mixed-K cached MoE kernel; the target's output stays correct). Measured on the full
model: **1.04 tok/s, accept length 1.00-1.02** versus 11.8 tok/s without it, so it is off. Causes: the draft's proposals are wrong (still open;
the draft linears were first loaded without their fp8 block scales, fixed, acceptance unchanged) and its MXFP4 MoE is a torch dequant + bf16
matmul per (token, slot) pair, ~1 s per step. Misses per token do not fall with verify width on this offload design, so even a correct draft
could only amortise the non-MoE ~30 ms. Code paths stay inert unless `--speculative-algorithm DSPARK` is passed and are not in the image.

## Other runs on this machine (kept for honesty)
- Image `206339e7…` (chunk 8192, no deep-chunk budget): prefill 412/412/403/385 tok/s, decode C1 11.6, C2 11.9, C3 11.8, C4 11.8, C1@32k 11.8;
  its 250k needle ran out of memory (640 MiB allocation, 572 MiB free) — the cause of the later row-chunk/deep-chunk changes.
- Image `c3474850…` (chunk 4096): prefill 270-275 tok/s at 8k/16k/32k (aborted run) — prefill is chunk-bound (each chunk streams/reconstructs every
  active expert), which is why the submitted config uses 8192.

## Safe knobs to tune
`X3_CACHE_GB` (decode speed: 14 GB 11.1, 17 GB 11.8-12.7, 20 GB ~13.2, 22 GB 14.4 tok/s on short prompts; above ~17 GB long prompts OOM
during prefill), `--chunked-prefill-size` (4096: 272 tok/s prefill, 8192: 410; 16384 does not fit with a 17 GB cache),
`--context-length`/`--max-total-tokens` (the decode indexer's cost grows with the length bound: ctx 16384 is ~3 % faster than 262144).
Anything else: describe it in the PR.
