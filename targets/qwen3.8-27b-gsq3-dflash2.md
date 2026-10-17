# Target: Qwen3.8-27B (GSQ3 3.125 bpw) on one 16 GB consumer GPU, DFlash2 K=7 speculative decoding

- Weights: `roofkid/Qwen3.8-27B-GSQ3-NInfer` @ `2359d374b0f6ed3cd400815e5114d32ce03b5899`
  (`qwen3_8_27b_gsq3.ninfer`, 12.42 GiB, SHA-256 `c6f27073…550d7` verified).
- Model: Qwen3.8-27B (hybrid GatedDeltaNet + attention, MTP-capable), GSQ Q3G128_F16S quant, 3.125 bpw.
- Engine: ninfer rtx4080-port v0.6.2 (github.com/roofkid/ninfer-4080). Linux/Docker image
  `roofkid/ninfer-4080:gsq3`; on Windows no container exists — build from source (CUDA 13.4,
  MSVC 2022, 8-file compatibility patch; see the submission on this card).
- How it fits: the whole artifact in VRAM (weights ~11-12 GiB + KV + workspace ≈ 15.0-15.3 GiB
  at ready), an optional host-KV pool (`--host-kv-mib`, 4 GiB default — 16 GiB changes nothing
  at ≤32k depth), DFlash2 K=7 draft with `--lm-head-draft`.

## Needs (measured on RTX 4060 Ti 16 GB — your probe must meet these)
| | RTX 4060 Ti 16 GB |
|---|---|
| VRAM | 15,019 MiB at ready, 15,344 MiB peak (DFlash2, flat through chat) |
| Free system RAM | 38-42 GB (weights read + host-KV pool) |
| Disk | ~13 GB free on a local NVMe SSD (12.42 GiB artifact) |
| GPU | sm_89 (Ada) measured; healthy power state 156-164 W, 2.6-2.8 GHz SM |

## Known-good config — RTX 4060 Ti 16 GB (Windows source build)

```text
ninfer-serve <models dir>/qwen3_8_27b_gsq3.ninfer --host 127.0.0.1 --port 8080
  --max-context 100000 --kv-capacity 100000 --kv-dtype rk4v4-e8
  --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 1200000
  --prefill-chunk 1024 --host-kv-mib 4096
  --spec dflash2 --draft-tokens 7 --lm-head-draft --preserve-thinking
```

Measured (medians of 3, warmed GPU, uncapped generations):
- 100K profile, rk4v4-e8 KV: prefill 1,147 / 1,120 / 1,052 / 932 tok/s @ 8k/16k/32k/64k;
  decode at depth 61.8 / 69.9 / 55.7 / 52.0 tok/s @ 8K/32K/64K/98K; chat C1 60.9 tok/s;
  structured-JSON at 32.8K prompt depth 102.6 tok/s server-timed (DFlash2 acceptance 81.2 %).
- 131,072 context with rk2v4-e8 KV: 657 tok/s prefill, 54.6 tok/s decode (six-gate lab run:
  load / chat / reasoning / tools / context / speed all pass).
- `ninfer_bench` (publisher methodology, 3 reps): MTP3 + ngram chain 174.7 / 183.4 / 164.7
  tok/s @ 8K/32K/64K; DFlash2 K=7 80.1 / 125.7 / 117.4 tok/s.

Fit limits (measured, not estimated): 131,072 with rk4v4-e8 KV is refused deterministically
(needs 3,419,995,392 B runtime vs 3,402,194,432 B budget — 17.8 MB over; ceiling ≈129.8K on a
fully drained GPU); rk2v4-e8 KV fits the full 131,072. 262,144 refuses at startup (5.70 GB
reservation vs 3.40 GB budget).

## Quality

Panel not run: the kit's reference panel is Qwen3.8-Flash-Next-specific and no Qwen3.8-27B
panel exists, so there is no honest top1-agreement/KL number. Point references from the
artifact's own card: MBPP 90-92 %, HumanEval 95-96 %, quick perplexity ~4.596.

## Safe knobs to tune

`--kv-dtype` (rk2v4-e8 fits the full 131,072; rk4v4-e8 is the higher-fidelity default at
≤129.8K), `--host-kv-mib` (4 default; 16 holds long/multiple histories), `--draft-tokens`
(K=7 measured), `--prefill-chunk`. Anything else: describe it in the PR.
