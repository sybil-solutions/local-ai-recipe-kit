# Measurement protocol (every speed number in a submission)

Tool: `tools/sweep.py`. Same protocol on every card so numbers compare.

1. Warm-up request (512 tokens, not recorded).
2. **Prefill** 8k → 16k → 32k → 64k tokens: 3 fresh random-token prompts each (unique ids, so no prefix-cache hits),
   rate = prompt tokens / time-to-first-token. Reported as median with min/max and n. Lengths above the server's
   context are skipped (recorded as skipped).
3. **Decode at 1, 2, 3, 4 concurrent users** (+ optional `--high 8 16 32`): distinct short chat prompts, completions run
   to their natural end (no max_tokens cap), 2 rounds each. Aggregate tok/s = all completion tokens / (last token −
   first first-token); per-stream tok/s per request. Plus 1 user at 32k context.
4. **Early exit**: after each prefill length and after decode C1/C4, the median is compared with
   `reference/best_known.json[<card>]`; below 95 % of the best known → stop with exit code 3. Investigate first
   (other load on the box, thermals, PCIe link width/gen, a different config). Pass `--no-early-exit` only to record
   a deliberately slower config (say so in the submission).
5. **Memory needs** come from `tools/memwatch.sh` (MemAvailable drop and swap at ready and peak; GPU memory) — not from
   file sizes or estimates. The submission carries them in `needs_measured` (`make_submission.py --memwatch`); the
   validator rejects a submission that does not.
6. **Evidence fields** (LocalMaxxing-compatible): every decode prompt starts with a random cache-bust nonce line and its
   SHA-256 is recorded; GPU power (nvidia-smi, every 0.25 s) is averaged over each decode tier; time-to-first-token comes
   from the first streamed token of each request. `tools/to_localmaxxing.py` converts a submission to LocalMaxxing's run
   fields (it refuses to write personal paths, IPs, tokens, or words you pass with `--private`).
7. **Quality** first: `tools/score_ref_panel.py` must be inside the target's band before any speed number counts.

Hold nothing else on the GPU or CPU while measuring. Record `nvidia-smi` / `xpu-smi` and `free -g` before the run.
