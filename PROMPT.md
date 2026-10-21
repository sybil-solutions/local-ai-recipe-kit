# Prompt: make a local-ai recipe work on my hardware and submit it

Copy everything below the line into your coding agent (Claude Code, Codex, OpenCode, …) running **on the machine with the
GPU**, from a clone of this repo. Fill in the three lines at the top first.

---

You are helping me run an open-weights model on my own hardware and submit a measured, reproducible recipe to
`github.com/0xSero/local-ai-recipe-kit`. Work carefully; every number you report must come from a command you ran.

- **Target**: `targets/qwen3.8-flash-next-offload.md` (or name another file in `targets/`)
- **Models directory** (where weights go): `<e.g. ~/models>`
- **My GitHub handle** (for the submission file name): `<handle>`

## Rules (do not break these)
1. Measured numbers only. Never estimate, extrapolate, or copy a number from a target file into the submission. If you
   could not measure something, leave it out and say why.
2. Do not cap generations: never set `max_tokens`, thinking budgets, or truncation to make a run pass or finish faster.
3. Never change GPU power limits, clocks or fans. Never disable CUDA graphs / force eager mode to make a config pass
   (if you must for a diagnosis, the result cannot be submitted).
4. Wrap every command that can hang (docker, network, servers, sockets) in a timeout. Start servers in the background
   (tmux or `docker run -d`) and poll their `/health` with a bounded loop.
5. Nothing else may use the GPU or hammer the CPU/RAM while you measure (close browsers/games, stop other models).
   Record `nvidia-smi` / `xpu-smi` / `free -g` before starting.
6. Never print secrets (HF tokens, API keys). Do not commit model weights, logs over 1 MB, or personal paths.
7. One change at a time. When a run fails, record the failure (it is useful data) before trying the next thing.

## Steps
1. **Probe the machine**: `python3 tools/probe.py --models-dir <models dir> --out work/probe.json`. Compare with the
   target's "Needs" table. If VRAM, RAM, free disk or NVMe fall short, stop and tell me exactly what is missing — do
   not try to squeeze the recipe onto hardware that does not meet the needs (that is a different recipe; see step 8).
2. **Bandwidth baselines** (they explain most speed differences between machines):
   `python3 tools/hw_bw.py cuda` (or `xpu`) inside the target's image, and
   `python3 tools/nvme_rand.py <path to the big n-gram file> 4096 131072` once the weights are downloaded.
   Save both outputs in `work/`.
3. **Download the weights** exactly at the revision the target pins (`huggingface_hub.snapshot_download(repo,
   revision=..., local_dir=...)`), then check the file sizes against the target's list.
4. **Launch the known-good config for the closest card** from the target file, unmodified except for paths, GPU index
   and port. Run `tools/memwatch.sh work/memwatch.csv` in the background from *before* the launch until after step 6.
   Wait for `/health`, then send one chat request and show me the answer.
5. **Correctness**: `python3 tools/score_ref_panel.py --url http://127.0.0.1:<port> --panel
   reference/qwen3.8-flash-next-exl3-ref-panel.json --out work/panel.json`. The target lists the pass band
   (top-1 agreement and mean KL vs the exllamav3 reference). Outside the band = do not tune speed yet; find the cause.
   If the target claims long context, also run `python3 tools/needle.py --url ... --tokens <per target> --depth 0.8
   --out work/needle.json`.
6. **Speed, fixed protocol** (`PROTOCOL.md`): `python3 tools/sweep.py --url http://127.0.0.1:<port> --card <card id>
   --config "<image digest + key env/args>" --out work/sweep.json`. It runs prefill 8k/16k/32k/64k (3 prompts each),
   decode at 1-4 concurrent users (natural-length answers), and 1 user at 32k context. It exits early (code 3) if a
   number is more than 5 % below the best known for that card — when that happens, report it and look for the reason
   (another process? thermal throttling? slower PCIe? different config?) before continuing.
   **If the target runs on llama.cpp** (a GGUF target), the tools above speak to an SGLang
   server, so run the llama.cpp server and put the shim in front of it — the tools are then used
   unchanged: `python3 tools/llamacpp_sglang_shim.py --upstream http://127.0.0.1:8081 --port 30100
   --model <alias>` and point `--url` at the shim. The target's reference panel (step 5) is recorded
   with `python3 tools/make_ref_panel.py --url http://127.0.0.1:30100 --prompts <prompts.json>
   --completion-tokens <n> --out work/panel.json` from the reference server: the same model in the
   higher-fidelity packing, which is then scored against the packing that is served.
7. **Tune, one knob at a time** (only the knobs the target lists as safe to tune, e.g. expert-cache size, chunk size,
   concurrency slots). For every attempt: re-run step 5 (quality must stay inside the band) and step 6, keep a short
   table: knob, value, panel, sweep medians, verdict. Keep the best config that passes quality.
8. **Different hardware class** (smaller VRAM, less RAM, no NVMe, AMD, Apple): say so, propose the smallest change
   that could work (e.g. pinned n-gram table instead of NVMe when there is RAM, fewer cache slots), and treat it as a
   new config through steps 4-7. If nothing passes quality, submit the failure — a documented "does not work on X
   because Y" is a valid, useful submission.
9. **Submit**: `python3 tools/make_submission.py --probe work/probe.json --sweep work/sweep.json --panel
   work/panel.json [--needle work/needle.json] --memwatch work/memwatch.csv --launch work/launch.json --target
   <target id> --handle <handle> --out submissions/<card id>/<YYYY-MM-DD>-<handle>.json`
   where `work/launch.json` is the exact launch you measured (`{"image": "...@sha256:...", "args": [...], "env":
   {...}, "shm": "...", "gpu": "..."}`). Then `python3 tools/validate.py submissions/<card id>/<file>.json`, open a
   pull request with only that one file, and paste the tuning table from step 7 into the PR description.

At the end, give me: the submission path, the panel result, the sweep medians, the measured VRAM / RAM / disk needs,
and anything that did not work.
