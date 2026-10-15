# local-ai-recipe-kit

Get a big model running on **your** GPU, measure it the same way we do, and send the result back as one file.
Good submissions become recipes in [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) (and show up in the
[Omarchy Local AI plugin](https://github.com/sybil-solutions/omarchy-local-ai) for machines that can run them). This repo is the
intake: raw, measured submissions live here so the registry stays small and reviewed.

**First target:** Qwen3.8-Flash-Next (180B MoE, 6B active, AA Intelligence Index 40) on one 24-32 GB consumer GPU —
experts in system RAM, n-gram table on NVMe. See [`targets/qwen3.8-flash-next-offload.md`](targets/qwen3.8-flash-next-offload.md)
for what it needs (VRAM, free RAM, NVMe space) and the known-good configs (RTX 3090, Arc Pro B70).

## Bounties: $50 per recipe

Cards and models with no tested recipe carry a bounty: [open bounties](https://github.com/sybil-solutions/local-ai-registry/issues?q=is%3Aissue+is%3Aopen+label%3Abounty).
Claim one with `/attempt` on the issue, run the steps below on that card, then turn the result into a recipe with
`lab.py try --on endpoint` in [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) and open one PR there.
Rules and payment: [BOUNTIES.md](https://github.com/sybil-solutions/.github/blob/main/BOUNTIES.md).

## How to contribute a result
1. Check the target's **Needs** table against your machine: `python3 tools/probe.py --models-dir ~/models`.
2. Open your coding agent on that machine in a clone of this repo and give it [`PROMPT.md`](PROMPT.md).
   It downloads the pinned weights, starts the published image, checks quality against the reference panel, runs the
   fixed speed protocol ([`PROTOCOL.md`](PROTOCOL.md)), tunes one knob at a time, and writes
   `submissions/<card id>/<date>-<handle>.json`.
3. `python3 tools/validate.py submissions/<card id>/<file>.json`, then open a PR with **only that file**.
   Failures are welcome too ("does not fit in 64 GB RAM", "crashes on driver X") — say so in `notes`.

## What is in here
| path | what |
|---|---|
| `PROMPT.md` | the agent prompt (rules + steps) |
| `PROTOCOL.md` | how every number is measured (prefill 8k/16k/32k/64k, decode 1-4 users, early exit) |
| `targets/` | one file per model/setup: weights, needs, known-good launches, quality band, safe knobs |
| `tools/probe.py` | GPUs, RAM, free disk, and whether the models dir really sits on NVMe (follows dm-crypt/LVM/btrfs); AMD cards from sysfs when `rocm-smi` is absent |
| `tools/memwatch.sh` | records RAM/swap/GPU memory while the server starts and runs (for the real `needs`); AMD VRAM from sysfs when `nvidia-smi` is absent |
| `tools/hw_bw.py`, `tools/nvme_rand.py` | host↔GPU bandwidth and NVMe random-read baselines |
| `tools/sweep.py` | the speed protocol |
| `tools/score_ref_panel.py`, `tools/needle.py` | quality vs the exllamav3 reference; long-context needle |
| `--server vllm` (sweep, score_ref_panel, needle) | the same tools against vLLM's OpenAI API: `--model <served name>` (or `SERVED_MODEL`), API key from `VLLM_API_KEY`; sweep takes the launch's settings via `--server-args '<json>'`, the panel scorer uses `prompt_logprobs` top-`--k` and reports how many reference ids fell outside it. Default stays SGLang |
| `tools/make_submission.py`, `tools/validate.py` | build and check a submission (CI runs the validator) |
| `reference/` | reference panel (teacher-forced top-20 logprobs from exllamav3) and best-known speeds per card |
| `submissions/<card id>/` | one JSON per measured run |

## Rules that keep submissions comparable
Measured numbers only (no estimates); no output-length caps; no power/clock changes; image pinned by digest; exact
argv and env; quality inside the target's band before speed counts. Details in `PROMPT.md`.

## From submission to recipe
Maintainers reproduce promising submissions, then add a recipe (with its measured `needs`) to local-ai-registry through
its normal review. Submissions stay here as the evidence.
