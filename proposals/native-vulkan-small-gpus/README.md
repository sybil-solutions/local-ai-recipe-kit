# Measured native Vulkan proposals for small NVIDIA GPUs

These are measured, reproducible configurations for Qwen2.5-Coder-1.5B-Instruct Q4_K_M on a GeForce GT 1030 2 GB and a T1200 Laptop GPU 4 GB. Measurements were collected on the owner's Linux machines on October 7, 2026 (America/Toronto). The evidence retains UTC timestamps.

The current kit targets Qwen3.8-Flash-Next on much larger GPUs. These files propose a separate native workflow for small GPUs. They are not active registry recipes or `local-ai-recipe-kit/submission/v1` records, and they do not claim reference-panel qualification or an OCI image digest.

## Measurements

| Measurement | GT 1030 2 GB, i3-10100F | T1200 Laptop 4 GB, i7-11800H |
|---|---:|---:|
| Synthetic prefill, 512 tokens, mean tok/s | 184.77 | 1445.90 |
| Synthetic prefill, 2048 tokens, mean tok/s | 169.89 | 1332.29 |
| Synthetic decode, 128 tokens, mean tok/s | 14.04 | 117.02 |
| Natural HTTP C1, median end-to-end tok/s | 13.16 | 96.15 |
| Peak total GPU memory, MiB | 1529 | 1240 |
| Baseline GPU memory, MiB | 296 | 5 |
| Long prompt checked, actual prompt tokens | 7714 | 7713 |

Synthetic tests used `llama-bench -p 512,2048 -n 128 -r 3 -t 4 -ngl 99`. Its fixed-length workload is distinct from the HTTP test. Natural HTTP completions had no output-token cap and all finished with `stop`; C1/C2/C4 each used two rounds. Their aggregate metric includes request and prefill time. The server had one inference slot, so C2/C4 clients queued. Raw usage, timing, finish reasons, prompt hashes and standard deviations are in the JSON records.

Peak GPU memory includes the desktop's baseline allocation. The recorded MemAvailable drop is an observation during these runs, not a guarantee of minimum RAM requirements on another system. Swap was already in use before serving and did not increase at the observed peak.

The GT 1030 name alone does not establish its memory subtype or bandwidth. No bandwidth value is inferred, and these measurements describe this particular machine.

## Model and engine pins

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF@f86cb2c1fa58255f8052cc32aeede1b7482d4361`.
- File: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, 1,117,320,768 bytes.
- SHA-256: `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`.
- Engine: `ggml-org/llama.cpp@9c2e0e491a822adae1f0b1c831adb4160057d24f`, Release, native CPU instructions, Vulkan.
- Vulkan-Headers: `c46850864f4661461b0f6cb9922c058ffea4915e`.
- SPIRV-Headers: `86f980c731e62ae4eaf383d320449d71687936bf`.
- Build tools: isolated CMake 4.1.3 and Ninja 1.13.0; host GCC 16.2.1 and glslc 2026.3.

Both model downloads were checked against the official file size and SHA-256. The model repository declares Apache-2.0. No model weights are included here.

## Serving configuration

Both machines used the same serving settings:

```text
llama-server -m <MODEL_FILE> --alias qwen25-coder15b \
  --host 127.0.0.1 --port <PORT> -ngl 99 -c 8192 -np 1 \
  -t 4 -b 512 -ub 128 --flash-attn off --n-predict -1
```

The NVIDIA Vulkan ICD was selected explicitly through `VK_ICD_FILENAMES` and `VK_DRIVER_FILES`; `GGML_VK_VISIBLE_DEVICES=0` selected the only NVIDIA GPU. No GPU power limit, clock or fan settings were changed. NVIDIA container tooling was not installed, and no system daemon configuration was changed.

## Observed correctness and limitations

Both configurations loaded, returned chat responses, answered `17 * 23` with `391`, produced parseable JSON, and retrieved a marker from a prompt of about 7.7k tokens. These are basic smoke checks, not a broad model-quality evaluation or a full-context guarantee.

The tool test returned a `get_weather` JSON object as ordinary text. The OpenAI API response did not contain `tool_calls`. Tool calling is therefore unqualified in the measured configuration.

GT 1030 measured synthetic decode below the registry's 15 tok/s gate. Neither observation should be presented as passing all six registry checks. No Artificial Analysis score, teacher-forced top-1 agreement, or KL value is supplied or inferred from a different model.

## Reproduce and clean up

On a compatible Linux machine with an existing NVIDIA Vulkan driver, `python3`, Git, curl, a C++ compiler, glslc and the Vulkan loader:

```sh
python3 reproduce.py my-hardware-label > result.log
```

The script creates an isolated temporary workspace, downloads pinned sources and weights, builds with two jobs, performs the tests, emits a `RESULT_JSON:` line, and removes its temporary models, sources, virtual environment and build files. Its own server processes are stopped before completion. It uses an ephemeral loopback port. No global packages are installed.

The reproducer disables compiler caching to avoid writing into an existing shared cache. That is a build-cache storage refinement; the recorded measurement build arguments and serving arguments remain in the evidence. An initial build attempt failed before model download because SPIR-V headers were missing from the compiler's include path. The corrected configuration supplies that path explicitly.

Independent SSH checks after both runs found zero task-owned temporary directories and zero task-owned model processes. GPU memory returned to the recorded desktop baselines. Task-identifiable entries created in an existing compiler cache were removed separately; unrelated shared-cache entries were retained.

## Evidence files

- [GT 1030 observation](gt-1030-2gb.json)
- [T1200 observation](t1200-laptop-4gb.json)
- [Ephemeral reproducer](reproduce.py)
- [Artifact checks](validate.py)

Measured and contributed by [zsoXi](https://github.com/zsoXi). Maintainer review is needed to decide how a native serving contract and lower-memory targets should enter the official intake.
