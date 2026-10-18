# Target: Ternary Bonsai 2 27B (NInfer v3) on one 11 GB GPU, 140K context, MTP

- Weights: `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3`, revision `841f7812793972b5de6dc48910bdd1e2722ae48b`
  (`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, 9.52 GB).
- Engine: ninfer-all `d1a0e070` (github.com/iamwavecut/ninfer-all), sm_75 build.
- How it fits: weights (7.28 GiB) and 143,360 tokens of `rk4v4` KV in VRAM; 2 GiB pinned host KV.

## Needs (measured on the reference machine — your probe must meet these)
| | RTX 2080 Ti 11 GB |
|---|---|
| VRAM | 10,567 MiB at ready, 10,601 MiB peak |
| Free system RAM | 2 GiB pinned host KV |
| Disk | 9.6 GB |

## Known-good config — `rtx-2080-ti-11gb`
- Image: `ghcr.io/iamwavecut/ninfer-all@sha256:3c7e99a162cc1d3c9b586f1283f133f9f1500c6bac0adef5920080ba6f35ac01`
- Launch (`-v <models dir>:/models:ro`, `--entrypoint /opt/ninfer/sm75/ninfer-serve`):
  `/models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b --max-context 143360 --kv-capacity 143360
  --kv-dtype rk4v4 --gdn-state-fp16 --host-kv-mib 2048 --host-state-slots 8 --spec mtp --draft-tokens 3 --lm-head-draft
  --host 0.0.0.0 --port 8080`
- Reference result (protocol sweep): prefill 8k 1,358 / 16k 1,296 / 32k 1,189 / 64k 1,028 tok/s; decode 1 user
  113.2 tok/s, 1 user at 32k 96.2 tok/s; one request at a time.

## Quality
No reference panel exists for this model. Needle found at 129,353 tokens.
