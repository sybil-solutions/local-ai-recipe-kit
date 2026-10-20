#!/usr/bin/env python3
"""Turn one kit submission into the field set LocalMaxxing (localmaxxing.com) shows for a run, so it can be posted there.
Only measured values are copied; anything missing stays null. A scrubber refuses to write personal paths, IPs, hostnames
you pass with --private, or anything that looks like a token.

  python3 tools/to_localmaxxing.py submissions/<card>/<file>.json --private myhost myuser --out lm.json
"""
import argparse, json, re, sys

SECRETISH = re.compile(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|(?:\d{1,3}\.){3}\d{1,3}|/(?:Users|home)/[^/\s\"]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission"); ap.add_argument("--out", required=True)
    ap.add_argument("--private", nargs="*", default=[], help="words that must not appear (hostnames, usernames)")
    ap.add_argument("--engine-name", default="sglang + trellis-serve (EXL3)")
    ap.add_argument("--engine-repo", default="https://github.com/0xSero/trellis-serve")
    a = ap.parse_args()
    s = json.load(open(a.submission))
    sp, L, pr = s["speed"], s["launch"], s["probe"]
    c1 = (sp.get("decode") or {}).get("C1") or {}
    r0 = ((c1.get("rounds") or [{}])[0]) if isinstance(c1.get("rounds"), list) else {}
    gpus = [g for g in pr.get("gpus", []) if g.get("vendor") in ("nvidia", "intel", "amd")]
    gpu = gpus[0] if gpus else {}
    nm = s.get("needs_measured") or {}
    # the peak VRAM of one card: the busiest GPU, never the sum over the box
    peak = nm.get("gpu_used_peak_mib_max")
    if peak is None and len(gpus) == 1:
        peak = nm.get("gpu_used_peak_mib")          # one card: the sum is that card
    pre = (sp.get("prefill") or {})
    run = {
        "model": s.get("target"),
        "hardware": {"gpuName": gpu.get("name"), "gpuCount": len(gpus), "vramGb": gpu.get("vram_gb"),
                     "cpu": pr.get("host", {}).get("cpu")},
        "engine": {"engineName": a.engine_name, "engineRepository": a.engine_repo, "engineBuild": L.get("image"),
                   "quantization": "EXL3-3.05bpw"},
        "engineFlags": {"commandSnippet": " ".join(L.get("args", [])), "env": L.get("env"),
                        "kvCacheDtype": (sp.get("server_args") or {}).get("kv_cache_dtype"), "specDecoding": False},
        "contextLength": (sp.get("server_args") or {}).get("context_length"),
        "batchSize": 1,
        "tokSOut": (c1.get("aggregate") or {}).get("median"),
        "outputTokens": r0.get("tokens"),
        "ttftMs": round(r0["ttft_max"] * 1000, 1) if r0.get("ttft_max") else None,
        "tokSPrefill": {k: v.get("median") for k, v in pre.items() if isinstance(v, dict) and "median" in v},
        "gpuPowerWatts": c1.get("gpu_power_w_mean"),
        "peakVramGb": round(peak / 1024, 2) if peak else None,
        "notes": (f"Protocol: local-ai-recipe-kit PROTOCOL.md (prefill 8k-64k x3 medians; decode C1-C4 natural-length answers, no "
                  f"max_tokens cap; cache-bust nonce per prompt). Quality vs exllamav3 reference panel: top-1 "
                  f"{s['quality']['panel'].get('top1_agreement')}, KL {s['quality']['panel'].get('mean_kl_top20')}. "
                  f"{s.get('notes', '')}").strip(),
    }
    txt = json.dumps(run)
    bad = [m for m in SECRETISH.findall(txt) if m not in ("0.0.0.0", "127.0.0.1")]
    bad += [w for w in a.private if w and re.search(r"\b" + re.escape(w) + r"\b", txt, re.I)]
    if bad:
        sys.exit(f"refusing to write: private-looking strings found: {sorted(set(bad))}")
    missing = [k for k in ("outputTokens", "ttftMs", "gpuPowerWatts", "peakVramGb") if run.get(k) is None]
    if missing:
        print("WARNING: " + a.out + " has no " + ", ".join(missing) + ": this run did not measure them",
              file=sys.stderr)
    json.dump(run, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
