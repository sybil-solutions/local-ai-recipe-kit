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
    ap.add_argument("--engine-name")
    ap.add_argument("--engine-repo")
    a = ap.parse_args()
    s = json.load(open(a.submission))
    sp, L, pr = s["speed"], s["launch"], s["probe"]
    c1 = (sp.get("decode") or {}).get("C1") or {}
    r0 = ((c1.get("rounds") or [{}])[0]) if isinstance(c1.get("rounds"), list) else {}
    quantization = next(
        (L.get("args", [])[i + 1] for i, value in enumerate(L.get("args", [])) if value == "--quantization" and i + 1 < len(L["args"])),
        None,
    )
    if a.engine_name is None:
        for i, value in enumerate(L.get("args", [])):
            if value in ("-m", "--module") and i + 1 < len(L["args"]):
                module_name = L["args"][i + 1]
                if module_name.startswith("sglang."):
                    a.engine_name = "sglang"
                elif module_name.startswith("vllm."):
                    a.engine_name = "vllm"
                break
    gpu = next((g for g in pr.get("gpus", []) if g.get("vendor") in ("nvidia", "intel", "amd")), {})
    pre = (sp.get("prefill") or {})
    run = {
        "model": s.get("target"),
        "hardware": {"gpuName": gpu.get("name"), "gpuCount": 1, "vramGb": gpu.get("vram_gb"), "cpu": pr.get("host", {}).get("cpu")},
        "engine": {"engineName": a.engine_name, "engineRepository": a.engine_repo, "engineBuild": L.get("image"),
                   "quantization": quantization},
        "engineFlags": {"commandSnippet": " ".join(L.get("args", [])), "env": L.get("env"),
                        "kvCacheDtype": (sp.get("server_args") or {}).get("kv_cache_dtype"),
                        "specDecoding": "--speculative-config" in L.get("args", [])},
        "contextLength": (sp.get("server_args") or {}).get("context_length"),
        "batchSize": None,
        "tokSOut": (c1.get("aggregate") or {}).get("median"),
        "outputTokens": r0.get("tokens"),
        "ttftMs": round(r0["ttft_max"] * 1000, 1) if r0.get("ttft_max") else None,
        "tokSPrefill": {k: v.get("median") for k, v in pre.items() if isinstance(v, dict) and "median" in v},
        "gpuPowerWatts": c1.get("gpu_power_w_mean"),
        "peakVramGb": round(s["needs_measured"]["gpu_used_peak_mib"] / 1024, 2) if (s.get("needs_measured") or {}).get("gpu_used_peak_mib") else None,
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
    json.dump(run, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
