#!/usr/bin/env python3
"""Assemble one submission file from the measured artifacts (see PROMPT.md step 9). Copies numbers, never invents them."""
import argparse, csv, datetime, json, os


def memwatch_summary(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    avail = [float(r["mem_available_gb"]) for r in rows if r["mem_available_gb"]]
    swap = [float(r["swap_used_gb"]) for r in rows if r["swap_used_gb"]]
    gpu = [float(r["gpu_used_mib"]) for r in rows if r.get("gpu_used_mib")]
    return {"samples": len(rows), "mem_available_start_gb": avail[0], "mem_available_min_gb": min(avail),
            "host_ram_drop_peak_gb": round(avail[0] - min(avail), 2), "swap_used_max_gb": max(swap) if swap else None,
            "gpu_used_peak_mib": max(gpu) if gpu else None}


def main():
    ap = argparse.ArgumentParser()
    for k in ("probe", "sweep", "panel", "launch"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--needle"); ap.add_argument("--memwatch"); ap.add_argument("--notes", default="")
    ap.add_argument("--target", required=True); ap.add_argument("--handle", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda p: json.load(open(p)) if p else None
    sweep, panel = L(a.sweep), L(a.panel)
    sub = {"schema": "local-ai-recipe-kit/submission/v1", "submitted": datetime.date.today().isoformat(),
           "handle": a.handle, "target": a.target, "launch": L(a.launch), "probe": L(a.probe),
           "quality": {"panel": {k: panel.get(k) for k in ("top1_agreement", "mean_kl_top20", "positions")},
                       "needle": L(a.needle)},
           "speed": {"status": sweep.get("status"), "error": sweep.get("error"), "card": sweep.get("card"),
                     "baseline": sweep.get("baseline"), "best_known_before": sweep.get("best_known_before"),
                     "config": sweep.get("config"), "prefill": sweep.get("prefill"),
                     "decode": {k: {"aggregate": v.get("aggregate"), "per_stream_mean": v.get("per_stream_mean"),
                                    "gpu_power_w_mean": v.get("gpu_power_w_mean"),
                                    "rounds": [{x: r.get(x) for x in ("aggregate", "tokens", "ttft_max", "finish")}
                                               for r in (v.get("rounds") or [])]}
                                for k, v in (sweep.get("decode") or {}).items()},
                     "prompt_sha256": sweep.get("prompt_sha256"), "nonce_prefix": sweep.get("nonce_prefix"),
                     "server_args": sweep.get("server_args"), "gates": sweep.get("gates")},
           "needs_measured": memwatch_summary(a.memwatch) if a.memwatch else None, "notes": a.notes}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sub, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
