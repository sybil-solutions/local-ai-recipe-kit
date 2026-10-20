#!/usr/bin/env python3
"""Assemble one submission file from the measured artifacts (see PROMPT.md step 9). Copies numbers, never invents them."""
import argparse, csv, datetime, json, os, sys


def memwatch_summary(path):
    """What the memwatch CSV measured: the RAM drop at ready and the peak (PROTOCOL.md §5)."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    avail = [float(r["mem_available_gb"]) for r in rows if r.get("mem_available_gb")]
    swap = [float(r["swap_used_gb"]) for r in rows if r.get("swap_used_gb")]
    gpu = [float(r["gpu_used_mib"]) for r in rows if r.get("gpu_used_mib")]
    if not avail:
        # the writer could not compute the numbers: say so instead of dying on the first sample
        return {"samples": len(rows), "error": "the memwatch file holds no memory numbers; the tool that wrote it could not compute them"}
    # one column per GPU (gpu0_used_mib, gpu1_used_mib, ...): the peak of every card, not the sum
    cols = [c for c in rows[0] if c.startswith("gpu") and c.endswith("_used_mib") and c != "gpu_used_mib"]
    each = {c[len("gpu"):-len("_used_mib")]: max((float(r[c]) for r in rows if r.get(c)), default=0.0) for c in cols}
    # the "ready" row is the first sample taken when the server answered; the one before it is the baseline
    ready = next((r for r in rows if r.get("phase") == "ready"), None)
    load = [r for r in rows if r.get("phase") == "load"]
    base = load[-1] if load else None
    drop = None
    if base and ready and base.get("mem_available_gb") and ready.get("mem_available_gb"):
        drop = round(float(base["mem_available_gb"]) - float(ready["mem_available_gb"]), 2)
    return {"samples": len(rows), "mem_available_start_gb": avail[0] if avail else None,
            "mem_available_min_gb": min(avail) if avail else None,
            "host_ram_drop_peak_gb": round(avail[0] - min(avail), 2) if avail else None,
            "mem_available_drop_at_ready_gb": drop,
            "swap_used_at_ready_gb": float(ready["swap_used_gb"]) if ready and ready.get("swap_used_gb") else None,
            "swap_used_max_gb": max(swap) if swap else None,
            "gpu_used_peak_mib": max(gpu) if gpu else None,
            "gpu_used_peak_mib_max": max(each.values()) if each else None,
            "gpu_used_peak_mib_by_gpu": each or None,
            "memwatch_start_t": rows[0].get("t"), "memwatch_ready_t": ready.get("t") if ready else None,
            "memwatch_end_t": rows[-1].get("t")}


def main():
    ap = argparse.ArgumentParser()
    for k in ("probe", "sweep", "panel", "launch"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--needle"); ap.add_argument("--memwatch"); ap.add_argument("--notes", default="")
    ap.add_argument("--target", required=True); ap.add_argument("--handle", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda p: json.load(open(p)) if p else None
    sweep, panel = L(a.sweep), L(a.panel)
    needs = memwatch_summary(a.memwatch) if a.memwatch else None
    if needs and needs.get("mem_available_drop_at_ready_gb") is None:
        print("WARNING: the memwatch file has no ready row: the MemAvailable drop at ready is NOT measured. "
              "Start the tool with the server's health url (tools/memwatch.sh work/memwatch.csv http://127.0.0.1:30100)",
              file=sys.stderr)
    sub = {"schema": "local-ai-recipe-kit/submission/v1", "submitted": datetime.date.today().isoformat(),
           "handle": a.handle, "target": a.target, "launch": L(a.launch), "probe": L(a.probe),
           "quality": {"panel": {k: panel.get(k) for k in ("top1_agreement", "mean_kl_top20", "positions")},
                       "needle": L(a.needle)},
           "speed": {"status": sweep.get("status"), "config": sweep.get("config"), "prefill": sweep.get("prefill"),
                     "decode": {k: {"aggregate": v.get("aggregate"), "per_stream_mean": v.get("per_stream_mean"),
                                    "gpu_power_w_mean": v.get("gpu_power_w_mean"),
                                    "rounds": [{x: r.get(x) for x in ("aggregate", "tokens", "ttft_max", "finish")}
                                               for r in (v.get("rounds") or [])]}
                                for k, v in (sweep.get("decode") or {}).items()},
                     "prompt_sha256": sweep.get("prompt_sha256"),
                     "server_args": sweep.get("server_args"), "gates": sweep.get("gates")},
           "needs_measured": needs, "notes": a.notes}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sub, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
