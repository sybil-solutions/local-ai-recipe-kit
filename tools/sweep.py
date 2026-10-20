#!/usr/bin/env python3
"""Campaign sweep protocol (mandatory for every speed claim, both cards). See PROTOCOL.md.

  1. warm-up (512-token request, not recorded)
  2. PREFILL 8k -> 16k -> 32k -> 64k: REPS fresh random-token prompts each (unique ids, no radix hits), TTFT-based
     tok/s; median + min/max. After each length the median is compared with bench/best_known.json[card]; if it is
     below (1 - TOL) x best known the sweep EXITS EARLY (exit code 3, status EARLY_EXIT in the JSON).
  3. DECODE concurrency C = 1,2,3,4 (+ --high list, e.g. 8 16 32): C simultaneous chat requests with distinct short
     prompts, completions run to natural EOS (max_new_tokens = remaining context: uncapped), REPS rounds per C.
     aggregate tok/s = all completion tokens / (last token time - first first-token time); per-stream tok/s =
     (n-1)/(t_last - t_first) per request. Also C1 decode at 32k context. Early-exit check vs best known C1/C4.
  --update-best rewrites best_known.json entries this run beat (only for a full, non-early-exit run).
"""
import argparse, json, os, random, statistics, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DEFAULT = os.path.join(HERE, "..", "reference", "best_known.json")
OUT = None  # the result file of the run in progress: a crash handler still marks it FAILED
TOPICS = ["the history of the printing press", "how a jet engine works", "the causes of the French revolution",
          "photosynthesis in C4 plants", "how TCP congestion control works", "the life cycle of stars",
          "how vaccines train the immune system", "the economics of container shipping", "how compilers optimise loops",
          "the geology of volcanoes", "the rules and strategy of chess openings", "how GPS determines position",
          "the history of the bicycle", "how noise-cancelling headphones work", "the water cycle",
          "how databases implement transactions"]


def post(url, payload, first_only=False, timeout=7200):
    req = urllib.request.Request(url + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); times = []; last = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            last = json.loads(body); times.append(time.perf_counter())
            if first_only:
                break
    return t0, times, last


def tokenize(url, text):
    req = urllib.request.Request(url + "/tokenize", data=json.dumps({"prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


def rand_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


NONCE = __import__("secrets").token_hex(16)
PROMPT_HASHES = []


def chat_ids(url, i):
    t = (f"<|im_start|>user\n[cache-bust nonce: {NONCE}-{i}]\nWrite a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} "
         f"(variant {i}).<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    PROMPT_HASHES.append(__import__("hashlib").sha256(t.encode()).hexdigest())
    return tokenize(url, t)


class Power:
    """nvidia-smi power.draw sampled every 0.25 s while active (NVIDIA only; silently absent elsewhere)."""
    def __init__(self):
        self.samples, self.on = [], False
    def __enter__(self):
        import subprocess
        self.on = True
        def run():
            while self.on:
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5).stdout.split()
                    if o: self.samples.append(sum(float(x) for x in o))
                except Exception:
                    pass
                time.sleep(0.25)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.on = False; self.t.join(timeout=6)
    def mean(self):
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs), "all": xs}


def best_keys(path):
    """The card keys of a best_known.json (the '_note' entry is not a card)."""
    try:
        d = json.load(open(path))
    except Exception:
        return []
    return [k for k in d if not k.startswith("_")]


class SweepFailed(Exception):
    """The run cannot continue: the result file is marked FAILED with this reason."""


def mark_failed(why):
    """Mark the result file of a run that died: FAILED, with the reason and the data measured so far."""
    if not OUT:
        return
    try:
        d = json.load(open(OUT))
    except Exception:
        return
    d["status"] = "FAILED"
    d["error"] = why
    d["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(d, open(OUT, "w"), indent=1)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--card", required=True,
                    help="key in reference/best_known.json" +
                         (": " + ", ".join(best_keys(BEST_DEFAULT)) if best_keys(BEST_DEFAULT) else "") +
                         " (an unknown key means no baseline: no early exit)")
    ap.add_argument("--config", required=True, help="label: commit + env summary")
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--high", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dec-reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--no-early-exit", action="store_true")
    ap.add_argument("--update-best", action="store_true")
    ap.add_argument("--best", help="best-known JSON for early exit (default reference/best_known.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    OUT = a.out
    bk_path = a.best or BEST_DEFAULT
    best_all = json.load(open(bk_path)) if os.path.exists(bk_path) else {}
    known = a.card in best_all and not a.card.startswith("_")
    best = best_all.get(a.card, {})
    if not known:
        print("WARNING: card " + repr(a.card) + " is not a key of " + os.path.relpath(bk_path, os.path.join(HERE, "..")) +
              " (known keys: " + (", ".join(best_keys(bk_path)) or "none") + ")", file=sys.stderr)
        print("WARNING: no baseline for this card: the early exit cannot run and no gate is marked as passed "
              "(use --best <file> or record a baseline with --update-best)", file=sys.stderr)
    info = json.loads(urllib.request.urlopen(a.url + "/server_info", timeout=30).read())
    ctx_len = int(info.get("context_length") or (info.get("server_args") or {}).get("context_length") or 131072)
    res = {"card": a.card, "config": a.config, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ctx_len": ctx_len,
           "protocol": "bench/sweep.py v1", "tol": a.tol, "best_known_before": best,
           "baseline": a.card if known else "MISSING",
           "server_args": {k: (info.get("server_args") or info).get(k) for k in
                           ("max_running_requests", "max_total_num_tokens", "chunked_prefill_size", "kv_cache_dtype",
                            "context_length", "mem_fraction_static", "max_mamba_cache_size")},
           "prefill": {}, "decode": {}, "status": "RUNNING"}
    # the evidence of PROTOCOL.md §6: the cache-bust nonce and the SHA-256 of the decode prompts sent
    # so far. PROMPT_HASHES grows while the sweep runs, so every save() carries the hashes it knows -
    # an early exit keeps the evidence of the prompts it already asked.
    res["prompt_sha256"] = PROMPT_HASHES; res["nonce_prefix"] = NONCE

    def save():
        json.dump(res, open(a.out, "w"), indent=1)

    def fail(why):
        """Stop the run: keep what was measured, say what failed (the handler writes the status)."""
        res["error"] = why
        save()
        raise SweepFailed(why)

    def gate(key, val):
        b = best.get(key)
        ok = None if b is None else val >= (1 - a.tol) * b
        res.setdefault("gates", {})[key] = {"value": round(val, 2), "best_known": b, "pass": ok}
        if ok is False and not a.no_early_exit:
            res["status"] = f"EARLY_EXIT at {key}: {val:.1f} < {(1 - a.tol):.2f} x best {b}"
            save(); print(res["status"], flush=True); sys.exit(3)

    seed = int(time.time())
    save()                                   # the result file exists from now on, even if the run dies
    try:
        post(a.url, {"input_ids": rand_ids(512, seed), "sampling_params": {"temperature": 0}, "stream": True}, True)
    except Exception as e:
        fail(f"warm-up: {type(e).__name__}: {e}")
    for n in a.prefill:
        if n + 64 > ctx_len:
            res["prefill"][str(n)] = {"skipped": f"> context {ctx_len}"}; continue
        rates = []
        for rep in range(a.reps):
            try:
                t0, times, _ = post(a.url, {"input_ids": rand_ids(n, seed + 1000 * n + rep),
                                            "sampling_params": {"temperature": 0}, "stream": True}, True)
                rates.append(round(n / (times[0] - t0), 1))
            except Exception as e:
                fail(f"prefill {n} rep {rep}: {type(e).__name__}: {e}")
            print(f"prefill {n}: {rates[-1]} tok/s", flush=True)
        res["prefill"][str(n)] = med(rates); save()
        gate(f"prefill_{n}", statistics.median(rates))

    def run_conc(c, rnd, prefix=None):
        out = [None] * c
        def one(i):
            try:
                ids = (prefix or []) + chat_ids(a.url, rnd * 100 + i)
                p = {"input_ids": ids, "stream": True,
                     "sampling_params": {"temperature": 0, "max_new_tokens": max(64, (ctx_len - len(ids)) // max(c, 1) - 64)}}
                t0, times, last = post(a.url, p)
                n = (last or {}).get("meta_info", {}).get("completion_tokens", len(times))
                out[i] = {"t0": t0, "first": times[0], "last": times[-1], "tokens": n,
                          "finish": ((last or {}).get("meta_info", {}).get("finish_reason") or {}).get("type")}
            except Exception as e:            # one broken request must not take the whole run down
                out[i] = {"error": f"{type(e).__name__}: {e}"}
        ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        broken = [o["error"] for o in out if o.get("error")]
        if broken:
            return {"error": "; ".join(broken), "requests": c}
        tot = sum(o["tokens"] for o in out)
        span = max(o["last"] for o in out) - min(o["first"] for o in out)
        per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out if o["tokens"] > 1 and o["last"] > o["first"]]
        if span <= 0 or tot <= 0:
            return {"error": f"nothing to measure (tokens {tot}, span {span:.3f} s)", "requests": c}
        if not per:
            return {"error": "no stream produced more than one token", "requests": c}
        return {"aggregate": round(tot / span, 2), "per_stream_mean": round(statistics.mean(per), 2),
                "per_stream_min": round(min(per), 2), "tokens": tot,
                "finish": sorted({o["finish"] for o in out if o["finish"]}), "ttft_max": round(max(o["first"] - o["t0"] for o in out), 2)}

    def tier(name, c, rounds):
        """A tier that could not be measured ends the run: keep the rest, say what failed."""
        broken = [r["error"] for r in rounds if r.get("error")]
        if broken:
            res["decode"][name] = {"error": "; ".join(broken), "requests": c, "rounds": rounds}
            save()
            fail(f"decode {name}: " + "; ".join(broken))

    for c in list(a.conc) + list(a.high):
        with Power() as pw:
            rounds = [run_conc(c, r) for r in range(a.dec_reps)]
        tier(f"C{c}", c, rounds)
        agg = [r["aggregate"] for r in rounds]
        res["decode"][f"C{c}"] = {"aggregate": med(agg), "per_stream_mean": med([r["per_stream_mean"] for r in rounds]),
                                   "rounds": rounds, "gpu_power_w_mean": pw.mean()}
        print(f"decode C{c}: aggregate {statistics.median(agg):.2f} tok/s, per-stream "
              f"{statistics.median([r['per_stream_mean'] for r in rounds]):.2f}", flush=True)
        save()
        if c in (1, 4):
            gate(f"decode_C{c}", statistics.median(agg))
    pre = rand_ids(32768, seed + 99) if ctx_len > 40000 else None
    if pre:
        r = [run_conc(1, 90 + i, prefix=pre) for i in range(a.dec_reps)]
        tier("C1@32k", 1, r)
        res["decode"]["C1@32k"] = {"aggregate": med([x["aggregate"] for x in r]), "rounds": r}
        print(f"decode C1@32k: {statistics.median([x['aggregate'] for x in r]):.2f} tok/s", flush=True)
    res["status"] = "DONE"; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); save()
    if a.update_best:
        for k, g in res.get("gates", {}).items():
            if g["best_known"] is None or g["value"] > g["best_known"]:
                best_all.setdefault(a.card, {})[k] = g["value"]
        json.dump(best_all, open(bk_path, "w"), indent=1)
    print("SWEEP DONE", a.out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SweepFailed as e:
        mark_failed(str(e))
        print("SWEEP FAILED: " + str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        why = f"{type(e).__name__}: {e}"
        mark_failed(why)
        print("SWEEP FAILED: " + why, file=sys.stderr)
        sys.exit(1)
