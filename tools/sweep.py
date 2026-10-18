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
  --api sglang (default: /generate, /tokenize, /server_info) or llamacpp (llama-server: /completion with token ids and
  cache_prompt off, /tokenize, /props, chat prompts rendered by the model's own template via /apply-template).
"""
import argparse, json, os, random, statistics, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = ["the history of the printing press", "how a jet engine works", "the causes of the French revolution",
          "photosynthesis in C4 plants", "how TCP congestion control works", "the life cycle of stars",
          "how vaccines train the immune system", "the economics of container shipping", "how compilers optimise loops",
          "the geology of volcanoes", "the rules and strategy of chess openings", "how GPS determines position",
          "the history of the bicycle", "how noise-cancelling headphones work", "the water cycle",
          "how databases implement transactions"]


API = "sglang"


def to_llamacpp(payload):
    """An SGLang /generate payload as a llama-server /completion payload (same ids, greedy, no prompt cache)."""
    sp = payload.get("sampling_params") or {}
    out = {"prompt": payload["input_ids"], "stream": payload.get("stream", True), "temperature": sp.get("temperature", 0),
           "cache_prompt": False, "n_predict": sp.get("max_new_tokens", 1 if payload.get("_first_only") else -1)}
    return out


def post(url, payload, first_only=False, timeout=7200):
    if API == "llamacpp":
        return post_llamacpp(url, payload, first_only, timeout)
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


def post_llamacpp(url, payload, first_only=False, timeout=7200):
    """Stream llama-server /completion; returns (t0, token times, last chunk in the SGLang meta_info shape)."""
    body = to_llamacpp({**payload, "_first_only": first_only})
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); times = []; last = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            j = json.loads(line[5:].strip())
            if j.get("content") or j.get("tokens"):
                times.append(time.perf_counter())
            if j.get("stop"):
                last = {"meta_info": {"completion_tokens": j.get("tokens_predicted", len(times)),
                                      "finish_reason": {"type": j.get("stop_type")}}}
                break
            if first_only and times:
                break
    return t0, times, last


def tokenize(url, text):
    if API == "llamacpp":
        req = urllib.request.Request(url + "/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["tokens"]
    req = urllib.request.Request(url + "/tokenize", data=json.dumps({"prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


def apply_template(url, user):
    """llama-server renders the model's own chat template (generation prompt included)."""
    req = urllib.request.Request(url + "/apply-template", data=json.dumps({"messages": [{"role": "user", "content": user}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt"]


VOCAB = 150000


def rand_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, VOCAB) for _ in range(n)]


NONCE = __import__("secrets").token_hex(16)
PROMPT_HASHES = []


RAW_TEMPLATE = None


def chat_ids(url, i):
    user = f"[cache-bust nonce: {NONCE}-{i}]\nWrite a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} (variant {i})."
    if API == "llamacpp":
        t = apply_template(url, user)
        PROMPT_HASHES.append(__import__("hashlib").sha256(t.encode()).hexdigest())
        return tokenize(url, t)
    t = (RAW_TEMPLATE.replace("{prompt}", user) if RAW_TEMPLATE else
         f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--card", required=True, help="key in best_known.json, e.g. rtx3090 / b70")
    ap.add_argument("--config", required=True, help="label: commit + env summary")
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--high", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dec-reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--no-early-exit", action="store_true")
    ap.add_argument("--raw-template", help="SGLang API: render chat prompts with this template ('{prompt}' marks the "
                    "user message) instead of the built-in ChatML no-think form, for models with another chat format, "
                    "e.g. GLM-5.3: '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think></think>'")
    ap.add_argument("--update-best", action="store_true")
    ap.add_argument("--best", help="best-known JSON for early exit (default reference/best_known.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api", choices=["sglang", "llamacpp"], default="sglang", help="server API (see module doc)")
    a = ap.parse_args()
    global API, VOCAB, RAW_TEMPLATE
    RAW_TEMPLATE = a.raw_template
    API = a.api
    if API == "llamacpp":  # random prompt ids must stay inside this model's vocabulary
        try:
            m = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"][0].get("meta") or {}
            VOCAB = min(VOCAB, int(m.get("n_vocab") or VOCAB))
        except Exception:
            pass
    bk_path = a.best or os.path.join(HERE, "..", "reference", "best_known.json")
    best_all = json.load(open(bk_path)) if os.path.exists(bk_path) else {}
    best = best_all.get(a.card, {})
    if API == "llamacpp":
        props = json.loads(urllib.request.urlopen(a.url + "/props", timeout=30).read())
        gs = props.get("default_generation_settings") or {}
        info = {"context_length": gs.get("n_ctx") or props.get("n_ctx"), "server_args": {"context_length": gs.get("n_ctx"),
                "max_running_requests": props.get("total_slots"), "build": props.get("build_info")}}
    else:
        info = json.loads(urllib.request.urlopen(a.url + "/server_info", timeout=30).read())
    ctx_len = int(info.get("context_length") or (info.get("server_args") or {}).get("context_length") or 131072)
    res = {"card": a.card, "config": a.config, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ctx_len": ctx_len,
           "protocol": "bench/sweep.py v1", "api": API, "tol": a.tol, "best_known_before": best,
           "server_args": {k: (info.get("server_args") or info).get(k) for k in
                           ("max_running_requests", "max_total_num_tokens", "chunked_prefill_size", "kv_cache_dtype",
                            "context_length", "mem_fraction_static", "max_mamba_cache_size")},
           "prefill": {}, "decode": {}, "status": "RUNNING"}

    def save():
        json.dump(res, open(a.out, "w"), indent=1)

    def gate(key, val):
        b = best.get(key)
        ok = b is None or val >= (1 - a.tol) * b
        res.setdefault("gates", {})[key] = {"value": round(val, 2), "best_known": b, "pass": ok}
        if not ok and not a.no_early_exit:
            res["status"] = f"EARLY_EXIT at {key}: {val:.1f} < {(1 - a.tol):.2f} x best {b}"
            save(); print(res["status"], flush=True); sys.exit(3)

    seed = int(time.time())
    post(a.url, {"input_ids": rand_ids(512, seed), "sampling_params": {"temperature": 0}, "stream": True}, True)
    for n in a.prefill:
        if n + 64 > ctx_len:
            res["prefill"][str(n)] = {"skipped": f"> context {ctx_len}"}; continue
        rates = []
        for rep in range(a.reps):
            t0, times, _ = post(a.url, {"input_ids": rand_ids(n, seed + 1000 * n + rep),
                                        "sampling_params": {"temperature": 0}, "stream": True}, True)
            rates.append(round(n / (times[0] - t0), 1))
            print(f"prefill {n}: {rates[-1]} tok/s", flush=True)
        res["prefill"][str(n)] = med(rates); save()
        gate(f"prefill_{n}", statistics.median(rates))

    def run_conc(c, rnd, prefix=None):
        out = [None] * c
        def one(i):
            ids = (prefix or []) + chat_ids(a.url, rnd * 100 + i)
            p = {"input_ids": ids, "stream": True,
                 "sampling_params": {"temperature": 0, "max_new_tokens": max(64, (ctx_len - len(ids)) // max(c, 1) - 64)}}
            t0, times, last = post(a.url, p)
            n = (last or {}).get("meta_info", {}).get("completion_tokens", len(times))
            out[i] = {"t0": t0, "first": times[0], "last": times[-1], "tokens": n,
                      "finish": ((last or {}).get("meta_info", {}).get("finish_reason") or {}).get("type")}
        ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        tot = sum(o["tokens"] for o in out)
        span = max(o["last"] for o in out) - min(o["first"] for o in out)
        per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out if o["tokens"] > 1 and o["last"] > o["first"]]
        return {"aggregate": round(tot / span, 2), "per_stream_mean": round(statistics.mean(per), 2),
                "per_stream_min": round(min(per), 2), "tokens": tot,
                "finish": sorted({o["finish"] for o in out if o["finish"]}), "ttft_max": round(max(o["first"] - o["t0"] for o in out), 2)}

    for c in list(a.conc) + list(a.high):
        with Power() as pw:
            rounds = [run_conc(c, r) for r in range(a.dec_reps)]
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
        res["decode"]["C1@32k"] = {"aggregate": med([x["aggregate"] for x in r]), "rounds": r}
        print(f"decode C1@32k: {statistics.median([x['aggregate'] for x in r]):.2f} tok/s", flush=True)
    res["prompt_sha256"] = PROMPT_HASHES; res["nonce_prefix"] = NONCE; res["raw_template"] = RAW_TEMPLATE
    res["status"] = "DONE"; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); save()
    if a.update_best:
        for k, g in res.get("gates", {}).items():
            if g["best_known"] is None or g["value"] > g["best_known"]:
                best_all.setdefault(a.card, {})[k] = g["value"]
        json.dump(best_all, open(bk_path, "w"), indent=1)
    print("SWEEP DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
