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
  --api sglang (default: /generate, /tokenize, /server_info) or openai (any server with /v1/chat/completions and no
  tokenizer endpoint, e.g. Strata: prompts are random-word text with a cache-bust nonce, token counts and the
  prompt/completion split come from the server's own streamed `usage`, prefill requests stop after the first token with
  max_tokens 1, decode requests are uncapped; --api-key-file sends a Bearer key without printing it).
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
KEY = MODEL = None
WPT = 0.75  # openai mode: prompt words per token, calibrated against the server's reported usage in main()
WORDS = ("river mountain lantern harbor orchard meadow copper violet thunder glacier canyon ember willow falcon "
         "compass quartz maple tide prairie beacon cobalt saffron marble lagoon summit basalt heron juniper").split()


def headers():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def get_json(url, path, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url + path, headers=headers()), timeout=timeout) as r:
        return json.loads(r.read())


def rand_text(n_tokens, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(max(1, int(n_tokens * WPT))))


def post_openai(url, payload, first_only=False, timeout=7200):
    """Stream /v1/chat/completions (thinking off, greedy). Returns (t0, times of the content chunks, last) with `last`
    in the SGLang meta_info shape plus the server's usage (prompt_tokens, cached_tokens) and timings."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": payload["text"]}], "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if first_only:
        body["max_tokens"] = 1
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers())
    t0 = time.perf_counter(); times = []; meta = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            if s == "[DONE]":
                break
            j = json.loads(s)
            for ch in j.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    times.append(time.perf_counter())
                if ch.get("finish_reason"):
                    meta["finish_reason"] = {"type": ch["finish_reason"]}
            if j.get("usage"):
                u = j["usage"]
                meta.update(prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"],
                            cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            if j.get("timings"):
                meta["server_timings"] = j["timings"]
    return t0, times, {"meta_info": meta}


def post(url, payload, first_only=False, timeout=7200):
    if API == "openai":
        return post_openai(url, payload, first_only, timeout)
    req = urllib.request.Request(url + "/generate", data=json.dumps(payload).encode(), headers=headers())
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
    if API == "openai":
        t = (f"[cache-bust nonce: {NONCE}-{i}]\nWrite a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} "
             f"(variant {i}).")
        PROMPT_HASHES.append(__import__("hashlib").sha256(t.encode()).hexdigest())
        return t
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


def main():
    global API, KEY, MODEL, WPT
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
    ap.add_argument("--update-best", action="store_true")
    ap.add_argument("--best", help="best-known JSON for early exit (default reference/best_known.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api", choices=["sglang", "openai"], default="sglang", help="server API (see module doc)")
    ap.add_argument("--api-key-file", help="openai API: file holding the Bearer key")
    ap.add_argument("--model", help="openai API: served model id (default: the first /v1/models entry)")
    a = ap.parse_args()
    API = a.api
    KEY = open(a.api_key_file).read().strip() if a.api_key_file else None
    MODEL = a.model
    bk_path = a.best or os.path.join(HERE, "..", "reference", "best_known.json")
    best_all = json.load(open(bk_path)) if os.path.exists(bk_path) else {}
    best = best_all.get(a.card, {})
    if API == "openai":
        MODEL = MODEL or get_json(a.url, "/v1/models")["data"][0]["id"]
        st = get_json(a.url, "/v1/status")
        info = {"context_length": st.get("cache_max_tokens"),
                "server_args": {"context_length": st.get("cache_max_tokens"), "max_total_num_tokens": st.get("cache_max_tokens"),
                                "max_running_requests": (st.get("concurrency") or {}).get("serving")}}
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
    if API == "openai":
        # no /tokenize here: a ~512-token warm-up and one ~4k-token calibration request (neither recorded) set the
        # words-per-token ratio from the server's reported prompt_tokens
        post(a.url, {"text": rand_text(512, seed)}, True)
        _, _, cal = post(a.url, {"text": rand_text(4000, seed + 1)}, True)
        WPT = 4000 * WPT / cal["meta_info"]["prompt_tokens"]
        print(f"calibration: {cal['meta_info']['prompt_tokens']} prompt tokens -> {WPT:.4f} words/token", flush=True)
    else:
        post(a.url, {"input_ids": rand_ids(512, seed), "sampling_params": {"temperature": 0}, "stream": True}, True)
    for n in a.prefill:
        if n + 64 > ctx_len:
            res["prefill"][str(n)] = {"skipped": f"> context {ctx_len}"}; continue
        rates, ptoks, cached, srv = [], [], [], []
        for rep in range(a.reps):
            if API == "openai":
                t0, times, last = post(a.url, {"text": rand_text(n, seed + 1000 * n + rep)}, True)
                m = last["meta_info"]; nt = m["prompt_tokens"]
                ptoks.append(nt); cached.append(m.get("cached_tokens", 0))
                srv.append((m.get("server_timings") or {}).get("prompt_per_second"))
            else:
                t0, times, _ = post(a.url, {"input_ids": rand_ids(n, seed + 1000 * n + rep),
                                            "sampling_params": {"temperature": 0}, "stream": True}, True)
                nt = n
            rates.append(round(nt / (times[0] - t0), 1))
            print(f"prefill {n}: {rates[-1]} tok/s", flush=True)
        res["prefill"][str(n)] = med(rates)
        if ptoks:
            res["prefill"][str(n)].update(prompt_tokens=ptoks, cached_tokens=cached, server_prompt_per_second=srv)
        save()
        gate(f"prefill_{n}", statistics.median(rates))

    def run_conc(c, rnd, prefix=None):
        out = [None] * c
        def one(i):
            if API == "openai":
                p = {"text": ((prefix + "\n\n") if prefix else "") + chat_ids(a.url, rnd * 100 + i)}
            else:
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
    pre = (rand_text(32768, seed + 99) if API == "openai" else rand_ids(32768, seed + 99)) if ctx_len > 40000 else None
    if pre:
        r = [run_conc(1, 90 + i, prefix=pre) for i in range(a.dec_reps)]
        res["decode"]["C1@32k"] = {"aggregate": med([x["aggregate"] for x in r]), "rounds": r}
        print(f"decode C1@32k: {statistics.median([x['aggregate'] for x in r]):.2f} tok/s", flush=True)
    res["prompt_sha256"] = PROMPT_HASHES; res["nonce_prefix"] = NONCE
    res["status"] = "DONE"; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); save()
    if a.update_best:
        for k, g in res.get("gates", {}).items():
            if g["best_known"] is None or g["value"] > g["best_known"]:
                best_all.setdefault(a.card, {})[k] = g["value"]
        json.dump(best_all, open(bk_path, "w"), indent=1)
    print("SWEEP DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
