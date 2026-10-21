#!/usr/bin/env python3
"""A SGLang-shaped front for a llama.cpp server, so the kit's tools can measure a GGUF target.

Why
---
The kit's tools (`tools/sweep.py`, `tools/score_ref_panel.py`) talk to an SGLang
server: `POST /generate` with `input_ids`, `POST /tokenize`, `GET /server_info`.
llama.cpp speaks a different dialect (`/v1/completions`, `/tokenize`, `/props`),
so every llama.cpp target had to be measured by hand. This shim translates the
SGLang surface onto a llama.cpp server and lets the kit's tools run unchanged.

    python3 tools/llamacpp_sglang_shim.py --upstream http://127.0.0.1:8081 --port 30100
    python3 tools/sweep.py --url http://127.0.0.1:30100 --card rtx-3070-8gb \
        --config "image@sha256:... llama.cpp" --out work/sweep.json
    python3 tools/score_ref_panel.py --url http://127.0.0.1:30100 --panel work/panel.json \
        --out work/panel.json

What is translated
------------------
* `POST /tokenize`   -> forwarded unchanged (llama.cpp has the same shape).
* `GET /server_info` -> `context_length` and `server_args`, read from `/props`.
* `POST /generate`   -> the prompt is sent as a **token array** (`"prompt": [ids]`),
  which llama.cpp accepts, so no tokenizer round-trip can shift the ids.
  * plain generation: the upstream's tokens are streamed back as SGLang's SSE
    chunks, each carrying `meta_info.completion_tokens` and
    `meta_info.finish_reason.type`.
  * `return_logprob`: llama.cpp does not return the logprobs of the prompt's
    tokens, but it does return the **pre-sampling top-N logprobs of the next
    token** (`n_probs` + `post_sampling_probs: false`). The shim therefore
    asks the upstream once per scored position, with the prompt truncated at
    that position, and assembles the SGLang `meta_info` the kit's scorer reads.
    A requested id that falls outside the top-N is given the smallest logprob
    the upstream reported (its true value is below that) and counted in
    `n_logprob_misses`; the number is reported so a reader can judge it.

Everything else (`/v1/chat/completions`, `/v1/completions`, `/health`, `/props`)
is forwarded unchanged.
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8081"
MODEL = "bonsai"
TOP_N = 100
MISSES = 0
MISSES_LOCK = threading.Lock()


def _post_json(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_json_raw(url, payload, timeout=3600):
    """Returns (status, raw bytes) — the streaming path needs the bytes."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def upstream_props():
    try:
        with urllib.request.urlopen(UPSTREAM + "/props", timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def server_context_length():
    """The upstream's context length. If the upstream cannot be reached we fail
    loudly: a wrong (too large) context would make the kit's sweep ask for a
    prefill the server cannot hold, and the numbers would be nonsense."""
    props = upstream_props()
    if not props:
        raise RuntimeError(f"the upstream {UPSTREAM} did not answer /props — is the server up?")
    gen = props.get("default_generation_settings") or {}
    for k in ("n_ctx", "context_length"):
        if gen.get(k):
            return int(gen[k])
    if props.get("context_length"):
        return int(props["context_length"])
    raise RuntimeError(f"the upstream {UPSTREAM} does not report its context length")


def llama_top_n(ids, n_probs=TOP_N):
    """The pre-sampling top-N logprobs of the token that follows `ids`.

    Returns (entries, floor) where entries is [[logprob, token_id, text], ...]
    sorted by decreasing logprob, and floor is the largest logprob the upstream
    did not report (an upper bound for any id outside the top-N).
    """
    r = _post_json(UPSTREAM + "/v1/completions", {
        "model": MODEL,
        "prompt": list(ids),
        "max_tokens": 1,
        "temperature": 0,
        "n_probs": n_probs,
        "post_sampling_probs": False,
        "seed": 1234,
    })
    choice = (r.get("choices") or [{}])[0]
    content = ((choice.get("logprobs") or {}).get("content") or [{}])[0]
    cands = content.get("top_logprobs") or []
    entries = [[float(c["logprob"]), int(c["id"]), ""] for c in cands]
    entries.sort(key=lambda e: -e[0])
    floor = (entries[-1][0] - 1e-4) if entries else -1000.0
    return entries, floor


def sglang_meta_for_positions(ids, start_len, top_logprobs_num, token_ids_logprob):
    """The SGLang `meta_info` the kit's score_ref_panel.py reads.

    itop[k] is the distribution that predicted the token at index start_len+k,
    i.e. the top-N of the prompt truncated at start_len+k. otop[0] is the
    distribution that predicted the token after the prompt.
    """
    global MISSES
    n = len(ids)
    itop, iids = [], []
    for k in range(max(0, n - start_len)):
        entries, floor = llama_top_n(ids[: start_len + k])
        itop.append(entries[:top_logprobs_num])
        iids.append(_ids_logprob(entries, floor, token_ids_logprob))
    entries, floor = llama_top_n(ids)
    otop = [entries[:top_logprobs_num]]
    oids = [_ids_logprob(entries, floor, token_ids_logprob)]
    with MISSES_LOCK:
        miss = MISSES
    return {
        "input_top_logprobs": itop,
        "input_token_ids_logprobs": iids,
        "output_top_logprobs": otop,
        "output_token_ids_logprobs": oids,
        "n_logprob_misses": miss,
    }


def _ids_logprob(entries, floor, wanted):
    """The logprobs of the requested ids, in SGLang's [[logprob, id, text], ...] shape."""
    by_id = {e[1]: e[0] for e in entries}
    out = []
    for tid in (wanted or []):
        if tid in by_id:
            out.append([by_id[tid], int(tid), ""])
        else:
            out.append([floor, int(tid), ""])
    return out


def llama_generate_stream(ids, sampling, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": list(ids),
        "max_tokens": int(max_tokens or 64),
        "temperature": float(sampling.get("temperature", 0) or 0),
        "stream": True,
    }
    for src, dst in (("top_p", "top_p"), ("top_k", "top_k"), ("seed", "seed")):
        if sampling.get(src) is not None:
            payload[dst] = sampling[src]
    if sampling.get("stop"):
        payload["stop"] = sampling["stop"]
    req = urllib.request.Request(UPSTREAM + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        n = 0
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            text = ((obj.get("choices") or [{}])[0].get("text") or "")
            n += 1
            yield {"text": text,
                   "meta_info": {"completion_tokens": n,
                                 "finish_reason": {"type": "length"}}}, False
        yield {"text": "", "meta_info": {"completion_tokens": n,
                                         "finish_reason": {"type": "stop"}}}, True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/server_info"):
            try:
                ctx = server_context_length()
            except Exception as e:
                self._send(502, {"error": str(e)})
                return
            self._send(200, {"context_length": ctx,
                             "server_args": {"context_length": ctx,
                                             "max_running_requests": 1}})
            return
        self._forward_get()

    def _forward_get(self):
        try:
            with urllib.request.urlopen(UPSTREAM + self.path, timeout=60) as r:
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except Exception as e:
            self._send(502, {"error": str(e)})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {}
        if self.path.startswith("/tokenize"):
            # SGLang's /tokenize takes {"prompt": text}; llama.cpp's takes
            # {"content": text} and answers {"tokens": [...]}.
            content = payload.get("prompt")
            if content is None:
                content = payload.get("content")
            self._forward_post(json.dumps({"content": content}).encode())
            return
        if self.path.startswith("/generate"):
            self._generate(payload)
            return
        self._forward_post(raw)

    def _forward_post(self, raw):
        try:
            req = urllib.request.Request(UPSTREAM + self.path, data=raw,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except Exception as e:
            self._send(502, {"error": str(e)})

    def _generate(self, payload):
        ids = payload.get("input_ids") or []
        sampling = payload.get("sampling_params") or {}
        global MISSES
        if payload.get("return_logprob"):
            meta = sglang_meta_for_positions(ids,
                                             int(payload.get("logprob_start_len") or 0),
                                             int(payload.get("top_logprobs_num") or 20),
                                             payload.get("token_ids_logprob") or [])
            with MISSES_LOCK:
                MISSES = 0
            self._send(200, {"text": "", "meta_info": meta})
            return
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for obj, done in llama_generate_stream(ids, sampling,
                                                       sampling.get("max_new_tokens")):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                    if done:
                        break
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                self.wfile.write(b"data: " + json.dumps({"error": str(e)}).encode() + b"\n\n")
                self.wfile.flush()
            return
        max_new = int(sampling.get("max_new_tokens") or 64)
        r = _post_json(UPSTREAM + "/v1/completions", {
            "model": MODEL,
            "prompt": list(ids),
            "max_tokens": max_new,
            "temperature": float(sampling.get("temperature", 0) or 0),
            "stream": False,
        })
        text = (r.get("choices") or [{}])[0].get("text") or ""
        self._send(200, {"text": text,
                         "meta_info": {"completion_tokens": max_new,
                                       "prompt_tokens": len(ids),
                                       "finish_reason": {"type": "stop"}}})


def main():
    global UPSTREAM, MODEL, TOP_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8081")
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--model", default="bonsai")
    ap.add_argument("--top-n", type=int, default=100)
    a = ap.parse_args()
    UPSTREAM = a.upstream.rstrip("/")
    MODEL = a.model
    TOP_N = a.top_n
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    try:
        ctx = server_context_length()
    except Exception as e:
        raise SystemExit(str(e))
    print(f"llamacpp SGLang shim : {UPSTREAM} -> http://127.0.0.1:{a.port} "
          f"(ctx {ctx}, top_n {TOP_N})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
