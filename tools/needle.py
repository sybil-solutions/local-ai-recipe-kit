#!/usr/bin/env python3
"""Long-context needle check against an OpenAI-compatible server: filler text of ~N tokens with one fact inserted at a
given depth, then a question about it (chat, thinking off, temperature 0, completion runs to EOS).

  python3 tools/needle.py --url http://127.0.0.1:30100 --tokens 131072 --depth 0.6
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("river mountain lantern harbor orchard meadow copper violet thunder glacier canyon ember willow falcon "
         "compass quartz maple tide prairie beacon cobalt saffron marble lagoon summit basalt heron juniper").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds to wait for the answer (slow long-context prefill)")
    ap.add_argument("--raw-template", help="send the prompt through /v1/completions rendered with this template "
                    "('{prompt}' marks the user message), for chat templates without an enable_thinking switch, e.g. "
                    "GLM-5.3: '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think></think>'")
    a = ap.parse_args()
    rng = random.Random(7)
    # ~1.3 tokens per word for these words; sentences of 12 words
    n_sent = int(a.tokens / 1.3 / 13)
    sents = [" ".join(rng.choice(WORDS) for _ in range(12)).capitalize() + "." for _ in range(n_sent)]
    code = f"{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}"
    needle = f"Important: the vault access code for the Kestrel project is {code}."
    sents.insert(int(len(sents) * a.depth), needle)
    doc = " ".join(sents)
    msg = doc + "\n\nQuestion: What is the vault access code for the Kestrel project? Reply with the code only."
    def post(path, payload):
        req = urllib.request.Request(a.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            return json.loads(r.read())

    t0 = time.time()
    if a.raw_template:
        # /v1/completions defaults to a 16-token limit on some servers, so the answer gets the rest of the context:
        # a 1-token call returns the prompt length (its prefix is then cached), then the real call runs to EOS.
        prompt = a.raw_template.replace("{prompt}", msg)
        models = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=60).read())["data"][0]
        ctx = int(models.get("max_model_len") or models.get("context_length"))
        n = post("/v1/completions", {"model": "x", "prompt": prompt, "temperature": 0, "max_tokens": 1})
        n = int(n["usage"]["prompt_tokens"])
        out = post("/v1/completions", {"model": "x", "prompt": prompt, "temperature": 0, "max_tokens": ctx - n})
        ans = out["choices"][0]["text"]
    else:
        out = post("/v1/chat/completions", {"model": "flashnext", "messages": [{"role": "user", "content": msg}],
                                            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        ans = out["choices"][0]["message"]["content"]
    dt = time.time() - t0
    res = {"prompt_tokens": out["usage"]["prompt_tokens"], "depth": a.depth, "code": code, "answer": ans,
           "pass": code in (ans or ""), "seconds": round(dt, 1)}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
