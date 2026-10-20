#!/usr/bin/env python3
"""Long-context needle check against an OpenAI-compatible server: filler text of ~N tokens with one fact inserted at a
given depth, then a question about it (chat, thinking off, temperature 0, completion runs to EOS).

  python3 tools/needle.py --url http://127.0.0.1:30100 --tokens 131072 --depth 0.6
"""
import argparse
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _auth  # noqa: F401  (KIT_API_KEY bearer header)
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
    payload = {"model": "flashnext", "messages": [{"role": "user", "content": msg}], "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ans = out["choices"][0]["message"]["content"]
    res = {"prompt_tokens": out["usage"]["prompt_tokens"], "depth": a.depth, "code": code, "answer": ans,
           "pass": code in (ans or ""), "seconds": round(dt, 1)}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
