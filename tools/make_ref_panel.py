#!/usr/bin/env python3
"""Build a reference panel for a llama.cpp target, in the format tools/score_ref_panel.py reads.

The kit ships one reference panel, `reference/qwen3.8-flash-next-exl3-ref-panel.json`: the
reference implementation's top-20 logprobs at each position of eight prompts, recorded
against the *shipped* packing so that the shipped one can be scored against it. A new
target needs its own panel, and for a llama.cpp target there is no exllamav3 reference
to record — the panel is recorded from the same model in the **higher-fidelity packing**
(PQ2_0, 2,125 bits/weight) and the *served* packing (PTQ1_0, 1,75 bits/weight) is scored
against it. Same weights, two packings: the same comparison the kit already makes.

    python3 tools/make_ref_panel.py --url http://127.0.0.1:30100 \
        --prompts work/prompts.json --completion-tokens 16 --top-n 20 --out work/panel.json

The prompt is the *reference* server's greedy continuation, exactly as in the shipped
panel, and the scored positions are the completion's tokens.

`--url` may point at the shim (tools/llamacpp_sglang_shim.py) or at an SGLang server:
both speak the same `/generate` and `/tokenize`.
"""

import argparse
import json
import urllib.request


def post(url, path, payload, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(url, text):
    return post(url, "/tokenize", {"prompt": text})["tokens"]


def generate_text(url, ids, n):
    r = post(url, "/generate", {"input_ids": ids,
                                "sampling_params": {"temperature": 0, "max_new_tokens": n},
                                "stream": False})
    return r.get("text") or ""


def top_at(url, ids, start_len, top_n):
    """The top-N logprobs of every position from `start_len` onward."""
    r = post(url, "/generate", {"input_ids": ids,
                                "return_logprob": True,
                                "logprob_start_len": start_len,
                                "top_logprobs_num": top_n,
                                "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
    meta = r["meta_info"]
    itop = meta.get("input_top_logprobs") or []
    otop = meta.get("output_top_logprobs") or []
    # itop[k] is the distribution that predicted the token at index start_len+k;
    # the distribution that predicted the last token is the output one.
    return list(itop[1:]) + list(otop[:1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--prompts", help="JSON file: a list of prompt strings")
    ap.add_argument("--prompt", action="append", help="one prompt (repeatable)")
    ap.add_argument("--completion-tokens", type=int, default=16)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = []
    if a.prompts:
        prompts = json.load(open(a.prompts))
        if isinstance(prompts, dict):
            prompts = prompts.get("prompts", [])
    if a.prompt:
        prompts += a.prompt
    if not prompts:
        raise SystemExit("no prompt: use --prompts or --prompt")

    panel = []
    for i, prompt in enumerate(prompts):
        ids_prompt = tokenize(a.url, prompt)
        completion = generate_text(a.url, ids_prompt, a.completion_tokens)
        ids_comp = tokenize(a.url, completion)
        tokens = list(ids_prompt) + list(ids_comp)
        P = len(ids_prompt)
        tops = top_at(a.url, tokens, P - 1, a.top_n)
        n = min(len(tops), len(tokens) - P)
        item = {
            "prompt": prompt,
            "prompt_len": P,
            "tokens": tokens,
            "completion": completion,
            "top20_ids": [[e[1] for e in tops[j][:a.top_n]] for j in range(n)],
            "top20_lp": [[e[0] for e in tops[j][:a.top_n]] for j in range(n)],
        }
        panel.append(item)
        print(f"[{i}] {n} positions scorés (prompt {P} tokens)", flush=True)
    json.dump(panel, open(a.out, "w"))
    total = sum(len(it["top20_ids"]) for it in panel)
    print(f"panneau écrit : {a.out} — {len(panel)} prompts, {total} positions", flush=True)


if __name__ == "__main__":
    main()
