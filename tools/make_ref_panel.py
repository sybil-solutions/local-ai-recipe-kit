#!/usr/bin/env python3
"""Build a reference panel for tools/score_ref_panel.py from a vLLM server running the reference configuration
(the stock kernels / KV format a target is compared against). Same format as
reference/qwen3.8-flash-next-exl3-ref-panel.json: per prompt the chat-template token ids, the greedy completion
(temperature 0, runs to its natural end), and the reference top-20 logprobs at every position from the last prompt
token to one past the completion's end (teacher-forced, so every position is scored on the same tokens).

  python3 tools/make_ref_panel.py --url http://127.0.0.1:8000 --model deepseek-v4.1-flash-1017 \
      --prompts-from reference/qwen3.8-flash-next-exl3-ref-panel.json --out reference/<name>-ref-panel.json
"""
import argparse
import json
import urllib.request


def post(url, path, payload, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tid(key):
    return int(key.split(":", 1)[1]) if isinstance(key, str) and key.startswith("token_id:") else int(key)


def top_k(entry, k):
    """vLLM prompt_logprobs entry {id: {logprob, rank}} or top_logprobs entry {"token_id:N": lp} -> (ids, lps) by rank."""
    items = [(tid(t), v["logprob"] if isinstance(v, dict) else v) for t, v in entry.items()]
    items.sort(key=lambda e: -e[1])
    return [i for i, _ in items[:k]], [lp for _, lp in items[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-from", required=True, help="an existing panel JSON (its `prompt` fields are reused)")
    ap.add_argument("--chat-kwargs", default='{"thinking": false}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kw = json.loads(a.chat_kwargs)
    models = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"]
    ctx = next(m for m in models if m["id"] == a.model)["max_model_len"]
    panel = []
    for i, src in enumerate(json.load(open(a.prompts_from))):
        ids = post(a.url, "/tokenize", {"model": a.model, "messages": [{"role": "user", "content": src["prompt"]}],
                                        "add_generation_prompt": True, "chat_template_kwargs": kw})["tokens"]
        # vLLM's completions default is 16 tokens; the remaining context lets the answer run to its natural end
        g = post(a.url, "/v1/completions", {"model": a.model, "prompt": ids, "temperature": 0, "logprobs": 1,
                                            "max_tokens": ctx - len(ids), "return_tokens_as_token_ids": True})["choices"][0]
        comp = [tid(t) for t in g["logprobs"]["tokens"]]
        toks = ids + comp
        P = len(ids)
        r = post(a.url, "/v1/completions", {"model": a.model, "prompt": toks, "temperature": 0, "max_tokens": 1,
                                            "prompt_logprobs": 20, "logprobs": 20,
                                            "return_tokens_as_token_ids": True})["choices"][0]
        # prompt_logprobs[k] is the distribution that predicted token k; position p (predicting p+1) is entry p+1,
        # and the last position (predicting past the end) is the generated token's distribution.
        dists = list(r["prompt_logprobs"][P:]) + [r["logprobs"]["top_logprobs"][0]]
        ids20, lps20 = zip(*(top_k(d, 20) for d in dists))
        panel.append({"prompt": src["prompt"], "prompt_len": P, "tokens": toks, "completion": g["text"],
                      "finish_reason": g.get("finish_reason"), "top20_ids": list(ids20), "top20_lp": list(lps20)})
        print(f"[{i}] prompt {P} + completion {len(comp)} tokens ({g.get('finish_reason')})", flush=True)
    json.dump(panel, open(a.out, "w"))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
