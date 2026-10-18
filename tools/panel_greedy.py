#!/usr/bin/env python3
"""Free-running greedy check against the reference panel, for servers that cannot score teacher-forced logprobs
(no /generate, no token_ids_logprob, no OpenAI `logprobs`), where tools/score_ref_panel.py cannot run.

Each panel prompt is sent as one user chat message (thinking off, temperature 0, no max_tokens) and the completion is
compared with the exllamav3 reference completion in the panel: how many characters match before the first difference,
and whether the whole text is identical. This is NOT the panel's top-1 / KL metric and has no band; a greedy run
diverges at the first near-tie, so read it as "how far does this server follow the reference", nothing more.

  python3 tools/panel_greedy.py --url http://127.0.0.1:30100 --panel reference/qwen3.8-flash-next-exl3-ref-panel.json \
      [--api-key-file key] [--model id] --out panel_greedy.json
"""
import argparse
import json
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--api-key-file")
    ap.add_argument("--model")
    ap.add_argument("--out")
    a = ap.parse_args()
    hdr = {"Content-Type": "application/json"}
    if a.api_key_file:
        hdr["Authorization"] = "Bearer " + open(a.api_key_file).read().strip()
    model = a.model or json.loads(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/models", headers=hdr),
                                                         timeout=30).read())["data"][0]["id"]
    per = []
    for i, item in enumerate(json.load(open(a.panel))):
        body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}], "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                           headers=hdr), timeout=3600) as r:
            out = json.loads(r.read())
        got, ref = out["choices"][0]["message"]["content"] or "", item["completion"]
        n = next((k for k, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
        row = {"i": i, "identical": got == ref, "matching_prefix_chars": n, "reference_chars": len(ref),
               "prefix_fraction": round(n / max(1, len(ref)), 4), "completion_chars": len(got),
               "completion_tokens": out["usage"]["completion_tokens"],
               "reference_completion_tokens": len(item["tokens"]) - item["prompt_len"], "seconds": round(time.time() - t0, 1)}
        per.append(row)
        print(json.dumps(row), flush=True)
    res = {"kind": "greedy_text_match", "prompts": len(per), "identical": sum(r["identical"] for r in per),
           "mean_prefix_fraction": round(sum(r["prefix_fraction"] for r in per) / len(per), 4), "per_prompt": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
