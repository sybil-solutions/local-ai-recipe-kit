#!/usr/bin/env python3
"""Score an SGLang server against the exllamav3 reference panel (reference/qwen3.8-flash-next-exl3-ref-panel.json):
teacher-force each panel sequence (`tokens` = prompt + reference greedy completion) and compare the server's
distributions at positions prompt_len-1 .. end with the reference top-20 logprobs.

Metrics: top-1 agreement (server argmax == reference argmax), mean KL(ref || server) on the reference top-20 support
(both renormalised over those 20 ids; server logprobs for the ids come from `token_ids_logprob`, exact), plus the
fraction of positions where the reference greedy token is the server argmax.

  python3 tools/score_ref_panel.py --url http://127.0.0.1:30100 --panel ref_panel.json --out score.json

--api vllm scores a vLLM server: one /v1/completions call per sequence with prompt_logprobs=K (the server's
--max-logprobs must be >= K). vLLM returns the server's own top-K plus the actual token, not arbitrary ids, so a
reference id outside the server's top-K gets the K-th logprob as an upper bound; `missing_ref_ids` counts them
(with K = 128 they are rare and carry next to no reference mass).
"""
import argparse
import json
import math
import urllib.request


def post(url, path, payload, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vllm_dists(a, toks, P):
    """(top list [(lp, id)], {id: lp}) per scored position, from vLLM prompt_logprobs + the generated token."""
    r = post(a.url, "/v1/completions", {"model": a.model, "prompt": toks, "temperature": 0, "max_tokens": 1,
                                        "prompt_logprobs": a.k, "logprobs": a.k,
                                        "return_tokens_as_token_ids": True})["choices"][0]
    out = []
    for d in list(r["prompt_logprobs"][P:]) + [r["logprobs"]["top_logprobs"][0]]:
        lp = {int(str(t).split(":")[-1]): (v["logprob"] if isinstance(v, dict) else v) for t, v in d.items()}
        out.append(([(v, t) for t, v in lp.items()], lp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out")
    ap.add_argument("--api", choices=["sglang", "vllm"], default="sglang")
    ap.add_argument("--model", help="served model name (--api vllm)")
    ap.add_argument("--k", type=int, default=128, help="server top-K logprobs per position (--api vllm)")
    a = ap.parse_args()
    panel = json.load(open(a.panel))
    tot = {"pos": 0, "top1": 0, "kl": 0.0, "greedy": 0, "missing": 0}
    per = []
    for i, item in enumerate(panel):
        toks, P = item["tokens"], item["prompt_len"]
        ref_ids, ref_lp = item["top20_ids"], item["top20_lp"]
        support = sorted({t for row in ref_ids for t in row})
        if a.api == "vllm":
            dv = vllm_dists(a, toks, P)
            dists_top, dists_ids = [x[0] for x in dv], [x[1] for x in dv]
        else:
            dists_top = dists_ids = None
        # positions P-1 .. len-1 predict tokens P .. len (the last one predicts past the end: the generated token)
        r = None if dists_top else post(a.url, "/generate", {"input_ids": toks, "return_logprob": True, "logprob_start_len": P - 1,
                                      "top_logprobs_num": 20, "token_ids_logprob": support,
                                      "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
        if r is not None:
            m = r["meta_info"]
            itop, iids = m["input_top_logprobs"], m["input_token_ids_logprobs"]
            otop, oids = m["output_top_logprobs"], m["output_token_ids_logprobs"]
        # SGLang: entry j of the input lists belongs to token (P-1+j) and holds the distribution that predicted it,
        # i.e. logits at position P-2+j; entry 0 is the prompt's first scored token. Shift by one: our position
        # p = P-1+j (predicting token p+1) is input entry j+1, and the last position is the first output entry.
        if r is not None:
            dists_top = list(itop[1:]) + [otop[0]]
            dists_ids = [{e[1]: e[0] for e in row} for row in list(iids[1:]) + [oids[0]]]
        n = len(ref_ids)
        assert len(dists_top) >= n, (len(dists_top), n)
        s = {"pos": 0, "top1": 0, "kl": 0.0, "greedy": 0, "missing": 0}
        for j in range(n):
            srv_top = dists_top[j]
            srv_arg = max(srv_top, key=lambda e: e[0])[1]
            ref_arg = ref_ids[j][0]
            lp_by_id = dists_ids[j]
            pr = [math.exp(x) for x in ref_lp[j]]
            zr = sum(pr)
            floor = min(lp_by_id.values()) if a.api == "vllm" else None  # bound for ids outside the server's top-K
            s["missing"] += sum(1 for t in ref_ids[j] if lp_by_id.get(t) is None)
            qs = [math.exp(lp_by_id[t]) if lp_by_id.get(t) is not None else (math.exp(floor) if floor is not None else 1e-30)
                  for t in ref_ids[j]]
            zq = sum(qs)
            kl = sum((p / zr) * (math.log(p / zr) - math.log(max(q / zq, 1e-30))) for p, q in zip(pr, qs))
            s["pos"] += 1
            s["top1"] += int(srv_arg == ref_arg)
            s["kl"] += kl
            if P + j < len(toks):
                s["greedy"] += int(srv_arg == toks[P + j])
        per.append({"i": i, "positions": s["pos"], "top1": s["top1"] / s["pos"], "kl": s["kl"] / s["pos"],
                    "greedy_match": s["greedy"] / max(1, s["pos"] - 1)})
        for k in tot:
            tot[k] += s[k]
        print(f"[{i}] {s['pos']} pos  top1 {s['top1'] / s['pos']:.4f}  KL {s['kl'] / s['pos']:.5f}", flush=True)
    res = {"positions": tot["pos"], "top1_agreement": tot["top1"] / tot["pos"], "mean_kl_top20": tot["kl"] / tot["pos"],
           "greedy_match": tot["greedy"] / max(1, tot["pos"] - len(panel)), "missing_ref_ids": tot["missing"],
           "per_prompt": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
