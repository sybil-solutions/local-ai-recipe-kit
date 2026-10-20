#!/usr/bin/env python3
"""Tests for the card of a submission (tools/validate.py): the card the sweep ran on must be the
directory the submission lives in.  Stdlib only.  python3 tools/test_validate_card.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATE = os.path.join(HERE, "validate.py")
TARGET = "qwen3.8-flash-next-offload"
CARD = "rtx-3090-24gb"


def submission(card=CARD, **kw):
    s = {"schema": "local-ai-recipe-kit/submission/v1", "submitted": "2026-10-20", "handle": "tester",
         "target": TARGET,
         "launch": {"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64,
                    "args": ["--model-path", "/models/qwen3.8-flash-next-exl3"], "env": {}},
         "probe": {"host": {"cpu": "EPYC 7443P"}, "gpus": []},
         "quality": {"panel": {"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001}},
         "speed": {"status": "DONE", "card": card, "baseline": card, "best_known_before": {"prefill_8192": 2619},
                   "prefill": {"8192": {"median": 2627, "n": 3}, "16384": {"median": 2785, "n": 3},
                               "32768": {"median": 2846, "n": 3}},
                   "decode": {"C1": {"aggregate": {"median": 62.7}}}},
         "needs_measured": {"samples": 120, "mem_available_start_gb": 82.95, "mem_available_min_gb": 28.49,
                            "host_ram_drop_peak_gb": 54.46, "swap_used_max_gb": 0.0, "gpu_used_peak_mib": 63932.0},
         "notes": ""}
    s.update(kw)
    return s


def validate(tmp, s):
    d = os.path.join(tmp, CARD)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "2026-10-20-tester.json")
    json.dump(s, open(p, "w"))
    return subprocess.run([sys.executable, VALIDATE, p], capture_output=True, text=True)


def test_validate_checks_the_card_against_the_directory():
    with tempfile.TemporaryDirectory() as tmp:
        r = validate(tmp, submission(card="rtx3090"))
        assert r.returncode == 1, r.stdout
        assert "speed.card" in r.stdout and "rtx-3090-24gb" in r.stdout, r.stdout
        r = validate(tmp, submission())
        assert r.returncode == 0, r.stdout + r.stderr


TESTS = [test_validate_checks_the_card_against_the_directory]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print("ok   " + t.__name__)
        except Exception as e:
            bad += 1
            print("FAIL " + t.__name__ + ": " + str(e))
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)
