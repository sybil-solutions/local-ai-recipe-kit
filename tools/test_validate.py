#!/usr/bin/env python3
"""Tests for tools/validate.py (stdlib only).  python3 tools/test_validate.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VALIDATE = os.path.join(HERE, "validate.py")
TARGET = "qwen3.8-flash-next-offload"
CARD = "rtx-3090-24gb"


def submission(**kw):
    s = {"schema": "local-ai-recipe-kit/submission/v1", "submitted": "2026-10-20", "handle": "tester",
         "target": TARGET,
         "launch": {"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64,
                    "args": ["--model-path", "/models/qwen3.8-flash-next-exl3"], "env": {}},
         "probe": {"host": {"cpu": "EPYC 7443P"}, "gpus": []},
         "quality": {"panel": {"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001}},
         "speed": {"status": "DONE", "card": CARD, "baseline": CARD,
                   "prefill": {"8192": {"median": 2627, "n": 3}, "16384": {"median": 2785, "n": 3},
                               "32768": {"median": 2846, "n": 3}},
                   "decode": {"C1": {"aggregate": {"median": 62.7}}}},
         "needs_measured": {"samples": 120, "mem_available_start_gb": 82.95, "mem_available_min_gb": 28.49,
                            "host_ram_drop_peak_gb": 54.46, "swap_used_max_gb": 0.0, "gpu_used_peak_mib": 63932.0},
         "notes": ""}
    s.update(kw)
    return s


def validate(sub):
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, CARD))
        p = os.path.join(tmp, CARD, "sub.json")
        json.dump(sub, open(p, "w"), indent=1)
        r = subprocess.run([sys.executable, VALIDATE, p], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


def test_panel_outside_the_band_is_rejected():
    rc, out = validate(submission(quality={"panel": {"top1_agreement": 0.01, "mean_kl_top20": 40.0}}))
    assert rc == 1, "expected a failure, got:\n" + out
    assert "outside the target's band" in out, out
    assert "0.01" in out and "40" in out, out
    assert "0.987" in out and "0.0013" in out, out


def test_panel_inside_the_band_passes():
    rc, out = validate(submission())
    assert rc == 0, out
    assert out.startswith("ok"), out


def test_submission_from_before_the_rule_is_grandfathered():
    rc, out = validate(submission(submitted="2026-09-29",
                                  quality={"panel": {"top1_agreement": 0.97001, "mean_kl_top20": 0.0147}}))
    assert rc == 0, out
    assert "band not enforced for a submission from before the rule" in out, out
    assert "0.97001" in out and "0.0147" in out, out


def test_submissions_in_the_repo_still_pass():
    r = subprocess.run([sys.executable, VALIDATE, "submissions/**/*.json"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


TESTS = [test_panel_outside_the_band_is_rejected, test_panel_inside_the_band_passes,
         test_submission_from_before_the_rule_is_grandfathered, test_submissions_in_the_repo_still_pass]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print("ok   " + t.__name__)
        except AssertionError as e:
            bad += 1
            print("FAIL " + t.__name__ + ": " + str(e))
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)
