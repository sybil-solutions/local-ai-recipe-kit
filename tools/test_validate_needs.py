#!/usr/bin/env python3
"""Tests for the measured memory needs of a submission (tools/validate.py, tools/make_submission.py).
Stdlib only.  python3 tools/test_validate_needs.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VALIDATE = os.path.join(HERE, "validate.py")
MAKE = os.path.join(HERE, "make_submission.py")
TARGET = "qwen3.8-flash-next-offload"
CARD = "rtx-3090-24gb"
MEASURED = {"samples": 120, "mem_available_start_gb": 82.95, "mem_available_min_gb": 28.49,
            "host_ram_drop_peak_gb": 54.46, "swap_used_max_gb": 0.0, "gpu_used_peak_mib": 63932.0}


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
         "needs_measured": dict(MEASURED), "notes": ""}
    s.update(kw)
    return s


def validate(sub):
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, CARD))
        p = os.path.join(tmp, CARD, "sub.json")
        json.dump(sub, open(p, "w"), indent=1)
        r = subprocess.run([sys.executable, VALIDATE, p], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


def test_missing_needs_measured_is_rejected():
    rc, out = validate(submission(needs_measured=None))
    assert rc == 1, out
    assert "needs_measured" in out and "memwatch" in out, out


def test_needs_measured_without_numbers_is_rejected():
    rc, out = validate(submission(needs_measured={"samples": 0, "host_ram_drop_peak_gb": 0.0,
                                                   "gpu_used_peak_mib": None}))
    assert rc == 1, out
    assert "host_ram_drop_peak_gb" in out and "gpu_used_peak_mib" in out, out
    assert "must be > 0" in out, out


def test_measured_needs_pass():
    rc, out = validate(submission())
    assert rc == 0, out
    assert out.startswith("ok"), out


def test_submission_from_before_the_rule_is_grandfathered():
    rc, out = validate(submission(submitted="2026-09-29", needs_measured=None))
    assert rc == 0, out
    assert "needs_measured not enforced for a submission from before the rule" in out, out


def test_submissions_in_the_repo_still_pass():
    r = subprocess.run([sys.executable, VALIDATE, "submissions/**/*.json"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def artifacts(tmp):
    json.dump({"host": {"cpu": "EPYC 7443P"}, "gpus": []}, open(os.path.join(tmp, "probe.json"), "w"))
    json.dump({"status": "DONE", "config": "test", "card": CARD, "baseline": CARD,
               "best_known_before": {"prefill_8192": 2619},
               "prefill": {"8192": {"median": 2627, "n": 3}, "16384": {"median": 2785, "n": 3},
                           "32768": {"median": 2846, "n": 3}},
               "decode": {"C1": {"aggregate": {"median": 62.7}, "rounds": []}}},
              open(os.path.join(tmp, "sweep.json"), "w"))
    json.dump({"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001},
              open(os.path.join(tmp, "panel.json"), "w"))
    json.dump({"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64, "args": [], "env": {}},
              open(os.path.join(tmp, "launch.json"), "w"))
    open(os.path.join(tmp, "memwatch.csv"), "w").write(
        "t,mem_available_gb,swap_used_gb,gpu_used_mib\n100,82.95,0.00,1200\n200,28.49,0.00,63932\n")


def test_make_submission_requires_the_memwatch_csv():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts(tmp)
        cmd = [sys.executable, MAKE, "--probe", os.path.join(tmp, "probe.json"),
               "--sweep", os.path.join(tmp, "sweep.json"), "--panel", os.path.join(tmp, "panel.json"),
               "--launch", os.path.join(tmp, "launch.json"), "--target", TARGET, "--handle", "tester",
               "--out", os.path.join(tmp, "out", "sub.json")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode != 0, r.stdout
        assert "--memwatch" in r.stderr, r.stderr


def test_make_submission_copies_the_measured_needs():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts(tmp)
        out = os.path.join(tmp, "out", "sub.json")
        cmd = [sys.executable, MAKE, "--probe", os.path.join(tmp, "probe.json"),
               "--sweep", os.path.join(tmp, "sweep.json"), "--panel", os.path.join(tmp, "panel.json"),
               "--memwatch", os.path.join(tmp, "memwatch.csv"), "--launch", os.path.join(tmp, "launch.json"),
               "--target", TARGET, "--handle", "tester", "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        nm = json.load(open(out))["needs_measured"]
        assert nm["host_ram_drop_peak_gb"] == 54.46, nm
        assert nm["gpu_used_peak_mib"] == 63932.0, nm


TESTS = [test_missing_needs_measured_is_rejected, test_needs_measured_without_numbers_is_rejected,
         test_measured_needs_pass, test_submission_from_before_the_rule_is_grandfathered,
         test_submissions_in_the_repo_still_pass, test_make_submission_requires_the_memwatch_csv,
         test_make_submission_copies_the_measured_needs]

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
