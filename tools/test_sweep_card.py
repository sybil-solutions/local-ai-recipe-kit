#!/usr/bin/env python3
"""Tests for the sweep's baseline (tools/sweep.py, tools/make_submission.py, tools/validate.py).
Stdlib only.  python3 tools/test_sweep_card.py"""
import json, os, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SWEEP = os.path.join(HERE, "sweep.py")
MAKE = os.path.join(HERE, "make_submission.py")
TARGET = "qwen3.8-flash-next-offload"
URL = ""


class Stub(BaseHTTPRequestHandler):
    """Just enough of an SGLang server for tools/sweep.py: /server_info, /tokenize, /generate."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"context_length": 32768, "server_args": {"context_length": 32768,
                                                                    "max_running_requests": 4,
                                                                    "max_total_num_tokens": 210000,
                                                                    "chunked_prefill_size": 8192,
                                                                    "kv_cache_dtype": "fp8_e4m3",
                                                                    "mem_fraction_static": 0.88,
                                                                    "max_mamba_cache_size": 16}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        p = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            body = json.dumps({"tokens": [1000 + i for i in range(16)]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            time.sleep(0.05)                       # time to first token
            for i in range(4):
                body = {"text": "tok%d" % i, "meta_info": {"completion_tokens": i + 1,
                                                           "finish_reason": {"type": "stop" if i == 3 else None}}}
                chunk = ("data: " + json.dumps(body) + "\n\n").encode()
                self.wfile.write(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                self.wfile.flush()
                time.sleep(0.005)
            chunk = b"data: [DONE]\n\n"
            self.wfile.write(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass                                   # the client stopped reading (warm-up)
        self.close_connection = True


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def sweep(tmp, card, best=None, name="sweep.json"):
    cmd = [sys.executable, SWEEP, "--url", URL, "--card", card, "--config", "test", "--prefill", "1024",
           "--conc", "1", "--reps", "1", "--dec-reps", "1", "--out", os.path.join(tmp, name)]
    if best:
        cmd += ["--best", best]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r, os.path.join(tmp, name)


def submission(card="rtx-3090-24gb", **kw):
    s = {"schema": "local-ai-recipe-kit/submission/v1", "submitted": "2026-10-20", "handle": "tester",
         "target": TARGET,
         "launch": {"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64,
                    "args": ["--model-path", "/models/qwen3.8-flash-next-exl3"], "env": {}},
         "probe": {"host": {"cpu": "EPYC 7443P"}, "gpus": []},
         "quality": {"panel": {"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001}},
         "speed": {"status": "DONE", "card": card, "baseline": card, "best_known_before": {"prefill_8192": 2619},
                   "prefill": {"8192": {"median": 2627}, "16384": {"median": 2785}, "32768": {"median": 2846}},
                   "decode": {"C1": {"aggregate": {"median": 62.7}}}},
         "needs_measured": {"samples": 120, "mem_available_start_gb": 82.95, "mem_available_min_gb": 28.49,
                            "host_ram_drop_peak_gb": 54.46, "swap_used_max_gb": 0.0, "gpu_used_peak_mib": 63932.0},
         "notes": ""}
    s.update(kw)
    return s


def test_help_lists_the_real_card_keys():
    r = subprocess.run([sys.executable, SWEEP, "--help"], capture_output=True, text=True)
    assert "rtx-3090-24gb" in r.stdout, r.stdout
    assert "intel-arc-pro-b70-32gb" in r.stdout, r.stdout
    assert "rtx3090" not in r.stdout, r.stdout


def test_unknown_card_is_never_a_pass():
    with tempfile.TemporaryDirectory() as tmp:
        r, out = sweep(tmp, "rtx3090")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "WARNING" in r.stderr, r.stderr
        assert "rtx-3090-24gb" in r.stderr, r.stderr
        d = json.load(open(out))
        assert d["baseline"] == "MISSING", d
        assert d["gates"], d
        for k, g in d["gates"].items():
            assert g["pass"] is not True, (k, g)


def test_known_card_still_exits_early():
    with tempfile.TemporaryDirectory() as tmp:
        best = os.path.join(tmp, "best.json")
        json.dump({"test-card": {"prefill_1024": 1000000}}, open(best, "w"))
        r, out = sweep(tmp, "test-card", best=best)
        assert r.returncode == 3, r.stdout + r.stderr
        assert "EARLY_EXIT" in r.stdout, r.stdout
        d = json.load(open(out))
        assert d["status"].startswith("EARLY_EXIT"), d
        assert d["baseline"] == "test-card", d
        assert d["gates"]["prefill_1024"]["pass"] is False, d


def test_make_submission_copies_the_card():
    with tempfile.TemporaryDirectory() as tmp:
        json.dump({"host": {"cpu": "EPYC 7443P"}, "gpus": []}, open(os.path.join(tmp, "probe.json"), "w"))
        json.dump({"status": "DONE", "card": "rtx-3090-24gb", "baseline": "rtx-3090-24gb",
                   "best_known_before": {"prefill_8192": 2619}, "config": "test",
                   "prefill": {"8192": {"median": 2627}}, "decode": {"C1": {"aggregate": {"median": 62.7}}}},
                  open(os.path.join(tmp, "sweep.json"), "w"))
        json.dump({"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001},
                  open(os.path.join(tmp, "panel.json"), "w"))
        json.dump({"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64, "args": [], "env": {}},
                  open(os.path.join(tmp, "launch.json"), "w"))
        open(os.path.join(tmp, "memwatch.csv"), "w").write(
            "t,mem_available_gb,swap_used_gb,gpu_used_mib\n100,82.95,0.00,1200\n200,28.49,0.00,63932\n")
        out = os.path.join(tmp, "out", "sub.json")
        r = subprocess.run([sys.executable, MAKE, "--probe", os.path.join(tmp, "probe.json"),
                            "--sweep", os.path.join(tmp, "sweep.json"), "--panel", os.path.join(tmp, "panel.json"),
                            "--memwatch", os.path.join(tmp, "memwatch.csv"),
                            "--launch", os.path.join(tmp, "launch.json"), "--target", TARGET,
                            "--handle", "tester", "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        sp = json.load(open(out))["speed"]
        assert sp["card"] == "rtx-3090-24gb", sp
        assert sp["best_known_before"] == {"prefill_8192": 2619}, sp


# the validator half of this check (the card of the sweep must match the directory of the
# submission) lives in the validator branch: tools/test_validate_card.py


TESTS = [test_help_lists_the_real_card_keys, test_unknown_card_is_never_a_pass,
         test_known_card_still_exits_early, test_make_submission_copies_the_card]

if __name__ == "__main__":
    srv, URL = serve()
    bad = 0
    for t in TESTS:
        try:
            t()
            print("ok   " + t.__name__)
        except Exception as e:
            bad += 1
            print("FAIL " + t.__name__ + ": " + str(e))
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    srv.shutdown()
    sys.exit(1 if bad else 0)
