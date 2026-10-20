#!/usr/bin/env python3
"""Tests for the evidence of PROTOCOL.md §6 (the cache-bust nonce and the prompt hashes) in
tools/sweep.py and tools/make_submission.py.  Stdlib only.  python3 tools/test_sweep_evidence.py"""
import hashlib, json, os, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "sweep.py")
MAKE = os.path.join(HERE, "make_submission.py")
TARGET = "qwen3.8-flash-next-offload"
URL = ""


class Stub(BaseHTTPRequestHandler):
    """Just enough of an SGLang server for tools/sweep.py: /server_info, /tokenize, /generate.
    It keeps the prompts it was asked to tokenize, so the test can check the recorded hashes."""
    protocol_version = "HTTP/1.1"
    prompts = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"context_length": 32768, "server_args": {"context_length": 32768}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        p = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            with Stub.lock:
                Stub.prompts.append(p.get("prompt", ""))
            self._json({"tokens": [1000 + i for i in range(16)]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            time.sleep(0.02)                        # time to first token
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


def sweep(tmp, best, name="sweep.json"):
    out = os.path.join(tmp, name)
    cmd = [sys.executable, SWEEP, "--url", URL, "--card", "test-card", "--config", "test",
           "--prefill", "1024", "--conc", "1", "--reps", "1", "--dec-reps", "2",
           "--best", best, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r, out


def best_known(tmp, gates):
    p = os.path.join(tmp, "best.json")
    json.dump({"test-card": gates}, open(p, "w"))
    return p


def check_evidence(d):
    """The recorded hashes must be the hashes of the prompts the server really got."""
    hashes = d.get("prompt_sha256")
    assert hashes, "no prompt hash recorded"
    assert all(isinstance(h, str) and len(h) == 64 for h in hashes), hashes
    nonce = d.get("nonce_prefix")
    assert nonce, "no cache-bust nonce recorded"
    assert Stub.prompts, "the server got no prompt"
    for p in Stub.prompts:
        assert nonce in p, "the prompt does not carry the recorded nonce: " + p
        assert hashlib.sha256(p.encode()).hexdigest() in hashes, "the hash of the prompt is not recorded"


def test_the_early_exit_keeps_the_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        Stub.prompts = []
        r, out = sweep(tmp, best_known(tmp, {"decode_C1": 1e9}))
        assert r.returncode == 3, r.stdout + r.stderr
        d = json.load(open(out))
        assert d["status"].startswith("EARLY_EXIT"), d
        check_evidence(d)
        assert len(d["prompt_sha256"]) == len(Stub.prompts), d["prompt_sha256"]


def test_a_full_run_keeps_the_same_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        Stub.prompts = []
        r, out = sweep(tmp, best_known(tmp, {}))
        assert r.returncode == 0, r.stdout + r.stderr
        d = json.load(open(out))
        assert d["status"] == "DONE", d
        check_evidence(d)
        assert len(d["prompt_sha256"]) == len(Stub.prompts), d["prompt_sha256"]


def test_the_submission_carries_the_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        Stub.prompts = []
        r, out = sweep(tmp, best_known(tmp, {"decode_C1": 1e9}))
        d = json.load(open(out))
        json.dump({"host": {"cpu": "EPYC 7443P"}, "gpus": []}, open(os.path.join(tmp, "probe.json"), "w"))
        json.dump({"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001},
                  open(os.path.join(tmp, "panel.json"), "w"))
        json.dump({"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64, "args": [], "env": {}},
                  open(os.path.join(tmp, "launch.json"), "w"))
        open(os.path.join(tmp, "memwatch.csv"), "w").write(
            "t,mem_available_gb,swap_used_gb,gpu_used_mib\n100,82.95,0.00,1200\n200,28.49,0.00,63932\n")
        sub = os.path.join(tmp, "sub.json")
        r = subprocess.run([sys.executable, MAKE, "--probe", os.path.join(tmp, "probe.json"),
                            "--sweep", out, "--panel", os.path.join(tmp, "panel.json"),
                            "--memwatch", os.path.join(tmp, "memwatch.csv"),
                            "--launch", os.path.join(tmp, "launch.json"), "--target", TARGET,
                            "--handle", "tester", "--out", sub], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        sp = json.load(open(sub))["speed"]
        assert sp["prompt_sha256"] == d["prompt_sha256"], sp
        assert sp["nonce_prefix"] == d["nonce_prefix"], sp


TESTS = [test_the_early_exit_keeps_the_evidence,
         test_a_full_run_keeps_the_same_evidence,
         test_the_submission_carries_the_evidence]

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
