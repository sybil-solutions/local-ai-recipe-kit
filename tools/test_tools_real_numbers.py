#!/usr/bin/env python3
"""Tests for the tools that have to report what they measured (tools/nvme_rand.py, tools/memwatch.sh,
tools/make_submission.py, tools/to_localmaxxing.py).  Stdlib only.  python3 tools/test_tools_real_numbers.py"""
import csv, json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
NVME = os.path.join(HERE, "nvme_rand.py")
MEMWATCH = os.path.join(HERE, "memwatch.sh")
MAKE = os.path.join(HERE, "make_submission.py")
LM = os.path.join(HERE, "to_localmaxxing.py")
TARGET = "qwen3.8-flash-next-offload"

# a stand-in for nvidia-smi: two cards, 1200 MiB and 3800 MiB in use
FAKE_NVIDIA_SMI = """#!/bin/sh
case "$*" in
  *index*) printf '0\\n1\\n' ;;
  *memory.used*) printf '1200\\n3800\\n' ;;
  *) : ;;
esac
"""

# the upstream memwatch.sh calls bc (not installed everywhere): a stand-in for the two expressions it uses
FAKE_BC = """#!/bin/sh
in=$(cat)
expr=$(echo "$in" | sed 's/^scale=[0-9]*;//')
scale=$(echo "$in" | sed -n 's/^scale=\\([0-9]*\\);.*/\\1/p')
awk -v e="$expr" -v s="${scale:-0}" 'BEGIN{printf "%.*f\\n", s, (e+0)}'
"""


class Stub(BaseHTTPRequestHandler):
    """Answers /health only after a few polls: the memwatch marks the ready row then."""
    protocol_version = "HTTP/1.1"
    ready_after = 2
    polls = 0
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_GET(self):
        with Stub.lock:
            Stub.polls += 1
            ok = Stub.polls > Stub.ready_after
        body = b"OK" if ok else b"not ready"
        self.send_response(200 if ok else 503)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def memwatch(tmp, url, seconds=6):
    """Run memwatch.sh with the fake nvidia-smi and the stub server; return its rows."""
    d = os.path.join(tmp, "bin")
    os.makedirs(d, exist_ok=True)
    for name, body in (("nvidia-smi", FAKE_NVIDIA_SMI), ("bc", FAKE_BC)):
        fake = os.path.join(d, name)
        open(fake, "w").write(body)
        os.chmod(fake, 0o755)
    path = os.path.join(tmp, "memwatch.csv")
    env = dict(os.environ, PATH=d + os.pathsep + os.environ["PATH"])
    Stub.polls = 0                             # the server is not ready yet
    r = subprocess.run(["timeout", str(seconds), "bash", MEMWATCH, path, url], capture_output=True, text=True, env=env)
    return r, list(csv.DictReader(open(path)))


def artifacts(tmp):
    """The other inputs of make_submission.py: a probe with two GPUs, a sweep, a panel, a launch."""
    json.dump({"host": {"cpu": "EPYC 7443P"},
               "gpus": [{"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24},
                        {"vendor": "nvidia", "index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]},
              open(os.path.join(tmp, "probe.json"), "w"))
    json.dump({"status": "DONE", "config": "test", "prefill": {"8192": {"median": 2627}},
               "decode": {"C1": {"aggregate": {"median": 62.7},
                                 "gpu_power_w_mean": 48.2,
                                 "rounds": [{"aggregate": 62.7, "tokens": 60, "ttft_max": 0.02}]}}},
              open(os.path.join(tmp, "sweep.json"), "w"))
    json.dump({"top1_agreement": 0.9913, "mean_kl_top20": 0.00092, "positions": 3001},
              open(os.path.join(tmp, "panel.json"), "w"))
    json.dump({"image": "ghcr.io/0xsero/sglang-exl3-flashnext@sha256:" + "a" * 64, "args": [], "env": {}},
              open(os.path.join(tmp, "launch.json"), "w"))


def submission(tmp, csv_path):
    artifacts(tmp)
    out = os.path.join(tmp, "sub.json")
    r = subprocess.run([sys.executable, MAKE, "--probe", os.path.join(tmp, "probe.json"),
                        "--sweep", os.path.join(tmp, "sweep.json"), "--panel", os.path.join(tmp, "panel.json"),
                        "--memwatch", csv_path, "--launch", os.path.join(tmp, "launch.json"),
                        "--target", TARGET, "--handle", "tester", "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def test_nvme_rand_refuses_a_file_that_is_too_small():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "small.bin")
        open(p, "wb").write(b"\0" * 4096)
        r = subprocess.run([sys.executable, NVME, p, "65536"], capture_output=True, text=True)
        assert r.returncode != 0, "the tool exited 0 although it measured nothing"
        assert "65536" in (r.stdout + r.stderr), r.stdout + r.stderr
        assert "GBps" not in r.stdout, "the tool published a number it did not measure: " + r.stdout


def test_nvme_rand_measures_a_file_that_is_big_enough():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "big.bin")
        with open(p, "wb") as f:
            f.truncate(64 << 20)
        try:
            os.close(os.open(p, os.O_RDONLY | os.O_DIRECT))
        except OSError:
            print("skip: this filesystem does not support O_DIRECT")
            return
        r = subprocess.run([sys.executable, NVME, p, "4096"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        d = json.loads(r.stdout.strip().splitlines()[-1])
        for c in d["cells"]:
            assert c["GBps"] > 0, c


def test_memwatch_marks_the_ready_row_and_every_gpu():
    with tempfile.TemporaryDirectory() as tmp:
        srv, url = serve()
        try:
            r, rows = memwatch(tmp, url)
        finally:
            srv.shutdown()
        assert rows, "the memwatch wrote no rows: " + r.stderr
        head = list(rows[0].keys())
        assert "phase" in head, head
        assert "gpu0_used_mib" in head and "gpu1_used_mib" in head, head
        ready = [x for x in rows if x["phase"] == "ready"]
        assert len(ready) == 1, [x["phase"] for x in rows]
        i = rows.index(ready[0])
        assert [x["phase"] for x in rows[:i]] == ["load"] * i, [x["phase"] for x in rows]
        assert ready[0]["gpu0_used_mib"] == "1200" and ready[0]["gpu1_used_mib"] == "3800", ready[0]
        assert ready[0]["gpu_used_mib"] == "5000", ready[0]


def test_the_submission_carries_the_drop_at_ready():
    with tempfile.TemporaryDirectory() as tmp:
        srv, url = serve()
        try:
            r, rows = memwatch(tmp, url)
        finally:
            srv.shutdown()
        ready = next(x for x in rows if x["phase"] == "ready")
        base = [x for x in rows if x["phase"] == "load"][-1]
        path = os.path.join(tmp, "memwatch.csv")
        nm = json.load(open(submission(tmp, path)))["needs_measured"]
        want = round(float(base["mem_available_gb"]) - float(ready["mem_available_gb"]), 2)
        assert nm["mem_available_drop_at_ready_gb"] == want, (nm, want)
        assert nm["swap_used_at_ready_gb"] == float(ready["swap_used_gb"]), nm
        assert nm["gpu_used_peak_mib"] == 5000.0, nm
        assert nm["gpu_used_peak_mib_max"] == 3800.0, nm
        assert nm["gpu_used_peak_mib_by_gpu"] == {"0": 1200.0, "1": 3800.0}, nm
        assert nm["memwatch_ready_t"] == ready["t"], nm


def test_to_localmaxxing_publishes_one_card_not_the_box():
    with tempfile.TemporaryDirectory() as tmp:
        srv, url = serve()
        try:
            r, rows = memwatch(tmp, url)
        finally:
            srv.shutdown()
        sub = submission(tmp, os.path.join(tmp, "memwatch.csv"))
        out = os.path.join(tmp, "lm.json")
        r = subprocess.run([sys.executable, LM, sub, "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        d = json.load(open(out))
        assert d["hardware"]["gpuCount"] == 2, d["hardware"]
        assert d["peakVramGb"] == round(3800 / 1024, 2), d
        assert d["peakVramGb"] != round(5000 / 1024, 2), "the sum over both cards was published as one card"


TESTS = [test_nvme_rand_refuses_a_file_that_is_too_small,
         test_nvme_rand_measures_a_file_that_is_big_enough,
         test_memwatch_marks_the_ready_row_and_every_gpu,
         test_the_submission_carries_the_drop_at_ready,
         test_to_localmaxxing_publishes_one_card_not_the_box]

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
