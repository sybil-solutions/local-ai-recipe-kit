# E002: NVMe random-read baseline with O_DIRECT pread from a thread pool (no fio on the box).
# usage: nvme_rand.py <file> [row_bytes ...]  -> IOPS and GB/s per (block size, threads)
import os, sys, time, json, mmap, random, threading
path = sys.argv[1]; sizes = [int(x) for x in sys.argv[2:]] or [4096, 16384, 65536, 1 << 20]
fsz = os.path.getsize(path); ALIGN = 4096
if fsz < max(sizes) + ALIGN:
    # the random offsets need a block plus one aligned step: below that there is nothing to read
    sys.exit(f"{path}: {fsz} bytes cannot hold a {max(sizes)} byte block with random offsets "
             f"(it needs at least {max(sizes) + ALIGN} bytes): nothing was measured")
def worker(bs, seed, res):
    """One reader thread: appends the bytes it read, or the error that stopped it."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT); buf = mmap.mmap(-1, bs); r = random.Random(seed); done = 0
        t_end = time.perf_counter() + 4.0
        while time.perf_counter() < t_end:
            off = r.randrange(0, (fsz - bs) // ALIGN) * ALIGN; done += os.preadv(fd, [buf], off)
        res.append({"bytes": done}); os.close(fd)
    except Exception as e:
        res.append({"error": f"{type(e).__name__}: {e}"})
out = {"file": path, "file_bytes": fsz, "cells": []}
bad = 0
for bs in sizes:
    for th in (1, 8, 32, 64):
        res = []; ts = [threading.Thread(target=worker, args=(bs, i, res)) for i in range(th)]
        t = time.perf_counter(); [x.start() for x in ts]; [x.join() for x in ts]; dt = time.perf_counter() - t
        errs = [r["error"] for r in res if "error" in r]
        tot = sum(r["bytes"] for r in res if "bytes" in r)
        if errs or len(res) != th:
            cell = {"bs": bs, "threads": th, "error": "; ".join(errs) or "no measurement"}
        elif tot <= 0:
            cell = {"bs": bs, "threads": th, "error": "no bytes read"}
        else:
            cell = {"bs": bs, "threads": th, "GBps": round(tot / dt / 1e9, 3), "kIOPS": round(tot / bs / dt / 1e3, 1)}
        if "error" in cell:
            bad += 1
        out["cells"].append(cell)
        print(cell, flush=True)
print(json.dumps(out))
sys.exit(1 if bad else 0)
