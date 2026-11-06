# E002: NVMe random-read baseline with O_DIRECT pread from a thread pool (no fio on the box).
# usage: nvme_rand.py <file> [row_bytes ...]  -> IOPS and GB/s per (block size, threads)
# Linux: O_DIRECT preadv. Native Windows: unbuffered ReadFile (FILE_FLAG_NO_BUFFERING, positional via OVERLAPPED) so
# the page cache is bypassed the same way; run it on the host, not in WSL2 (/mnt/c goes through 9p and measures that).
import os, sys, time, json, mmap, random, threading
path = sys.argv[1]; sizes = [int(x) for x in sys.argv[2:]] or [4096, 16384, 65536, 1 << 20]
fsz = os.path.getsize(path); ALIGN = 4096

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    def open_direct():
        # GENERIC_READ, FILE_SHARE_READ|WRITE (a live server may hold the file), OPEN_EXISTING, FILE_FLAG_NO_BUFFERING
        h = k32.CreateFileW(path, 0x80000000, 0x3, None, 3, 0x20000000, None)
        if h in (None, wintypes.HANDLE(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        return h

    def pread(h, buf, off):
        ov = OVERLAPPED(Offset=off & 0xFFFFFFFF, OffsetHigh=off >> 32); n = wintypes.DWORD()
        if not k32.ReadFile(h, (ctypes.c_char * len(buf)).from_buffer(buf), len(buf), ctypes.byref(n), ctypes.byref(ov)):
            raise ctypes.WinError(ctypes.get_last_error())
        return n.value

    close = k32.CloseHandle
else:
    open_direct = lambda: os.open(path, os.O_RDONLY | os.O_DIRECT)
    pread = lambda fd, buf, off: os.preadv(fd, [buf], off)
    close = os.close


def worker(bs, n, res, seed):
    fd = open_direct(); buf = mmap.mmap(-1, bs); r = random.Random(seed); done = 0  # mmap: page-aligned buffer
    t_end = time.perf_counter() + 4.0
    while time.perf_counter() < t_end:
        off = r.randrange(0, (fsz - bs) // ALIGN) * ALIGN; done += pread(fd, buf, off)
    res.append(done); close(fd)
out = {"file": path, "file_bytes": fsz, "cells": []}
for bs in sizes:
    for th in (1, 8, 32, 64):
        res = []; ts = [threading.Thread(target=worker, args=(bs, 0, res, i)) for i in range(th)]
        t = time.perf_counter(); [x.start() for x in ts]; [x.join() for x in ts]; dt = time.perf_counter() - t
        tot = sum(res); out["cells"].append({"bs": bs, "threads": th, "GBps": round(tot / dt / 1e9, 3), "kIOPS": round(tot / bs / dt / 1e3, 1)})
        print(out["cells"][-1], flush=True)
print(json.dumps(out))
