#!/usr/bin/env python3
"""Describe this machine for a recipe submission: GPUs (NVIDIA / Intel XPU / AMD), host RAM, free disk and the storage
device behind a models directory (NVMe? rotational? through dm-crypt / LVM), CPU, kernel. No root needed.

  python3 tools/probe.py --models-dir ~/models --out probe.json
"""
import argparse, json, os, platform, re, shutil, subprocess


def run(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return {"total_gb": round(m["MemTotal"] / 1e9, 2), "available_gb": round(m["MemAvailable"] / 1e9, 2),
            "swap_total_gb": round(m.get("SwapTotal", 0) / 1e9, 2), "swap_used_gb": round((m.get("SwapTotal", 0) - m.get("SwapFree", 0)) / 1e9, 2)}


def leaf_disks(dev):
    """Follow a block device (dm-crypt, LVM, md) down to its physical disks via /sys/block/*/slaves."""
    name = os.path.basename(os.path.realpath(dev))
    slaves = f"/sys/class/block/{name}/slaves"
    if os.path.isdir(slaves) and os.listdir(slaves):
        out = []
        for s in os.listdir(slaves):
            out += leaf_disks("/dev/" + s)
        return out
    part = f"/sys/class/block/{name}/partition"
    if os.path.exists(part):  # partition -> parent disk
        name = os.path.basename(os.path.dirname(os.path.realpath(f"/sys/class/block/{name}")))
    return [name]


def storage(path):
    path = os.path.realpath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    du = shutil.disk_usage(path)
    src = run(["findmnt", "-n", "-o", "SOURCE,FSTYPE", "--target", path]).split()
    dev, fstype = (src + ["", ""])[:2]
    dev = re.sub(r"\[.*\]$", "", dev)  # btrfs subvolume suffix, e.g. /dev/mapper/root[/@home]
    disks = []
    if dev.startswith("/dev/"):
        for d in sorted(set(leaf_disks(dev))):
            rot = open(f"/sys/block/{d}/queue/rotational").read().strip() if os.path.exists(f"/sys/block/{d}/queue/rotational") else "?"
            model = open(f"/sys/block/{d}/device/model").read().strip() if os.path.exists(f"/sys/block/{d}/device/model") else ""
            disks.append({"disk": d, "nvme": d.startswith("nvme"), "rotational": rot == "1", "model": model})
    shown = "~" + path[len(os.path.expanduser("~")):] if path.startswith(os.path.expanduser("~")) else path
    return {"path": shown, "free_gb": round(du.free / 1e9, 2), "total_gb": round(du.total / 1e9, 2), "device": dev,
            "fstype": fstype, "physical_disks": disks, "all_nvme": bool(disks) and all(d["nvme"] for d in disks)}


def gpus():
    out = []
    q = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,driver_version",
             "--format=csv,noheader,nounits"])
    for line in q.strip().splitlines():
        i, n, mt, mu, bus, gen, wid, drv = [x.strip() for x in line.split(",")]
        mib = lambda v: int(v) if v.isdigit() else None
        # unified-memory GPUs (GB10, RTX Spark N1X) report memory as [N/A]: their memory is the host RAM above
        out.append({"vendor": "nvidia", "index": int(i), "name": n, "vram_gb": mib(mt) and round(mib(mt) / 1024),
                    "vram_mib": mib(mt), "vram_used_mib": mib(mu), "unified_memory": mib(mt) is None, "pci": bus,
                    "pcie_gen_max": gen, "pcie_width_max": wid, "driver": drv})
    x = run(["xpu-smi", "discovery", "-j"])
    if x.strip():
        try:
            for d in json.loads(x).get("device_list", []):
                out.append({"vendor": "intel", "index": d.get("device_id"), "name": d.get("device_name"),
                            "pci": d.get("pci_bdf_address"), "vram_gb": None})
        except Exception:
            pass
    elif os.path.isdir("/sys/class/drm"):
        for c in sorted(os.listdir("/sys/class/drm")):
            p = f"/sys/class/drm/{c}/device/vendor"
            if re.fullmatch(r"card\d+", c) and os.path.exists(p) and open(p).read().strip() == "0x8086":
                lspci = run(["lspci", "-s", os.path.basename(os.path.realpath(f"/sys/class/drm/{c}/device"))]).strip()
                if "VGA" in lspci or "Display" in lspci:
                    out.append({"vendor": "intel", "card": c, "name": lspci.split(": ", 1)[-1], "vram_gb": None})
    r = run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if r.strip():
        out.append({"vendor": "amd", "rocm_smi": json.loads(r)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="~/models")
    ap.add_argument("--out", default="probe.json")
    a = ap.parse_args()
    cpu = run(["lscpu"])
    res = {"host": {"kernel": platform.release(), "cpu": (re.search(r"Model name:\s*(.*)", cpu) or [None, ""])[1],
                    "cpus": os.cpu_count(), "avx512": "avx512f" in open("/proc/cpuinfo").read()},
           "ram": meminfo(), "storage": storage(a.models_dir), "gpus": gpus()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
