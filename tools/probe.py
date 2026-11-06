#!/usr/bin/env python3
"""Describe this machine for a recipe submission: GPUs (NVIDIA / Intel XPU / AMD), host RAM, free disk and the storage
device behind a models directory (NVMe? rotational? through dm-crypt / LVM), CPU, kernel. No root needed.
Linux and native Windows (run it on the Windows host, not in WSL2: WSL2 sees neither the host's RAM nor its disks/GPU).

  python3 tools/probe.py --models-dir ~/models --out probe.json
"""
import argparse, base64, json, os, platform, re, shutil, subprocess

WIN = os.name == "nt"


def run(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def ps(script, timeout=60):
    """Run a PowerShell script (Windows) and parse its JSON output; None on any failure."""
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    try:
        return json.loads(run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc], timeout) or "null")
    except Exception:
        return None


def gt_s(gen):
    """PCIe generation number (Windows DEVPKEY link speed) -> sysfs-style transfer rate."""
    return {1: "2.5 GT/s", 2: "5.0 GT/s", 3: "8.0 GT/s", 4: "16.0 GT/s", 5: "32.0 GT/s", 6: "64.0 GT/s"}.get(gen)


def meminfo():
    if WIN:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                       [(k, ctypes.c_ulonglong) for k in ("phys", "avail", "pagef", "pagef_avail", "virt", "virt_avail", "ext")]
        s = MS(); s.len = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        # Windows has no swap partition: report the page file(s) in the swap fields (sizes in MB, same source as memwatch.ps1)
        pf = ps("$u = @(Get-CimInstance Win32_PageFileUsage); ConvertTo-Json -Compress -InputObject @{"
                "total = ($u | Measure-Object AllocatedBaseSize -Sum).Sum; used = ($u | Measure-Object CurrentUsage -Sum).Sum }") or {}
        mb = lambda v: v is not None and round(v * 2**20 / 1e9, 2) or 0.0
        return {"total_gb": round(s.phys / 1e9, 2), "available_gb": round(s.avail / 1e9, 2),
                "swap_total_gb": mb(pf.get("total")), "swap_used_gb": mb(pf.get("used")), "swap_is_pagefile": True}
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
    if WIN:
        return storage_win(path, du)
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


def storage_win(path, du):
    """Drive letter -> partition -> disk(s) (a Storage Spaces / dynamic volume can span several) -> bus and media type."""
    letter = os.path.splitdrive(path)[0].rstrip(":")
    r = ps(f"""$v = Get-Volume -DriveLetter {letter} -ErrorAction SilentlyContinue
$n = @(Get-Partition -DriveLetter {letter} -ErrorAction SilentlyContinue | Get-Disk | ForEach-Object {{ $_.Number }})
$d = @(Get-PhysicalDisk | Where-Object {{ $n -contains [int]$_.DeviceId }} | ForEach-Object {{
  [pscustomobject]@{{ disk = "PhysicalDrive$($_.DeviceId)"; bus = "$($_.BusType)"; media = "$($_.MediaType)"; model = $_.FriendlyName }} }})
ConvertTo-Json -Depth 3 -Compress -InputObject @{{ fstype = $v.FileSystem; disks = $d }}""") or {}
    disks = [{"disk": d["disk"], "nvme": d["bus"] == "NVMe", "rotational": d["media"] == "HDD", "bus": d["bus"],
              "model": d["model"]} for d in r.get("disks") or []]
    shown = "~" + path[len(os.path.expanduser("~")):] if path.startswith(os.path.expanduser("~")) else path
    return {"path": shown, "free_gb": round(du.free / 1e9, 2), "total_gb": round(du.total / 1e9, 2), "device": letter + ":",
            "fstype": r.get("fstype") or "", "physical_disks": disks, "all_nvme": bool(disks) and all(d["nvme"] for d in disks)}


def amd_sysfs():
    """AMD GPUs from amdgpu sysfs, normalized like the NVIDIA entries. pcie_card_link is the GPU endpoint's own link;
    on cards with an on-board PCIe switch (RDNA3/RDNA4) that is the switch's internal link, so pcie_root_port_link (the
    CPU/chipset root port the card is plugged into) is the one that shows the slot's real generation and width."""
    out = []
    if not os.path.isdir("/sys/class/drm"):
        return out
    rd = lambda p: open(p).read().strip() if os.path.exists(p) else None
    spd = lambda d, k: rd(f"{d}/{k}_link_speed") and rd(f"{d}/{k}_link_speed").replace(" PCIe", "")  # "16.0 GT/s PCIe"
    link = lambda d, k="current": spd(d, k) and f"{spd(d, k)} x{rd(f'{d}/{k}_link_width')}"
    for c in sorted(os.listdir("/sys/class/drm")):
        dev = f"/sys/class/drm/{c}/device"
        if not re.fullmatch(r"card\d+", c) or rd(f"{dev}/vendor") != "0x1002" or not os.path.exists(f"{dev}/mem_info_vram_total"):
            continue
        real = os.path.realpath(dev)
        chain = [p for p in real.split("/") if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", p)]
        root = "/sys/bus/pci/devices/" + chain[0] if len(chain) > 1 else None
        bdf = os.path.basename(real)
        name = rd(f"{dev}/product_name") or run(["lspci", "-s", bdf]).strip().split(": ", 1)[-1]
        vt, vu = int(rd(f"{dev}/mem_info_vram_total")), int(rd(f"{dev}/mem_info_vram_used") or 0)
        out.append({"vendor": "amd", "card": c, "name": name, "pci": bdf, "vram_gb": round(vt / 2**30), "vram_mib": vt // 2**20,
                    "vram_used_mib": vu // 2**20, "pcie_card_link": link(dev), "pcie_card_link_max": link(dev, "max"),
                    "pcie_root_port_link": link(root) if root else None, "behind_switch": len(chain) > 2})
    return out


WIN_GPUS = r"""$cls = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$keys = 'DEVPKEY_PciDevice_CurrentLinkSpeed','DEVPKEY_PciDevice_CurrentLinkWidth','DEVPKEY_PciDevice_MaxLinkSpeed','DEVPKEY_PciDevice_MaxLinkWidth'
function Prop($id, $k) { (Get-PnpDeviceProperty -InstanceId $id -KeyName $k -ErrorAction SilentlyContinue).Data }
$out = @(Get-PnpDevice -Class Display -PresentOnly | Where-Object { $_.InstanceId -match '^PCI\\VEN_(1002|8086)' } | ForEach-Object {
  $id = $_.InstanceId; $link = @{}
  Get-PnpDeviceProperty -InstanceId $id -KeyName $keys -ErrorAction SilentlyContinue | ForEach-Object { $link[$_.KeyName.Split('_')[-1]] = $_.Data }
  $reg = Get-ItemProperty "$cls\$((Prop $id 'DEVPKEY_Device_Driver').Split('\')[-1])" -ErrorAction SilentlyContinue
  $bridges = 0; $cur = $id
  while (($cur = Prop $cur 'DEVPKEY_Device_Parent') -match '^PCI\\') { $bridges++ }
  [pscustomobject]@{ id = $id; name = $_.FriendlyName; location = Prop $id 'DEVPKEY_Device_LocationInfo'; bridges = $bridges
    vram_bytes = $reg.'HardwareInformation.qwMemorySize'; driver = $reg.DriverVersion; link = $link }
})
ConvertTo-Json -InputObject $out -Depth 4 -Compress"""


def gpus_win():
    """AMD / Intel GPUs on native Windows (NVIDIA comes from nvidia-smi, which works there too). VRAM is the driver's
    64-bit HardwareInformation.qwMemorySize (Win32_VideoController caps at 4 GB). Windows exposes PCIe link state only on
    the GPU endpoint, not on bridges or root ports, so behind an on-board switch (RDNA3/RDNA4: bridges > 1) the reported
    link is the switch's internal one and pcie_root_port_link stays null."""
    out = []
    for d in ps(WIN_GPUS) or []:
        vendor = "amd" if "VEN_1002" in d["id"] else "intel"
        vb, l = d.get("vram_bytes"), d.get("link") or {}
        fmt = lambda s, w: gt_s(s) and f"{gt_s(s)} x{w}"
        out.append({"vendor": vendor, "name": d["name"], "pci": d.get("location"), "vram_gb": vb and round(vb / 2**30),
                    "vram_mib": vb and vb // 2**20, "driver": d.get("driver"),
                    "pcie_card_link": fmt(l.get("CurrentLinkSpeed"), l.get("CurrentLinkWidth")),
                    "pcie_card_link_max": fmt(l.get("MaxLinkSpeed"), l.get("MaxLinkWidth")),
                    "pcie_root_port_link": None, "behind_switch": (d.get("bridges") or 0) > 1})
    return out


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
    if WIN:
        return out + gpus_win()
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
    amd = amd_sysfs()
    if amd:
        return out + amd
    r = run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])  # fallback: no amdgpu sysfs
    if r.strip():
        out.append({"vendor": "amd", "rocm_smi": json.loads(r)})
    return out


def host():
    if WIN:
        import ctypes, winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        return {"os": "windows", "kernel": platform.version(), "cpu": cpu, "cpus": os.cpu_count(),
                "avx512": bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(41))}  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
    cpu = run(["lscpu"])
    return {"kernel": platform.release(), "cpu": (re.search(r"Model name:\s*(.*)", cpu) or [None, ""])[1],
            "cpus": os.cpu_count(), "avx512": "avx512f" in open("/proc/cpuinfo").read()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="~/models")
    ap.add_argument("--out", default="probe.json")
    a = ap.parse_args()
    res = {"host": host(), "ram": meminfo(), "storage": storage(a.models_dir), "gpus": gpus()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
