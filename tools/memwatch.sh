#!/bin/bash
# Record host + GPU memory while a server starts and serves, to measure what a recipe really needs.
# Start it BEFORE launching the server; stop it (Ctrl-C) after your sweep. Writes one CSV line per second.
#   tools/memwatch.sh memwatch.csv
# Columns: unix_s, MemAvailable_GB, SwapUsed_GB, gpu_used_MiB (nvidia-smi, all GPUs summed; without nvidia-smi, the
# discrete AMD cards' sysfs mem_info_vram_used; empty otherwise)
OUT=${1:-memwatch.csv}
echo "t,mem_available_gb,swap_used_gb,gpu_used_mib" > "$OUT"
while true; do
  read -r ma st sf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a,t,f}' /proc/meminfo)
  g=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  if [ -z "$g" ]; then  # AMD: sum mem_info_vram_used of discrete cards (vram_total > 16 GiB), in MiB
    g=$(for d in /sys/class/drm/card[0-9]*/device; do t=$(cat "$d/mem_info_vram_total" 2>/dev/null) || continue
        [ "$t" -gt 17179869184 ] && cat "$d/mem_info_vram_used"; done | awk '{s+=$1} END{if (NR) printf "%d", s/1048576}')
  fi
  echo "$(date +%s),$(echo "scale=2;$ma/1000000" | bc),$(echo "scale=2;($st-$sf)/1000000" | bc),$g" >> "$OUT"
  sleep 1
done
