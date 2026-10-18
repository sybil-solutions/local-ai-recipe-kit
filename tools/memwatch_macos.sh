#!/bin/bash
# macOS counterpart of memwatch.sh for a Mac that hosts the server (no /proc, no nvidia-smi on the Mac).
#   tools/memwatch_macos.sh memwatch.csv [engine pid]
# Columns: unix_s, mem_available_gb (free + inactive + speculative + purgeable pages: what macOS can hand out without
# paging), swap_used_gb (sysctl vm.swapusage), gpu_used_mib (always empty here: the card is not visible to the Mac's
# tools), proc_rss_gb (resident set of the pid given as the 2nd argument, empty without one).
OUT=${1:-memwatch.csv}
PID=$2
echo "t,mem_available_gb,swap_used_gb,gpu_used_mib,proc_rss_gb" > "$OUT"
while true; do
  vm=$(vm_stat)
  ps_=$(echo "$vm" | awk '/page size of/{gsub(/[^0-9]/,"",$8); print $8}')
  av=$(echo "$vm" | awk -v ps="$ps_" '/Pages free/{f=$3}/Pages inactive/{i=$3}/Pages speculative/{s=$3}/Pages purgeable/{p=$3}END{printf "%.2f", (f+i+s+p)*ps/1e9}')
  sw=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%.2f", $6/1000}')
  rss=""
  [ -n "$PID" ] && rss=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.2f", $1*1024/1e9}')
  echo "$(date +%s),$av,$sw,,$rss" >> "$OUT"
  sleep 1
done
