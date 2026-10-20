#!/bin/bash
# Record host + GPU memory while a server starts and serves, to measure what a recipe really needs.
# Start it BEFORE launching the server; stop it (Ctrl-C) after your sweep. Writes one CSV line per second.
#   tools/memwatch.sh memwatch.csv [health url]
# The health url (e.g. http://127.0.0.1:30100) is polled every second: the first answer is the "ready"
# row - the moment the server has the weights loaded and serves (PROTOCOL.md: the MemAvailable drop and
# the swap at ready). Without a url there is no ready row and the drop at ready is not measured.
# Columns: unix_s, phase (load|ready|serve), MemAvailable_GB, SwapUsed_GB, gpu_used_MiB (all GPUs
# summed) and one gpuN_used_MiB column per GPU.
OUT=${1:-memwatch.csv}
URL=${2:-}
GPUCOLS=$(timeout 5 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk '{printf ",gpu%s_used_mib", $1}')
echo "t,phase,mem_available_gb,swap_used_gb,gpu_used_mib$GPUCOLS" > "$OUT"
phase=load
while true; do
  if [ "$phase" = load ] && [ -n "$URL" ]; then
    if python3 -c "import sys,urllib.request; urllib.request.urlopen(sys.argv[1] + '/health', timeout=2)" "${URL%/}" >/dev/null 2>&1; then
      phase=ready
    fi
  fi
  read -r ma st sf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a,t,f}' /proc/meminfo)
  g=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd'|' -)
  # awk does the arithmetic: bc is not installed everywhere, and without it every column came out empty
  awk -v t="$(date +%s)" -v p="$phase" -v ma="$ma" -v sw="$((st-sf))" -v g="$g" \
    'BEGIN{n=split(g,a,"|"); s=0; for(i=1;i<=n;i++) s+=a[i];
           printf "%s,%s,%.2f,%.2f", t, p, ma/1e6, sw/1e6;
           if(n>0){printf ",%d", s; for(i=1;i<=n;i++) printf ",%d", a[i]}
           printf "\n"}' >> "$OUT"
  [ "$phase" = ready ] && phase=serve
  sleep 1
done
