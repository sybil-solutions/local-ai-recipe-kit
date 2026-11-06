# Native Windows twin of memwatch.sh: same CSV columns, one line per second. Run it on the Windows host (not in WSL2).
# Start it BEFORE launching the server; stop it (Ctrl-C) after your sweep.
#   powershell -ExecutionPolicy Bypass -File tools\memwatch.ps1 memwatch.csv
# Columns: unix_s, mem_available_gb (\Memory\Available Bytes), swap_used_gb (page file in use, Win32_PageFileUsage),
# gpu_used_mib (\GPU Adapter Memory(*)\Dedicated Usage summed over adapters: every vendor, incl. an iGPU's small share).
# Counter paths are the English names; on a localized Windows they may need translating.
param([string]$Out = "memwatch.csv")
"t,mem_available_gb,swap_used_gb,gpu_used_mib" | Set-Content -Encoding ascii $Out
$ctr = '\Memory\Available Bytes', '\GPU Adapter Memory(*)\Dedicated Usage'
while ($true) {
  $t0 = Get-Date
  $s = (Get-Counter -Counter $ctr -ErrorAction SilentlyContinue).CounterSamples
  $ma = ($s | Where-Object Path -like '*\memory\available bytes').CookedValue
  $g = ($s | Where-Object Path -like '*dedicated usage' | Measure-Object CookedValue -Sum).Sum
  $sw = (Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum  # MB
  "{0},{1:F2},{2:F2},{3}" -f [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(), ($ma / 1e9), ($sw * 1MB / 1e9), [int64]($g / 1MB) |
    Add-Content -Encoding ascii $Out
  $left = 1000 - ((Get-Date) - $t0).TotalMilliseconds
  if ($left -gt 0) { Start-Sleep -Milliseconds $left }
}
