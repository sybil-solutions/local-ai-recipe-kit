"""Ephemeral native llama.cpp/Vulkan benchmark. All created remote files are cleaned on exit."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

LABEL = sys.argv[1]
LLAMA = "9c2e0e491a822adae1f0b1c831adb4160057d24f"
VULKAN = "c46850864f4661461b0f6cb9922c058ffea4915e"
SPIRV = "86f980c731e62ae4eaf383d320449d71687936bf"
MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
MODEL_REV = "f86cb2c1fa58255f8052cc32aeede1b7482d4361"
MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_SHA = "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"
MODEL_SIZE = 1117320768
report = {"schema": "zsoxi/native-node-recipe/v1", "node": LABEL, "backend_commit": LLAMA,
          "model": {"repository": MODEL_REPO, "revision": MODEL_REV, "file": MODEL_FILE,
                    "sha256": MODEL_SHA, "bytes": MODEL_SIZE, "license": "Apache-2.0"},
          "protocol": "Native Vulkan, fixed synthetic llama-bench plus uncapped natural-length HTTP completions; not the Qwen180B reference-panel protocol.",
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "attempts": []}
children = []
sampler_stop = threading.Event()
samples = []
temp_root = None

def status(stage):
    print(json.dumps({"node": LABEL, "stage": stage}), flush=True)

def sanitized(value):
    text = str(value)
    if temp_root:
        text = text.replace(str(temp_root), "<WORKDIR>")
    text = text.replace(str(Path.home()), "<HOME>")
    return text

def stop_signal(signum, frame):
    raise KeyboardInterrupt(f"Stopped by signal {signum}")

for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, stop_signal)

def run(args, *, env=None, timeout=180, cwd=None, filename="command.log"):
    logfile = temp_root / filename
    with logfile.open("w") as log:
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        children.append(proc)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError(f"Command timed out: {args[0]}")
    if rc:
        tail = sanitized(logfile.read_text(errors="replace")[-6000:])
        raise RuntimeError(f"Command exit {rc}: {args[0]}\n{tail}")
    return logfile.read_text(errors="replace")

def clone(repo, revision, destination):
    destination.mkdir()
    run(["git", "init", "-q", str(destination)])
    run(["git", "-C", str(destination), "remote", "add", "origin", "https://github.com/" + repo + ".git"])
    run(["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "fetch", "--depth", "1", "origin", revision], timeout=300)
    run(["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", "-q", revision])

def sys_memory():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0]) * 1024
    return fields

def sample_memory(server=None):
    memory = sys_memory()
    data = {"monotonic": time.monotonic(), "mem_available_bytes": memory["MemAvailable"],
            "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"]}
    query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
    if query.returncode == 0:
        values = query.stdout.strip().splitlines()[0].split(",")
        for name, value in zip(("gpu_used_mib", "gpu_power_w", "gpu_temperature_c", "gpu_sm_mhz", "gpu_memory_mhz"), values):
            try:
                data[name] = float(value.strip())
            except ValueError:
                data[name] = None
    if server:
        try:
            data["server_rss_bytes"] = int(Path(f"/proc/{server.pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError):
            pass
    return data

def sampler(server):
    while not sampler_stop.is_set():
        try:
            samples.append(sample_memory(server))
        except Exception:
            pass
        sampler_stop.wait(1)

def post(base, path, payload, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def stream(base, messages, tools=None):
    payload = {"model": "qwen25-coder15b", "messages": messages, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}, "cache_prompt": False}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    started = time.monotonic()
    first = None
    last = None
    text = ""
    calls = []
    usage = {}
    finish = None
    timings = None
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            if time.monotonic() - started > 120:
                raise TimeoutError("Natural completion exceeded 120 seconds; no truncated result accepted")
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or ""
                if piece or delta.get("tool_calls"):
                    stamp = time.monotonic()
                    first = first or stamp
                    last = stamp
                    text += piece
                    calls.extend(delta.get("tool_calls", []))
                finish = choice.get("finish_reason") or finish
    ended = time.monotonic()
    tokens = usage.get("completion_tokens")
    token_seconds = last - first if first is not None and last is not None else None
    result = {"text": text, "tool_calls": calls, "finish_reason": finish, "usage": usage,
              "prompt_sha256": prompt_hash, "ttft_seconds": first - started if first else None,
              "wall_seconds": ended - started, "decode_seconds": token_seconds,
              "stream_decode_tps": (tokens - 1) / token_seconds if tokens and tokens > 1 and token_seconds and token_seconds > 0 else None,
              "server_timings": timings}
    return result

def tagged_prompt(text):
    return [{"role": "user", "content": "Request nonce: " + os.urandom(8).hex() + "\n" + text}]

def stop_children():
    sampler_stop.set()
    for proc in reversed(children):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                except ProcessLookupError:
                    pass

try:
    with tempfile.TemporaryDirectory(prefix="sybil-recipe-20261007-") as directory:
        temp_root = Path(directory)
        try:
            status("hardware-probe")
            hardware = {}
            for field, command in {
                "cpu": ["lscpu", "-J"],
                "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                "kernel": ["uname", "-r"],
            }.items():
                value = subprocess.run(command, text=True, capture_output=True, timeout=15)
                hardware[field] = value.stdout.strip()
            hardware["ram_total_bytes"] = sys_memory()["MemTotal"]
            report["hardware"] = hardware
            status("isolated-build-tools")
            run(["python3", "-m", "venv", str(temp_root / "venv")])
            py = temp_root / "venv/bin/python"
            run([str(py), "-m", "pip", "install", "--no-cache-dir", "cmake==4.1.3", "ninja==1.13.0"], timeout=300)
            cmake = str(temp_root / "venv/bin/cmake")
            env = os.environ.copy()
            env["PATH"] = str(temp_root / "venv/bin") + os.pathsep + env["PATH"]
            env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
            env["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
            env["GGML_VK_VISIBLE_DEVICES"] = "0"
            env["GIT_TERMINAL_PROMPT"] = "0"
            env["XDG_CACHE_HOME"] = str(temp_root / "cache")
            env["CUDA_CACHE_PATH"] = str(temp_root / "cache/cuda")
            env["__GL_SHADER_DISK_CACHE_PATH"] = str(temp_root / "cache/nvidia")
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            status("pinned-sources")
            clone("ggml-org/llama.cpp", LLAMA, temp_root / "llama.cpp")
            clone("KhronosGroup/Vulkan-Headers", VULKAN, temp_root / "Vulkan-Headers")
            clone("KhronosGroup/SPIRV-Headers", SPIRV, temp_root / "SPIRV-Headers")
            prefix = temp_root / "deps"
            run([cmake, "-S", str(temp_root / "SPIRV-Headers"), "-B", str(temp_root / "spirv-build"),
                 "-DCMAKE_INSTALL_PREFIX=" + str(prefix)], env=env)
            run([cmake, "--install", str(temp_root / "spirv-build")], env=env)
            status("compile-vulkan")
            build = temp_root / "build"
            configure = [cmake, "-S", str(temp_root / "llama.cpp"), "-B", str(build), "-G", "Ninja",
                         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_VULKAN=ON", "-DGGML_CCACHE=OFF", "-DLLAMA_CURL=OFF",
                         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=ON", "-DGGML_NATIVE=ON",
                         "-DVulkan_INCLUDE_DIR=" + str(temp_root / "Vulkan-Headers/include"),
                         "-DCMAKE_CXX_FLAGS=-I" + str(temp_root / "SPIRV-Headers/include"),
                         "-DVulkan_LIBRARY=/usr/lib/libvulkan.so", "-DCMAKE_PREFIX_PATH=" + str(prefix)]
            run(configure, env=env, timeout=180, filename="configure.log")
            run([cmake, "--build", str(build), "--target", "llama-server", "llama-bench", "-j", "2"], env=env, timeout=1500, filename="build.log")
            report["build"] = {"args": [sanitized(a) for a in configure], "parallel_jobs": 2,
                               "vulkan_headers_commit": VULKAN, "spirv_headers_commit": SPIRV,
                               "cmake": "4.1.3", "ninja": "1.13.0"}
            status("pinned-model-download")
            model = temp_root / MODEL_FILE
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/{MODEL_FILE}"
            run(["curl", "-L", "--fail", "--connect-timeout", "15", "--max-time", "900", "--retry", "2",
                 "--retry-delay", "3", "-o", str(model), url], timeout=1000, filename="download.log")
            assert model.stat().st_size == MODEL_SIZE
            with model.open("rb") as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == MODEL_SHA
            report["model_checksum_verified"] = True
            status("synthetic-speed-check")
            bench = str(build / "bin/llama-bench")
            bench_args = [bench, "-m", str(model), "-ngl", "99", "-p", "512,2048", "-n", "128", "-r", "3", "-t", "4", "-o", "json"]
            try:
                raw = run(bench_args, env=env, timeout=240, filename="bench.log")
                decoded = None
                for start in (match.start() for match in re.finditer(r"\[", raw)):
                    try:
                        candidate, end = json.JSONDecoder().raw_decode(raw[start:])
                        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict) and "avg_ts" in candidate[0]:
                            decoded = candidate
                            break
                    except json.JSONDecodeError:
                        pass
                if decoded is None:
                    raise RuntimeError("No llama-bench JSON array found in captured output")
                report["synthetic_bench"] = decoded
                report["synthetic_bench_args"] = [sanitized(a) for a in bench_args]
            except Exception as error:
                report["synthetic_bench_error"] = sanitized(error)
            status("server-start")
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            base = f"http://127.0.0.1:{port}"
            server_args = [str(build / "bin/llama-server"), "-m", str(model), "--alias", "qwen25-coder15b",
                           "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-c", "8192", "-np", "1",
                           "-t", "4", "-b", "512", "-ub", "128", "--flash-attn", "off", "--n-predict", "-1"]
            report["launch"] = {"kind": "native", "args": [sanitized(a) if a != str(port) else "<PORT>" for a in server_args],
                                "environment": {key: env[key] for key in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES", "GGML_VK_VISIBLE_DEVICES")},
                                "context_tokens": 8192, "parallel_slots": 1, "gpu_layers": 99}
            baseline = sample_memory()
            server_log = (temp_root / "server.log").open("w")
            server = subprocess.Popen(server_args, env=env, stdout=server_log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(server)
            monitor = threading.Thread(target=sampler, args=(server,), daemon=True)
            monitor.start()
            for attempt in range(90):
                if server.poll() is not None:
                    raise RuntimeError("Server exited: " + sanitized((temp_root / "server.log").read_text()[-5000:]))
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError("Server did not become healthy in the bounded startup window")
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
                report["served_models"] = json.load(response)
            status("quality-checks")
            stream(base, tagged_prompt("Reply with OK."))
            report["quality"] = {}
            for name, prompt, expected in (
                ("chat", "Reply with the word READY.", "READY"),
                ("arithmetic", "Calculate 17 multiplied by 23. Reply with the number only.", "391"),
                ("json", 'Reply with exactly this JSON object: {"ok":true,"count":3}', None),
            ):
                answer = stream(base, tagged_prompt(prompt))
                if name == "json":
                    try:
                        extracted = re.search(r"\{.*\}", answer["text"], re.S).group()
                        passed = json.loads(extracted) == {"ok": True, "count": 3}
                    except Exception:
                        passed = False
                else:
                    passed = expected in answer["text"]
                report["quality"][name] = {"passed": passed, "response": answer}
            tool = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.",
                     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
            try:
                answer = stream(base, tagged_prompt("Use the get_weather tool to get weather in London."), tool)
                report["quality"]["tools"] = {"passed": bool(answer["tool_calls"]), "response": answer}
            except Exception as error:
                report["quality"]["tools"] = {"passed": False, "error": sanitized(error)}
            status("context-check")
            random.seed(29)
            filler = " ".join("entry" + str(random.randrange(100000)) for _ in range(1300))
            needle = "The exact project marker is LANTERN-CEDAR-7342."
            content = filler[:len(filler)//2] + "\n" + needle + "\n" + filler[len(filler)//2:] + "\nWhat is the exact project marker? Reply only with it."
            answer = stream(base, tagged_prompt(content))
            report["quality"]["context"] = {"passed": "LANTERN-CEDAR-7342" in answer["text"], "response": answer}
            status("natural-length-speed-check")
            speed = {}
            prompts = [
                "Write a Python function that deduplicates a list while preserving order, and briefly explain its complexity.",
                "Explain three practical differences between a process and a thread in operating systems.",
                "Write a Python binary search function for a sorted integer list, including the empty-list case.",
                "Describe how an HTTP client should distinguish retryable errors from invalid input.",
            ]
            for concurrency in (1, 2, 4):
                rounds = []
                for round_id in range(2):
                    start = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        results = list(pool.map(lambda i: stream(base, tagged_prompt(prompts[i])), range(concurrency)))
                    wall = time.monotonic() - start
                    total_tokens = sum((result["usage"].get("completion_tokens") or 0) for result in results)
                    rounds.append({"round": round_id, "tokens": total_tokens, "wall_seconds": wall,
                                   "end_to_end_aggregate_tps": total_tokens / wall, "streams": results})
                speed[f"C{concurrency}"] = {"rounds": rounds}
            report["natural_speed"] = speed
            status("collect-evidence")
            sampler_stop.set()
            monitor.join(timeout=10)
            report["memory"] = {"baseline": baseline, "samples": len(samples),
                                "ram_available_drop_peak_bytes": max(0, baseline["mem_available_bytes"] - min(s["mem_available_bytes"] for s in samples)),
                                "gpu_used_peak_mib": max(s.get("gpu_used_mib") or 0 for s in samples),
                                "server_rss_peak_bytes": max(s.get("server_rss_bytes", 0) for s in samples),
                                "swap_used_peak_bytes": max(s["swap_used_bytes"] for s in samples),
                                "gpu_temperature_peak_c": max(s.get("gpu_temperature_c") or 0 for s in samples)}
            report["server_log_tail"] = sanitized((temp_root / "server.log").read_text(errors="replace")[-7000:])
            report["status"] = "MEASURED"
        except BaseException as error:
            report["status"] = "FAILED"
            report["error"] = sanitized(error)
        finally:
            status("cleanup")
            stop_children()
            if "server_log" in locals():
                server_log.close()
    report["cleanup"] = {"temporary_directory_removed": not temp_root.exists(),
                         "owned_processes_stopped": all(p.poll() is not None for p in children),
                         "model_retained": False, "global_packages_installed": False}
except BaseException as error:
    report["status"] = "FAILED"
    report["outer_error"] = sanitized(error)
report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
report = json.loads(sanitized(json.dumps(report)))
print("RESULT_JSON:" + json.dumps(report), flush=True)
sys.exit(0 if report.get("status") == "MEASURED" else 1)
