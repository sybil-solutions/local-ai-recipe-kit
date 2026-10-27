#!/usr/bin/env python3
import json
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools" / "to_localmaxxing.py"


def convert(submission, *flags):
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "localmaxxing.json"
        subprocess.run(
            ["python3", str(CONVERTER), str(submission), "--out", str(output), *flags],
            check=True,
        )
        return json.loads(output.read_text())


def test_unknown_vllm_metadata_stays_null():
    run = convert(ROOT / "submissions/2x-r9700-32gb/2026-09-29-benstadin-m2.json")
    assert run["engine"]["engineName"] is None
    assert run["engine"]["engineRepository"] is None
    assert run["engine"]["quantization"] is None
    assert run["engineFlags"]["specDecoding"] is True
    assert run["batchSize"] is None


def test_explicit_sglang_metadata_is_exported():
    run = convert(ROOT / "submissions/rtx-3090-24gb/2026-09-29-0xSero.json")
    assert run["engine"]["engineName"] == "sglang"
    assert run["engine"]["quantization"] == "exl3"
    assert run["engine"]["engineRepository"] is None
    assert run["engineFlags"]["specDecoding"] is False
    assert run["batchSize"] is None


def test_cli_overrides_use_operator_verified_values():
    submission = ROOT / "submissions/2x-r9700-32gb/2026-09-29-benstadin-m2.json"
    run = convert(
        submission,
        "--engine-name", "vLLM with MTP",
        "--engine-repo", "https://github.com/operator/vllm",
    )
    assert run["engine"]["engineName"] == "vLLM with MTP"
    assert run["engine"]["engineRepository"] == "https://github.com/operator/vllm"

