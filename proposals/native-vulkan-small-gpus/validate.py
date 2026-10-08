"""Validate these native observations, without treating them as qualified Sybil submissions."""
from pathlib import Path
import ast
import hashlib
import json
import re

root = Path(__file__).resolve().parent
for name in ("gt-1030-2gb", "t1200-laptop-4gb"):
    path = root / (name + ".json")
    text = path.read_text(encoding="utf-8")
    result = json.loads(text)
    assert path.stat().st_size < 1_000_000
    assert result["status"] == "MEASURED" and result["model_checksum_verified"]
    assert result["qualified_for_current_sybil_target"] is False
    assert result["launch"]["kind"] == "native"
    assert result["launch"]["context_tokens"] == 8192 and result["launch"]["parallel_slots"] == 1
    assert result["cleanup"]["temporary_directory_removed"] and result["cleanup"]["owned_processes_stopped"]
    assert result["independent_cleanup"]["temporary_directories"] == []
    assert result["independent_cleanup"]["owned_model_processes"] == []
    assert result["model"]["sha256"] == "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"
    assert result["quality"]["arithmetic"]["response"]["text"].strip() == "391"
    assert result["quality"]["tools"]["passed"] is False
    for value in result["natural_speed"].values():
        for round in value["rounds"]:
            assert round["tokens"] == sum(item["usage"]["completion_tokens"] for item in round["streams"])
            assert abs(round["end_to_end_aggregate_tps"] - round["tokens"] / round["wall_seconds"]) < 1e-8
            assert all(item["finish_reason"] == "stop" for item in round["streams"])
    assert not re.search(r"192\.168\.|/(?:home|Users)/[A-Za-z0-9_.-]+/|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}", text)
    print("ok native observation", name)
ast.parse((root / "reproduce.py").read_text(encoding="utf-8"))
print("ok reproducer syntax")
