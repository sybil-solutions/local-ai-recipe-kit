#!/usr/bin/env python3
"""Validate submission files before they go into local-ai-registry (no third-party deps).
  python3 tools/validate.py <registry clone>/data/sources/recipe-kit/<card id>/<file>.json"""
import glob, json, os, re, sys

TARGETS = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(os.path.dirname(__file__), "..", "targets"))}
DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def check(path):
    errs = []
    try:
        s = json.load(open(path))
    except Exception as e:
        return [f"not JSON: {e}"]
    if os.path.getsize(path) > 1_000_000:
        errs.append("file over 1 MB")
    if s.get("schema") != "local-ai-recipe-kit/submission/v1":
        errs.append("schema must be local-ai-recipe-kit/submission/v1")
    for k in ("handle", "target", "launch", "probe", "quality", "speed"):
        if not s.get(k):
            errs.append(f"missing {k}")
    if s.get("target") and s["target"] not in TARGETS:
        errs.append(f"unknown target {s['target']} (have {sorted(TARGETS)})")
    L = s.get("launch") or {}
    if not DIGEST.match(str(L.get("image", ""))):
        errs.append("launch.image must be pinned by digest (repo@sha256:<64 hex>)")
    if not isinstance(L.get("args"), list):
        errs.append("launch.args must be the exact argv list")
    txt = json.dumps(s)
    if re.search(r"/(Users|home)/[A-Za-z0-9_.-]+/", txt):
        errs.append("personal absolute path found; use placeholders like <models dir>")
    if re.search(r"hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}", txt):
        errs.append("looks like a secret/token")
    p = (s.get("quality") or {}).get("panel") or {}
    if not isinstance(p.get("top1_agreement"), (int, float)) or not isinstance(p.get("mean_kl_top20"), (int, float)):
        errs.append("quality.panel needs top1_agreement and mean_kl_top20 (run tools/score_ref_panel.py)")
    sp = s.get("speed") or {}
    if sp.get("status") not in ("DONE",) and not str(sp.get("status", "")).startswith("EARLY_EXIT"):
        errs.append("speed.status must be DONE or EARLY_EXIT… (run tools/sweep.py)")
    for n in ("8192", "16384", "32768"):
        if n not in (sp.get("prefill") or {}):
            errs.append(f"speed.prefill missing {n}")
    if "C1" not in (sp.get("decode") or {}) and sp.get("status") == "DONE":
        errs.append("speed.decode missing C1")
    where = os.path.normpath(os.path.abspath(path)).split(os.sep)
    if len(where) < 3 or where[-3] != "recipe-kit":
        errs.append("put the file under data/sources/recipe-kit/<card id>/ in local-ai-registry")
    return errs


def main():
    if not sys.argv[1:]:
        sys.exit(__doc__)
    files = [f for a in sys.argv[1:] for f in glob.glob(a, recursive=True)]
    if not files:
        sys.exit("no files match " + " ".join(sys.argv[1:]))
    bad = 0
    for f in files:
        e = check(f)
        print(("FAIL " if e else "ok   ") + f + ("".join("\n  - " + x for x in e)))
        bad += bool(e)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
