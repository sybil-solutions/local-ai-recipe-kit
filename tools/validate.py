#!/usr/bin/env python3
"""Validate submission files (no third-party deps; CI runs this on every PR).
  python3 tools/validate.py submissions/**/*.json"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(HERE, "..", "targets"))}
DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
# the same shapes to_localmaxxing.py refuses to write, so a submission that passes here is one that tool would
# also take: a token of any of the three kinds, a bare address, or a personal path
SECRETISH = re.compile(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
                        r"|(?:\d{1,3}\.){3}\d{1,3}|/(?:Users|home)/[^/\s\"]+")
# The effective date of the submission rules below: a submission with an older "submitted"
# date is grandfathered - validate.py warns about it instead of failing it.
RULE_FROM = "2026-10-20"


def before_rule(sub):
    """True for submissions that predate the rule: they are warned about, not failed."""
    d = str(sub.get("submitted") or "")
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)) and d < RULE_FROM


def band_of(target):
    """The machine-readable quality band of targets/<target>.md (a fenced json block), or None."""
    path = os.path.join(HERE, "..", "targets", target + ".md")
    if not os.path.exists(path):
        return None
    for block in re.findall(r"```json\n(.*?)```", open(path, encoding="utf-8").read(), re.S):
        try:
            d = json.loads(block)
        except ValueError:
            continue
        if isinstance(d, dict) and ("top1_agreement_min" in d or "mean_kl_top20_max" in d):
            return d
    return None


def num(v):
    return f"{v:.5g}"


def band_errors(panel, band):
    """How panel is outside band, with the numbers and the band."""
    out = []
    lo, hi = band.get("top1_agreement_min"), band.get("mean_kl_top20_max")
    t1, kl = panel.get("top1_agreement"), panel.get("mean_kl_top20")
    if lo is not None and isinstance(t1, (int, float)) and t1 < lo:
        out.append(f"top1_agreement {num(t1)} below the band minimum {num(lo)}")
    if hi is not None and isinstance(kl, (int, float)) and kl > hi:
        out.append(f"mean_kl_top20 {num(kl)} above the band maximum {num(hi)}")
    return out


def check(path):
    errs, warns = [], []
    try:
        s = json.load(open(path))
    except Exception as e:
        return [f"not JSON: {e}"], []
    if os.path.getsize(path) > 1_000_000:
        errs.append("file over 1 MB")
    if s.get("schema") != "local-ai-recipe-kit/submission/v1":
        errs.append("schema must be local-ai-recipe-kit/submission/v1")
    for k in ("handle", "target", "launch", "probe", "quality", "speed"):
        if not s.get(k):
            errs.append(f"missing {k}")
    nm = s.get("needs_measured")
    if not isinstance(nm, dict) or not nm:
        problems = ["it is null or empty (run tools/memwatch.sh, pass --memwatch work/memwatch.csv)"]
    else:
        problems = []
        ram, gpu = nm.get("host_ram_drop_peak_gb"), nm.get("gpu_used_peak_mib")
        if not isinstance(ram, (int, float)) or ram <= 0:
            problems.append(f"host_ram_drop_peak_gb={ram!r} (the measured MemAvailable drop, must be > 0)")
        if not isinstance(gpu, (int, float)) or gpu <= 0:
            problems.append(f"gpu_used_peak_mib={gpu!r} (the measured GPU memory, must be > 0)")
    if problems:
        if before_rule(s):
            warns.append("needs_measured not enforced for a submission from before the rule (submitted "
                         f"{s.get('submitted')} < {RULE_FROM}): " + "; ".join(problems))
        else:
            errs.append("needs_measured: " + "; ".join(problems))
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
    # a token of any of the three kinds, a bare address or a personal path: the shapes the scrubber refuses
    bad = [m for m in SECRETISH.findall(txt) if m not in ("0.0.0.0", "127.0.0.1")]
    if bad:
        errs.append("looks like a secret/token, an address or a personal path: " + ", ".join(sorted(set(bad))[:5]))
    p = (s.get("quality") or {}).get("panel") or {}
    if not isinstance(p.get("top1_agreement"), (int, float)) or not isinstance(p.get("mean_kl_top20"), (int, float)):
        errs.append("quality.panel needs top1_agreement and mean_kl_top20 (run tools/score_ref_panel.py)")
    band = band_of(s["target"]) if s.get("target") in TARGETS else None
    if band:
        out = band_errors(p, band)
        if out:
            if before_rule(s):
                warns.append("band not enforced for a submission from before the rule (submitted "
                             f"{s.get('submitted')} < {RULE_FROM}): " + "; ".join(out))
            else:
                errs.append(f"quality.panel outside the target's band (targets/{s['target']}.md): " + "; ".join(out))
    sp = s.get("speed") or {}
    if sp.get("status") not in ("DONE",) and not str(sp.get("status", "")).startswith("EARLY_EXIT"):
        errs.append("speed.status must be DONE or EARLY_EXIT… (run tools/sweep.py)"
                    + (f"; the sweep failed: {sp['error']}" if sp.get("error") else ""))
    for n in ("8192", "16384", "32768"):
        v = (sp.get("prefill") or {}).get(n)
        if v is None:
            errs.append(f"speed.prefill missing {n}")
        elif not isinstance(v, dict) or not isinstance(v.get("median"), (int, float)) or not v.get("n"):
            why = f" (skipped: {v['skipped']})" if isinstance(v, dict) and v.get("skipped") else ""
            errs.append(f"speed.prefill[{n}] has no measurement{why} (a submission needs a measured median)")
    if "C1" not in (sp.get("decode") or {}) and sp.get("status") == "DONE":
        errs.append("speed.decode missing C1")
    card = os.path.basename(os.path.dirname(path))
    if card == "submissions":
        errs.append("put the file under submissions/<card id>/")
    else:
        sc = sp.get("card")
        if sc != card:
            detail = (f"speed.card {sc!r} does not match the directory {card!r}" if sc
                      else f"speed.card is missing (the sweep's --card); it must be {card!r}")
            if before_rule(s) and not sc:
                warns.append("speed.card not enforced for a submission from before the rule (submitted "
                             f"{s.get('submitted')} < {RULE_FROM}): " + detail)
            else:
                errs.append(detail)
    return errs, warns


def main():
    files = [f for a in (sys.argv[1:] or ["submissions/**/*.json"]) for f in glob.glob(a, recursive=True)]
    bad = 0
    for f in files:
        errs, warns = check(f)
        line = ("FAIL " if errs else "ok   ") + f
        if errs:
            line += "".join("\n  - " + x for x in errs)
        if warns:
            # the warnings are listed after the errors; a generator cannot be concatenated here
            line += ("\n  ! " + "\n  ! ".join(warns)) if errs else (" (" + "; ".join(warns) + ")")
        print(line)
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
