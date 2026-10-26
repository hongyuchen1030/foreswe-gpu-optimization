"""Best-effort recovery of the deleted Sierra 43-station ForeSWE sources from Codex session transcripts.

The scripts under the legacy experiments/foreswe_sierra_station/scripts were removed from disk; the
Codex `apply_patch` payloads that created/edited them survive in ~/.codex/sessions. This extracts every
patch touching that folder (chronologically), writes the raw patches, materialises "Add File" bodies,
and replays "Update File" hunks onto the documented base files (the upstream copies they started from).
Anything that was changed outside apply_patch (cp/sed/python) cannot be recovered; failures are logged.

    python scripts/recover_sierra_sources.py --out experiments/sierra_subset/recovered_from_transcripts
"""
import argparse
import glob
import json
import os
import re

TARGET = "foreswe_sierra_station/"
PATCH_RE = re.compile(r"\*\*\* Begin Patch\n.*?\*\*\* End Patch", re.S)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# base files the Sierra scripts were copied from (git: upstream 393de07 + local edits 32951e0)
BASES = {"scripts/forecasting_att_daily_model.py": "fore_swe/forecasting_att_daily_model.py",
         "scripts/forecasting_daily_gp_model.py": "fore_swe/forecasting_daily_gp_model.py",
         "scripts/extended_dl_forecasting.py": "fore_swe/extended_dl_forecasting.py",
         "scripts/custom_encoder_model.py": "fore_swe/custom_encoder_model.py"}


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def parse_patch(text):
    """Yield (op, path, body_lines) for each file section of a Codex patch."""
    sections = re.split(r"^\*\*\* (Add|Update|Delete) File: (.+)$", text, flags=re.M)
    for i in range(1, len(sections) - 2, 3):
        body = sections[i + 2].split("\n")
        body = [l for l in body if not l.startswith("*** End Patch") and not l.startswith("*** Move to")]
        yield sections[i], sections[i + 1].strip(), body


def apply_update(src, body):
    """Apply @@-delimited hunks (context ' ', removal '-', addition '+') by unique text replacement."""
    hunks, cur = [], []
    for l in body:
        if l.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = []
        elif l[:1] in (" ", "-", "+"):
            cur.append(l)
        elif l == "":
            cur.append(" ")
    if cur:
        hunks.append(cur)
    for h in hunks:
        old = "\n".join(l[1:] for l in h if l[0] in " -")
        new = "\n".join(l[1:] for l in h if l[0] in " +")
        if old and src.count(old) >= 1:
            src = src.replace(old, new, 1)
        elif old:
            return None
        else:
            src = src.rstrip("\n") + "\n" + new + "\n"
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", default=os.path.expanduser("~/.codex/sessions"))
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/sierra_subset/recovered_from_transcripts"))
    a = ap.parse_args()
    patches = {}
    for path in sorted(glob.glob(os.path.join(a.sessions, "**", "*.jsonl"), recursive=True)):
        with open(path, errors="ignore") as f:
            for line in f:
                if TARGET not in line or "Begin Patch" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                ts = obj.get("timestamp", "")
                for s in strings(obj):
                    for p in PATCH_RE.findall(s):
                        if TARGET in p and "\n*** Begin Patch" not in p[1:]:
                            patches.setdefault(p, (ts, os.path.basename(path)))
    ordered = sorted(patches.items(), key=lambda kv: kv[1][0])
    os.makedirs(os.path.join(a.out, "raw_patches"), exist_ok=True)
    files, log = {}, []
    for i, (p, (ts, sess)) in enumerate(ordered):
        with open(os.path.join(a.out, "raw_patches", "{:03d}_{}.patch".format(i, ts.replace(":", ""))), "w") as f:
            f.write("# session {} at {}\n".format(sess, ts) + p + "\n")
        for op, fpath, body in parse_patch(p):
            if TARGET not in fpath:
                continue
            rel = fpath.split(TARGET, 1)[1]
            if op == "Add":
                files[rel] = "\n".join(l[1:] if l.startswith("+") else l for l in body).rstrip("\n") + "\n"
                log.append("{} add    {} ok".format(ts, rel))
            elif op == "Update":
                base = files.get(rel)
                if base is None and rel in BASES:
                    base = open(os.path.join(REPO, BASES[rel])).read()
                    log.append("{} base   {} <- {}".format(ts, rel, BASES[rel]))
                new = apply_update(base, body) if base is not None else None
                if new is None:
                    log.append("{} update {} FAILED (context not found)".format(ts, rel))
                else:
                    files[rel] = new
                    log.append("{} update {} ok".format(ts, rel))
            elif op == "Delete":
                files.pop(rel, None)
                log.append("{} delete {}".format(ts, rel))
    for rel, text in files.items():
        dst = os.path.join(a.out, "files", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)
    with open(os.path.join(a.out, "RECOVERY_LOG.txt"), "w") as f:
        f.write("patches found: {}\nfiles reconstructed: {}\n\n".format(len(ordered), len(files)) + "\n".join(log) + "\n")
    print("patches", len(ordered), "files", len(files))
    print("\n".join(log))


if __name__ == "__main__":
    main()
