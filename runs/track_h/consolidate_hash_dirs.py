#!/usr/bin/env python3
"""Resolve `<canonical>_<8hex>` collision-fork dirs back to the canonical name.

Why this exists: the R4 seed0 patch (2026-07-23) was dispatched a few minutes BEFORE the
§1.7 identity guard was committed, so its long-running cells execute the OLD legacy
collision guard.  When those cells persist an arm whose canonical dir still holds a
PRE-patch config (extra provenance keys absent), the legacy guard writes to
`<name>_<hash>` instead of overwriting -- exactly the §1.7 false-fork.  The NUMBERS are
identical to a post-guard run (the commit changed only persist-time naming, not the
FL/scoring/T2 path, and ShapleyFL-beta is unused in R4), so the fix is purely to rename:
promote the freshest run to the canonical name, drop the stale siblings.  This is what
`RunLogger(..., identity=IDENTITY)` would have done had the guard been live at launch.

Rule: group every dir by its canonical name (strip a trailing _<8 lowercase-hex>); within
a group the NEWEST config.yaml mtime wins -> canonical; the rest are removed (their
numbers are either identical re-runs or superseded pre-patch versions, and every prior
version is in git history regardless).

Safety: dry-run by default.  Only groups whose members AGREE on the §1.7 identity fields
are touched -- a group that disagrees is reported and SKIPPED (that would be a real
name-collision between different experiments, not a fork, and must be looked at by hand).

    python runs/track_h/consolidate_hash_dirs.py            # dry-run
    python runs/track_h/consolidate_hash_dirs.py --apply    # execute
"""
import json
import os
import re
import shutil
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rundirs_llm")
HASH_RE = re.compile(r"^(?P<base>.+)_(?P<h>[0-9a-f]{8})$")
# The track_g rundir identity (mirror of experiments/track_g.py IDENTITY).
IDENTITY = ("track", "regime", "threat", "noisy_rate", "arm", "seed", "scale", "model")


def _cfg(d):
    try:
        with open(os.path.join(ROOT, d, "config.yaml")) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _ident(cfg):
    return tuple((k, cfg.get(k)) for k in IDENTITY)


def _mtime(d):
    try:
        return os.path.getmtime(os.path.join(ROOT, d, "config.yaml"))
    except OSError:
        return 0.0


def _sha(d):
    try:
        with open(os.path.join(ROOT, d, "meta.json")) as f:
            return (json.load(f).get("git_sha") or "")[:7]
    except Exception:
        return "?"


def main(apply):
    names = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    # Canonical name of a dir = itself, unless it is a <base>_<hash> whose <base> ALSO
    # exists as a dir (else the 8-hex tail is just part of the name, e.g. "v2_seed0").
    canon = {}
    for d in names:
        m = HASH_RE.match(d)
        canon[d] = m.group("base") if (m and m.group("base") in names) else d
    groups = {}
    for d in names:
        groups.setdefault(canon[d], []).append(d)

    forks = {c: ds for c, ds in groups.items() if len(ds) > 1}
    if not forks:
        print("포크 없음 — 정리할 해시 디렉터리가 없습니다.")
        return
    moved = removed = skipped = 0
    for c, ds in sorted(forks.items()):
        idents = {d: _ident(_cfg(d)) for d in ds}
        if len({v for v in idents.values()}) > 1:
            print(f"⚠ SKIP {c}: 정체성 불일치(수동 확인 필요)")
            for d in ds:
                print(f"    {d}  sha={_sha(d)}  ident={dict(idents[d])}")
            skipped += 1
            continue
        winner = max(ds, key=_mtime)                       # freshest write wins
        losers = [d for d in ds if d != winner]
        wtag = f"{winner}  (sha={_sha(winner)}, newest)"
        print(f"● {c}\n    keep: {wtag}")
        for d in losers:
            print(f"    drop: {d}  (sha={_sha(d)})")
        if apply:
            for d in losers:
                shutil.rmtree(os.path.join(ROOT, d))
                removed += 1
            if winner != c:                                # promote hash -> canonical
                shutil.move(os.path.join(ROOT, winner), os.path.join(ROOT, c))
                moved += 1
    verb = "적용됨" if apply else "DRY-RUN (적용하려면 --apply)"
    print(f"\n[{verb}] 그룹 {len(forks)}  승격 {moved}  제거 {removed}  스킵 {skipped}")
    if not apply:
        print("주의: 실행 중인 셀이 남아 있으면 완주 후에 --apply 하세요"
              " (noisy_obs_t2 = 마지막 포크 생성원).")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
