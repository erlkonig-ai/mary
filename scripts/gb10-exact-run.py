#!/usr/bin/env python3
"""Run a command on one Spark from exact, fetchable Cargo source revisions.

Usage: gb10-exact-run.py [--plan] HOST PRIMARY_REPO -- COMMAND [ARG ...]

Every tracked Cargo.toml in each discovered Git repo is scanned. This is a
conservative superset of one package because `cargo -p` still parses the full
workspace and Cargo.lock may lag path edits.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import tomllib

DEPS = ("dependencies", "dev-dependencies", "build-dependencies")
BUSY_PATTERN = "inkling_forward|inkling_membw|tp_allreduce_probe|nsys|ncu|sglang|vllm|cargo|rustc"


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class Slot:
    source: Path
    logical: Path
    commit: str
    origin: str
    origin_ref: str
    relative: str = ""


@dataclass(frozen=True)
class Plan:
    root: Path
    primary: str
    slots: tuple[Slot, ...]
    manifests: int
    digest: str

    def identity(self):
        return {
            "version": 1,
            "primary": self.primary,
            "repositories": [
                {"path": slot.relative, "commit": slot.commit} for slot in self.slots
            ],
        }

    def payload(self, command):
        return {
            "identity": self.identity(),
            "digest": self.digest,
            "command": command,
            "repositories": [
                {
                    "path": s.relative,
                    "commit": s.commit,
                    "origin": s.origin,
                    "ref": s.origin_ref,
                }
                for s in self.slots
            ],
        }


def git(path, *args, check=True):
    p = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode:
        raise PlanError(p.stderr.strip() or f"git failed in {path}: {shlex.join(args)}")
    return p


def git_root(path):
    p = git(path, "rev-parse", "--show-toplevel", check=False)
    if p.returncode:
        raise PlanError(f"local Cargo source is not in Git: {path}")
    return Path(p.stdout.strip()).resolve()


def repository(path):
    common = Path(git(path, "rev-parse", "--git-common-dir").stdout.strip())
    return (common if common.is_absolute() else path / common).resolve()


def inspect(source, logical):
    status = git(
        source,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ).stdout
    if status:
        lines = status.rstrip().splitlines()
        raise PlanError(
            f"refusing dirty/uncommitted repository {source}:\n"
            + "\n".join(lines[:12])
            + ("\n..." if len(lines) > 12 else "")
        )
    gitlinks = [
        line.split("\t", 1)[-1]
        for line in git(source, "ls-files", "--stage").stdout.splitlines()
        if line.startswith("160000 ")
    ]
    if gitlinks:
        raise PlanError(f"submodules are unsupported in {source}: {', '.join(gitlinks)}")
    for name in (".cargo/config", ".cargo/config.toml"):
        config = source / name
        if config.is_file():
            try:
                with config.open("rb") as f:
                    cargo_config = tomllib.load(f)
            except tomllib.TOMLDecodeError as error:
                raise PlanError(f"cannot parse {config}: {error}") from error
            if cargo_config.get("paths"):
                raise PlanError(f"Cargo paths override is unsupported for exact staging: {config}")
    commit = git(source, "rev-parse", "HEAD^{commit}").stdout.strip().lower()
    origin = git(source, "remote", "get-url", "origin", check=False)
    refs = git(
        source,
        "for-each-ref",
        "--format=%(refname)",
        "--contains",
        "HEAD",
        "refs/remotes/origin",
    ).stdout.splitlines()
    prefix = "refs/remotes/origin/"
    branches = sorted(
        "refs/heads/" + ref[len(prefix):]
        for ref in refs
        if ref.startswith(prefix) and ref != prefix + "HEAD"
    )
    if origin.returncode or not branches:
        raise PlanError(
            f"{source} HEAD {commit} needs an origin branch; push and fetch it locally"
        )
    return Slot(source, logical, commit, origin.stdout.strip(), branches[0])


def worktrees(source):
    entries = []
    for line in git(source, "worktree", "list", "--porcelain").stdout.splitlines():
        if line.startswith("worktree "):
            entries.append(Path(line[9:]).resolve())
    return entries


def manifests(source):
    names = git(
        source, "ls-files", "-z", "--", "Cargo.toml", ":(glob)**/Cargo.toml"
    ).stdout.split("\0")
    return [Path(name) for name in sorted(set(names) - {""})]


def path_specs(source_manifest):
    try:
        with source_manifest.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PlanError(f"cannot parse tracked {source_manifest}: {error}") from error
    tables = [data.get(name) for name in DEPS]
    if isinstance(data.get("workspace"), dict):
        tables.append(data["workspace"].get("dependencies"))
    for target in data.get("target", {}).values():
        if isinstance(target, dict):
            tables.extend(target.get(name) for name in DEPS)
    if isinstance(data.get("patch"), dict):
        tables.extend(data["patch"].values())
    tables.append(data.get("replace"))
    for table in tables:
        if not isinstance(table, dict):
            continue
        for name, spec in table.items():
            if isinstance(spec, dict) and "path" in spec:
                raw = spec["path"]
                if not isinstance(raw, str) or Path(raw).is_absolute():
                    raise PlanError(f"{source_manifest}: {name!r} has non-relative path {raw!r}")
                yield raw


def resolve_dependency(slot, manifest_name, raw):
    logical_manifest = slot.logical / manifest_name
    logical_target = (logical_manifest.parent / raw / "Cargo.toml").resolve()
    candidates = [slot.source, *[p for p in worktrees(slot.source) if p != slot.source]]
    source_target = None
    for checkout in candidates:
        candidate = checkout / manifest_name.parent / raw / "Cargo.toml"
        if candidate.is_file():
            source_target = candidate.resolve()
            break
    if source_target is None:
        raise PlanError(f"{logical_manifest}: missing path dependency {raw!r} in every worktree")
    owner = git_root(source_target.parent)
    inside = source_target.relative_to(owner)
    if git(owner, "ls-files", "--error-unmatch", inside.as_posix(), check=False).returncode:
        raise PlanError(f"path dependency manifest is not committed: {source_target}")
    logical_owner = logical_target.parents[len(inside.parts) - 1]
    if (
        owner == slot.source
        or (
            repository(owner) == repository(slot.source)
            and logical_target.is_relative_to(slot.logical)
        )
    ):
        logical_owner = slot.logical
    return owner, logical_owner, repository(owner)


def _discover(primary):
    try:
        primary = primary.expanduser().resolve(strict=True)
    except OSError as error:
        raise PlanError(f"primary repository does not exist: {primary}") from error
    if git_root(primary) != primary or not (primary / "Cargo.toml").is_file():
        raise PlanError(f"primary must be a Cargo Git top level: {primary}")
    primary_logical = worktrees(primary)[0]
    primary_repository = repository(primary)
    pending = [(primary, primary_logical)]
    # Linked worktrees may intentionally occupy distinct slots (cubecl-fork and
    # cubecl-graph); common-dir identity only adjudicates competing slot claims.
    slots, claims = {}, {primary_logical: primary_repository}
    count = 0
    while pending:
        source, logical = pending.pop()
        logical = logical.resolve()
        if logical in slots:
            if repository(slots[logical].source) != repository(source):
                raise PlanError(f"two source repositories map to {logical}")
            continue
        slot = inspect(source, logical)
        slots[logical] = slot
        names = manifests(source)
        if not names:
            raise PlanError(f"no tracked Cargo.toml in {source}")
        count += len(names)
        for name in names:
            for raw in path_specs(source / name):
                owner, logical_owner, owner_repository = resolve_dependency(slot, name, raw)
                if logical_owner in slots:
                    if repository(slots[logical_owner].source) != owner_repository:
                        raise PlanError(f"two source repositories map to {logical_owner}")
                    continue
                claimed = claims.get(logical_owner)
                if claimed is not None and claimed != owner_repository:
                    raise PlanError(f"two source repositories map to {logical_owner}")
                if claimed is None:
                    claims[logical_owner] = owner_repository
                    pending.append((owner, logical_owner))
    for slot in slots.values():
        if git(slot.source, "status", "--porcelain=v1", "--untracked-files=all").stdout:
            raise PlanError(f"repository changed while planning: {slot.source}")
        if git(slot.source, "rev-parse", "HEAD^{commit}").stdout.strip() != slot.commit:
            raise PlanError(f"HEAD changed while planning: {slot.source}")
    root = Path(os.path.commonpath([str(path) for path in slots]))
    if root in slots:
        root = root.parent
    ordered = tuple(
        Slot(s.source, s.logical, s.commit, s.origin, s.origin_ref,
             Path(os.path.relpath(s.logical, root)).as_posix())
        for s in sorted(slots.values(), key=lambda item: str(item.logical))
    )
    primary_relative = Path(os.path.relpath(primary_logical, root)).as_posix()
    identity = {
        "version": 1,
        "primary": primary_relative,
        "repositories": [{"path": s.relative, "commit": s.commit} for s in ordered],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Plan(root, primary_relative, ordered, count, digest)


def discover(primary):
    try:
        return _discover(primary)
    except OSError as error:
        raise PlanError(f"cannot inspect Cargo cohort path: {error}") from error


REMOTE = r'''#!/bin/bash
set -euo pipefail
python3 - "$1" <<'PY'
import base64, json, os, shutil, signal, subprocess, sys, tempfile
from pathlib import Path
p = json.loads(base64.urlsafe_b64decode(sys.argv[1])); ident = p["identity"]
base = Path.home()/"gb10"/"exact-source"; stage = base/p["digest"]
base.mkdir(parents=True, exist_ok=True)
def run(a, **kw): return subprocess.run(a, text=True, **kw)
def verify():
  print(f"verified revisions on {os.uname().nodename}:", flush=True)
  for s in ident["repositories"]:
    repo=stage/s["path"]; head=run(["git","-C",repo,"rev-parse","HEAD^{commit}"],capture_output=True)
    dirty=run(["git","-C",repo,"status","--porcelain=v1"],capture_output=True)
    if head.stdout.strip()!=s["commit"] or dirty.returncode or dirty.stdout: raise RuntimeError(f"invalid stage: {s['path']}")
    print(f"  {s['path']}\t{head.stdout.strip()}", flush=True)
if stage.exists():
  if json.loads((stage/"cohort.json").read_text()) != ident: raise RuntimeError(f"stage mismatch: {stage}")
else:
  tmp=Path(tempfile.mkdtemp(prefix=f".{p['digest']}.",dir=base))
  try:
    for s in sorted(p["repositories"],key=lambda x:(len(Path(x["path"]).parts),x["path"])):
      repo=tmp/s["path"]; repo.mkdir(parents=True,exist_ok=True); run(["git","init","--quiet",repo],check=True)
      env=os.environ|{"GIT_TERMINAL_PROMPT":"0","GIT_SSH_COMMAND":"ssh -o BatchMode=yes"}
      f=run(["git","-C",repo,"fetch","--quiet","--no-tags",s["origin"],f"+{s['ref']}:refs/remotes/exact/source"],capture_output=True,env=env)
      if f.returncode: raise RuntimeError(f"cannot fetch {s['ref']} for {s['path']}: "+(f.stderr or "no diagnostic").replace(s["origin"],"<origin>").strip())
      if run(["git","-C",repo,"cat-file","-e",f"{s['commit']}^{{commit}}"],capture_output=True).returncode: raise RuntimeError(f"{s['ref']} does not contain required {s['commit']} for {s['path']}; push it")
      run(["git","-C",repo,"checkout","--quiet","--detach",s["commit"]],check=True)
    (tmp/"cohort.json").write_text(json.dumps(ident,sort_keys=True)+"\n"); tmp.rename(stage)
  except BaseException: shutil.rmtree(tmp,ignore_errors=True); raise
verify(); target=stage/"target"; target.mkdir(exist_ok=True)
env=os.environ|{"CARGO_TARGET_DIR":str(target),"GB10_EXACT_SOURCE":str(stage),"PATH":str(Path.home()/".cargo/bin")+os.pathsep+os.environ.get("PATH","")}
print(f"exact source: {stage}\nCARGO_TARGET_DIR: {target}",flush=True)
child=None
def forward(sig,_):
  if child is not None and child.poll() is None:
    try: os.killpg(child.pid,sig)
    except ProcessLookupError: pass
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,forward)
child=subprocess.Popen(p["command"],cwd=stage/ident["primary"],env=env,start_new_session=True)
raise SystemExit(child.wait())
PY
'''


def call(script, action, host, tag=""):
    argv = [str(script), action, host]
    if tag:
        argv.append(tag)
    return subprocess.run(
        argv, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def report(label, process):
    output = "\n".join(x.strip() for x in (process.stdout, process.stderr) if x.strip())
    if output:
        print(f"{label}: {output}", file=sys.stderr, flush=True)


def execute(plan, host, command, heartbeat):
    if host.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:%+\-\[\]]+", host):
        raise PlanError(f"unsafe SSH host token: {host!r}")
    try:
        timeout = int(os.environ.get("GB10_LOCK_TIMEOUT_S", "5400"))
    except ValueError as error:
        raise PlanError("GB10_LOCK_TIMEOUT_S must be an integer") from error
    if heartbeat * 2 >= timeout:
        raise PlanError(f"heartbeat {heartbeat}s is not safely below lock timeout {timeout}s")
    scripts = Path(__file__).resolve().parent
    busy = call(scripts / "lib/box-busy.sh", "--remote", host, BUSY_PATTERN)
    report("idle inspection", busy)
    if busy.returncode != 1:
        raise PlanError(f"{host} is busy or unreachable; refusing its lock")
    lock = scripts / "gb10-lock.sh"
    tag = f"exact-{plan.digest[:12]}-{os.getpid()}-{secrets.token_hex(4)}"
    taken = call(lock, "take", host, tag); report("lock take", taken)
    if taken.returncode:
        raise PlanError(f"could not take lock on {host} (exit {taken.returncode})")
    encoded = base64.urlsafe_b64encode(
        json.dumps(plan.payload(command), separators=(",", ":")).encode()
    ).decode()
    argv = ["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",host,"bash","-s","--",encoded]
    process, safe = None, True
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE); safe = False
        process.stdin.write(REMOTE.encode()); process.stdin.close()
        next_beat = time.monotonic() + heartbeat
        while process.poll() is None:
            time.sleep(min(1, max(0, next_beat-time.monotonic())))
            if time.monotonic() < next_beat: continue
            beat = call(lock, "refresh", host, tag); report("lock refresh", beat)
            if beat.returncode:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise PlanError("heartbeat failed; remote termination is unconfirmed")
            next_beat = time.monotonic() + heartbeat
        result = process.wait(); safe = 0 <= result < 255; return result
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill()
        return 130
    finally:
        if safe:
            released = call(lock, "release", host, tag); report("lock release", released)
        else:
            print(f"lock retained: verify remote process, then release tag {tag} on {host}",file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("host"); parser.add_argument("primary", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command[:1] == ["--"]: args.command.pop(0)
    if not args.command or args.heartbeat_seconds < 1: parser.error("command after -- and positive heartbeat required")
    try:
        plan = discover(args.primary)
        print(f"cohort: sha256:{plan.digest}\nhost: {args.host}\nlayout root: {plan.root}")
        print(f"primary: {plan.primary}\nmanifests scanned: {plan.manifests}")
        for s in plan.slots: print(f"  {s.relative}\t{s.commit}\t{s.origin_ref}\tfrom {s.source}")
        print(f"command: {shlex.join(args.command)}")
        sys.stdout.flush()
        return 0 if args.plan else execute(plan,args.host,args.command,args.heartbeat_seconds)
    except PlanError as error:
        print(f"gb10-exact-run: {error}",file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
