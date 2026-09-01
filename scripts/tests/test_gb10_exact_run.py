#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "gb10-exact-run.py"
SPEC = importlib.util.spec_from_file_location("gb10_exact_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def git(path: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise AssertionError(process.stderr or process.stdout)
    return process.stdout.strip()


def make_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", "--initial-branch=main", str(path)], check=True)
    git(path, "config", "user.name", "Exact Runner Test")
    git(path, "config", "user.email", "exact-runner@example.invalid")
    for relative, contents in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "fixture")

    bare = path.parent / f"{path.name}-remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(bare)], check=True)
    git(path, "remote", "add", "origin", str(bare))
    git(path, "push", "--quiet", "--set-upstream", "origin", "main")


class ExactRunPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cohort = (Path(self.temporary.name) / "cohort").resolve()
        self.cohort.mkdir()

        make_repo(
            self.cohort / "primary",
            {
                "Cargo.toml": """
[workspace]
resolver = "2"
members = ["crates/*", "tools/member"]
exclude = ["crates/skipped"]

[workspace.dependencies]
shared = { path = "../shared" }

[patch.crates-io]
patched = { path = "../patched" }

[replace]
"replaced:0.1.0" = { path = "../replacement" }
""",
                "crates/app/Cargo.toml": """
[package]
name = "app"
version = "0.1.0"

[dependencies]
dep = { path = "../../../dep/crate", optional = true }

[target.'cfg(unix)'.build-dependencies]
target-leaf = { path = "../../../target-leaf" }
""",
                "crates/skipped/Cargo.toml": """
[package]
name = "skipped"
version = "0.1.0"
""",
                "tools/member/Cargo.toml": """
[package]
name = "tool"
version = "0.1.0"
""",
            },
        )
        make_repo(
            self.cohort / "dep",
            {
                "crate/Cargo.toml": """
[package]
name = "dep"
version = "0.1.0"

[dev-dependencies]
leaf = { path = "../../leaf" }
"""
            },
        )
        for name in ("leaf", "shared", "patched", "replacement", "target-leaf"):
            make_repo(
                self.cohort / name,
                {
                    "Cargo.toml": f"""
[package]
name = "{name}"
version = "0.1.0"
"""
                },
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_discovers_full_workspace_and_recursive_path_cohort(self) -> None:
        plan = RUNNER.discover(self.cohort / "primary")

        self.assertEqual(plan.root, self.cohort)
        self.assertEqual(plan.primary, "primary")
        self.assertEqual(
            {slot.relative for slot in plan.slots},
            {
                "dep",
                "leaf",
                "patched",
                "primary",
                "replacement",
                "shared",
                "target-leaf",
            },
        )
        self.assertEqual(plan.manifests, 10)

        encoded = json.dumps(
            plan.identity(), sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(plan.digest, __import__("hashlib").sha256(encoded).hexdigest())

    def test_refuses_tracked_and_untracked_changes(self) -> None:
        primary = self.cohort / "primary"
        manifest = primary / "Cargo.toml"
        original = manifest.read_text()
        manifest.write_text(original + "\n# dirty\n")
        with self.assertRaisesRegex(RUNNER.PlanError, "dirty/uncommitted"):
            RUNNER.discover(primary)

        manifest.write_text(original)
        (primary / "untracked.txt").write_text("not committed\n")
        with self.assertRaisesRegex(RUNNER.PlanError, "dirty/uncommitted"):
            RUNNER.discover(primary)

    def test_refuses_clean_but_unpushed_commit(self) -> None:
        primary = self.cohort / "primary"
        (primary / "README.md").write_text("local commit only\n")
        git(primary, "add", "README.md")
        git(primary, "commit", "--quiet", "-m", "not pushed")
        with self.assertRaisesRegex(RUNNER.PlanError, "needs an origin branch"):
            RUNNER.discover(primary)

    def test_refuses_cargo_paths_override(self) -> None:
        primary = self.cohort / "primary"
        config = primary / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text('paths = ["../local-override"]\n')
        git(primary, "add", ".cargo/config.toml")
        git(primary, "commit", "--quiet", "-m", "local Cargo override")
        git(primary, "push", "--quiet", "origin", "main")
        with self.assertRaisesRegex(RUNNER.PlanError, "Cargo paths override"):
            RUNNER.discover(primary)

    def test_plan_cli_never_contacts_host(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--plan",
                "definitely-not-a-real-spark",
                str(self.cohort / "primary"),
                "--",
                "cargo",
                "check",
                "-p",
                "app",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=10,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("host: definitely-not-a-real-spark", process.stdout)
        self.assertIn("command: cargo check -p app", process.stdout)

    def test_remote_runner_stages_exact_commits_and_preserves_argv(self) -> None:
        plan = RUNNER.discover(self.cohort / "primary")
        marker = "argument with spaces;$(not-a-shell)"
        code = """
import json
import os
from pathlib import Path
import sys
Path(os.environ["CARGO_TARGET_DIR"], "result.json").write_text(json.dumps({
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "target": os.environ["CARGO_TARGET_DIR"],
}))
"""
        payload = plan.payload([sys.executable, "-c", code, marker])
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode()
        home = (Path(self.temporary.name) / "remote-home").resolve()
        home.mkdir()
        environment = os.environ.copy()
        environment["HOME"] = str(home)
        process = subprocess.run(
            ["bash", "-s", "--", encoded],
            input=RUNNER.REMOTE,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

        stage = home / "gb10" / "exact-source" / plan.digest
        result = json.loads((stage / "target/result.json").read_text())
        self.assertEqual(result["argv"], [marker])
        self.assertEqual(Path(result["cwd"]), stage / "primary")
        self.assertEqual(Path(result["target"]), stage / "target")
        for slot in plan.slots:
            self.assertEqual(git(stage / slot.relative, "rev-parse", "HEAD"), slot.commit)

    def test_missing_siblings_resolve_through_main_worktree(self) -> None:
        primary = self.cohort / "primary"
        isolated = (Path(self.temporary.name) / "isolated").resolve()
        candidate = isolated / "primary"
        isolated.mkdir()
        git(primary, "worktree", "add", "--detach", str(candidate), "HEAD")
        try:
            self.assertFalse((isolated / "dep").exists())
            plan = RUNNER.discover(candidate)
        finally:
            git(primary, "worktree", "remove", "--force", str(candidate))

        self.assertEqual(plan.root, isolated)
        self.assertEqual(plan.primary, "primary")
        dep = next(slot for slot in plan.slots if slot.relative == "dep")
        self.assertEqual(dep.source, (self.cohort / "dep").resolve())
        self.assertEqual(dep.logical, isolated / "dep")


if __name__ == "__main__":
    unittest.main()
