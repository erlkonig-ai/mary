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


def make_repo(
    path: Path,
    files: dict[str, str],
    symlinks: dict[str, str] | None = None,
) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", "--initial-branch=main", str(path)], check=True)
    git(path, "config", "user.name", "Exact Runner Test")
    git(path, "config", "user.email", "exact-runner@example.invalid")
    for relative, contents in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)
    for relative, target in (symlinks or {}).items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)
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
                "Cargo.toml": """
[package]
name = "dep-root"
version = "0.1.0"
""",
                "crate/Cargo.toml": """
[package]
name = "dep"
version = "0.1.0"

[dependencies]
primary-back-edge = { path = "../../primary" }

[dev-dependencies]
leaf = { path = "../../leaf" }
""",
                "bench/Cargo.toml": """
[package]
name = "dep-bench"
version = "0.1.0"

[dependencies]
dep-root = { path = "subjects/current" }
"""
            },
            symlinks={"bench/subjects/current": "../.."},
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

    def run_remote_probe(
        self,
        inherited: dict[str, str] | None = None,
        command_prefix: list[str] | None = None,
        arguments: list[str] | None = None,
    ) -> tuple[RUNNER.Plan, Path, dict[str, object]]:
        plan = RUNNER.discover(self.cohort / "primary")
        home = (Path(self.temporary.name) / "remote-home").resolve()
        cargo_bin = home / ".cargo/bin"
        cargo_bin.mkdir(parents=True)
        observed = [
            "PILE",
            "PERSONA",
            "ORIENT_PILE",
            "ORIENT_PERSONA",
            "TELEMETRY_PILE",
            "TELEMETRY_COLLECTION_NAME",
            "DRIVE_MEMORY_PILE",
            "TRIBLESPACE_KEY",
            "TRIBLES_SIGNING_KEY",
            "TRIBLES_ORDER_KEY",
            "TRIBLESPACE_COLLECTION_WIKI",
            "TRIBLESPACE_COLLECTION_FUTURE_KIND",
            "TRIBLESPACE_METADATA_PILE",
            "PLAYGROUND_JAIL_HOST",
            "PLAYGROUND_FUTURE_STATE",
            "NOMIC_TEXT_PILE",
            "RUSTFLAGS",
            "CUDA_HOME",
            "SSH_AUTH_SOCK",
        ]
        code = f"""
import json
import os
from pathlib import Path
import sys
Path(os.environ["CARGO_TARGET_DIR"], "result.json").write_text(json.dumps({{
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "path": os.environ["PATH"],
    "target": os.environ["CARGO_TARGET_DIR"],
    "environment": {{name: os.environ[name] for name in {observed!r} if name in os.environ}},
}}))
"""
        probe = cargo_bin / "exact-path-probe"
        probe.write_text(f"#!{sys.executable}\n{code.lstrip()}")
        probe.chmod(0o755)
        command = [*(command_prefix or []), probe.name, *(arguments or [])]
        payload = plan.payload(command)
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode()
        environment = os.environ.copy()
        environment["HOME"] = str(home)
        environment.update(inherited or {})
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
        return plan, stage, result

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
        self.assertEqual(plan.manifests, 12)

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

    def test_remote_runner_stages_exact_commits_and_finds_cargo_bin(self) -> None:
        marker = "argument with spaces;$(not-a-shell)"
        plan, stage, result = self.run_remote_probe(arguments=[marker])
        self.assertEqual(result["argv"], [marker])
        self.assertEqual(Path(result["cwd"]), stage / "primary")
        self.assertEqual(
            result["path"].split(os.pathsep)[0], str(stage.parents[2] / ".cargo/bin")
        )
        self.assertEqual(Path(result["target"]), stage / "target")
        for slot in plan.slots:
            self.assertEqual(git(stage / slot.relative, "rev-parse", "HEAD"), slot.commit)

    def test_remote_command_scrubs_application_selectors_only(self) -> None:
        selectors = {
            "PILE": "/live/self.pile",
            "PERSONA": "ambient-persona",
            "ORIENT_PILE": "/live/orient.pile",
            "ORIENT_PERSONA": "ambient-persona",
            "TELEMETRY_PILE": "/live/telemetry.pile",
            "TELEMETRY_COLLECTION_NAME": "live-telemetry",
            "DRIVE_MEMORY_PILE": "/live/memory.pile",
            "TRIBLESPACE_KEY": "/live/self.key",
            "TRIBLES_SIGNING_KEY": "/live/legacy.key",
            "TRIBLES_ORDER_KEY": "live-order",
            "TRIBLESPACE_COLLECTION_WIKI": "aa" * 32,
            "TRIBLESPACE_COLLECTION_FUTURE_KIND": "bb" * 32,
            "TRIBLESPACE_METADATA_PILE": "/live/metadata.pile",
            "PLAYGROUND_JAIL_HOST": "live-jail.example.invalid",
            "PLAYGROUND_FUTURE_STATE": "live-playground-state",
        }
        ordinary = {
            "NOMIC_TEXT_PILE": "/models/nomic-text.pile",
            "RUSTFLAGS": "--cfg exact_runner_environment_probe",
            "CUDA_HOME": "/opt/cuda-exact-runner-probe",
            "SSH_AUTH_SOCK": "/tmp/exact-runner-agent.sock",
        }

        _, _, result = self.run_remote_probe(selectors | ordinary)

        environment = result["environment"]
        self.assertTrue(selectors.keys().isdisjoint(environment))
        for name, value in ordinary.items():
            self.assertEqual(environment[name], value)

    def test_remote_command_can_explicitly_restore_scrubbed_selector(self) -> None:
        _, _, result = self.run_remote_probe(
            {"PILE": "/ambient/must-not-win.pile"},
            command_prefix=[
                "env",
                "PILE=/fixture/explicit.pile",
                "PERSONA=fixture-persona",
                f"TRIBLESPACE_COLLECTION_WIKI={'cc' * 32}",
            ],
        )

        environment = result["environment"]
        self.assertEqual(environment["PILE"], "/fixture/explicit.pile")
        self.assertEqual(environment["PERSONA"], "fixture-persona")
        self.assertEqual(environment["TRIBLESPACE_COLLECTION_WIKI"], "cc" * 32)

    def test_arbitrary_primary_worktree_uses_canonical_topology(self) -> None:
        primary = self.cohort / "primary"
        isolated = (Path(self.temporary.name) / "isolated").resolve()
        candidate = isolated / "renamed-primary-candidate"
        isolated.mkdir()
        git(primary, "worktree", "add", "--detach", str(candidate), "HEAD")
        try:
            self.assertFalse((isolated / "dep").exists())
            plan = RUNNER.discover(candidate)
            same_repository_slots = sum(
                RUNNER.repository(slot.source) == RUNNER.repository(candidate)
                for slot in plan.slots
            )
        finally:
            git(primary, "worktree", "remove", "--force", str(candidate))

        self.assertEqual(plan.root, self.cohort)
        self.assertEqual(plan.primary, "primary")
        selected_primary = next(slot for slot in plan.slots if slot.relative == "primary")
        self.assertEqual(selected_primary.source, candidate)
        self.assertEqual(same_repository_slots, 1)
        dep = next(slot for slot in plan.slots if slot.relative == "dep")
        self.assertEqual(dep.source, (self.cohort / "dep").resolve())
        self.assertEqual(dep.logical, self.cohort / "dep")
        self.assertFalse(any("subjects" in slot.relative for slot in plan.slots))


if __name__ == "__main__":
    unittest.main()
