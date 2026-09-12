"""Cross-script contracts for the PowerShell maintenance entry points.

Two real defects motivated these, both of which fail silently or confusingly on
an operator's machine rather than in a test run:

- ``rebuild-knowledge-base.ps1`` refreshed the cosine sidecar and, on failure,
  only wrote a warning. PowerShell exits 0 when a script ends without a
  terminating error, so a caller (CI, or a chained import script) saw success
  while every query went on reporting the index as stale.
- A wrapper passed ``-KnowledgeBase`` to ``rebuild-cybersec.ps1``, which has no
  such parameter. PowerShell does not reject an unknown parameter for a
  positional-bindable script by name at parse time; it prompts for the missing
  mandatory parameter instead, so an unattended run hangs on stdin.

These assertions are pure text: they never spawn PowerShell, so they stay fast
and deterministic.
"""
import re
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def declared_parameters(text: str) -> set:
    """Parameters a script declares, read from its ``param(...)`` block only."""
    head = text.split("$ErrorActionPreference", 1)[0]
    block = head.split("param(", 1)[1] if "param(" in head else ""
    return set(re.findall(r"\[(?:string|switch|string\[\])\]\$(\w+)", block))


def invocations(text: str, binding: str) -> list:
    """Arguments of every ``& $binding ...`` call, with continuations joined."""
    joined = text.replace("`\r\n", " ").replace("`\n", " ")
    return re.findall(rf"&\s*\${binding}\b((?:[^\n]*))", joined)


class ScriptParameterContractTests(unittest.TestCase):
    def _declared(self) -> dict:
        return {
            path.name: declared_parameters(path.read_text(encoding="utf-8-sig"))
            for path in SCRIPTS.glob("*.ps1")
        }

    def test_every_script_declares_a_param_block(self):
        declared = self._declared()
        self.assertTrue(declared, "no PowerShell scripts found")
        for name, parameters in declared.items():
            self.assertTrue(parameters, f"{name} declares no parameters")

    def test_calls_only_pass_parameters_the_target_declares(self):
        declared = self._declared()
        checked = 0
        for path in SCRIPTS.glob("*.ps1"):
            text = path.read_text(encoding="utf-8-sig")
            # Script paths are Windows-style, so the capture must tolerate
            # backslashes: an earlier `[\\w.-]+\\.ps1` never matched and the
            # whole assertion ran vacuously.
            bindings = dict(
                re.findall(r'\$(\w+)\s*=\s*Join-Path[^\n]*?"[^"]*?([\w.-]+\.ps1)"', text)
            )
            for binding, target in bindings.items():
                self.assertIn(target, declared, f"{path.name} references {target}")
                for arguments in invocations(text, binding):
                    checked += 1
                    passed = set(re.findall(r"-(\w+)", arguments))
                    unknown = passed - declared[target]
                    self.assertFalse(
                        unknown,
                        f"{path.name} passes {sorted(unknown)} to {target}, "
                        f"which declares {sorted(declared[target])}",
                    )
        self.assertGreater(checked, 0, "no cross-script invocation was inspected")


class SidecarFailureTests(unittest.TestCase):
    def test_a_failed_sidecar_refresh_is_a_terminating_error(self):
        # A warning leaves the script's exit code at 0, and an operator or CI
        # job reading it concludes the index tree is usable while retrieval
        # rejects it as stale.
        text = (SCRIPTS / "rebuild-knowledge-base.ps1").read_text(encoding="utf-8-sig")
        sidecar = text.split("build_cosine", 1)[1]
        self.assertNotIn(
            "Write-Warning",
            sidecar,
            "the sidecar refresh must fail the script, not warn and exit 0",
        )
        self.assertIn("throw", sidecar)

    def test_the_sidecar_command_targets_the_requested_knowledge_base(self):
        text = (SCRIPTS / "rebuild-knowledge-base.ps1").read_text(encoding="utf-8-sig")
        command = text.split("build_cosine", 1)[1]
        for flag in ("--kb-root", "--knowledge-base"):
            self.assertIn(flag, command)


class RequirementPinTests(unittest.TestCase):
    """The pinned lock must stay installable on the platforms it claims.

    Windows-only distributions were pinned without their environment marker, so
    `pip install -r requirements.txt` failed outright on Linux and macOS (no
    wheel exists for pywin32 there). Nothing in the Windows test run could
    notice, because on Windows the pin is correct either way.
    """

    REQUIREMENTS = SCRIPTS.parent / "requirements.txt"

    def _pins(self):
        from packaging.requirements import Requirement

        lines = self.REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        return [
            Requirement(line.strip())
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_every_pin_parses_and_is_version_locked(self):
        pins = self._pins()
        self.assertTrue(pins, "requirements.txt has no pins")
        for pin in pins:
            self.assertTrue(pin.specifier, f"{pin.name} is not pinned to a version")

    def test_windows_only_distributions_are_marked(self):
        from packaging.markers import default_environment

        windows_only = {"pywin32", "pypiwin32", "pywinpty"}
        for pin in self._pins():
            if pin.name.lower() not in windows_only:
                continue
            self.assertIsNotNone(
                pin.marker,
                f"{pin.name} is Windows-only and must carry its environment marker, "
                "or the lock cannot install on Linux/macOS",
            )
            environment = dict(default_environment())
            environment["sys_platform"] = "linux"
            self.assertFalse(
                pin.marker.evaluate(environment),
                f"{pin.name} must not be installed on Linux",
            )


if __name__ == "__main__":
    unittest.main()
