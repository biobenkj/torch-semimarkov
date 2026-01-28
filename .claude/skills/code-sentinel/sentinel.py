#!/usr/bin/env python3
"""
Code Sentinel - Main CLI orchestrator for verification pipeline.

Unified interface for all sentinel operations in torch-semimarkov.

Usage:
    ./sentinel.py status                          # Show sentinel health
    ./sentinel.py verify --trace NAME             # Verify specific trace
    ./sentinel.py verify --all                    # Verify all traces
    ./sentinel.py pipeline                        # Full pre-commit pipeline
    ./sentinel.py retrace NAME                    # Interactive sentinel update
    ./sentinel.py install-hooks                   # Install git pre-commit hooks
    ./sentinel.py report --format json            # Generate verification report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Exit codes
EXIT_SUCCESS = 0
EXIT_ANCHOR_MISSING = 1
EXIT_ANCHOR_DRIFT = 2
EXIT_ANCHOR_AMBIGUOUS = 3
EXIT_CONSISTENCY_FAILED = 4
EXIT_ASSUMPTION_FAILED = 5
EXIT_GENERAL_ERROR = 10

# Importance patterns for init command (domain-specific for torch-semimarkov)
CRITICAL_PATTERNS = [
    r"def forward\(",
    r"def backward\(",
    r"\.apply\(",
    r"@triton\.jit",
    r"def semi_crf",
    r"def launch_",
]
HIGH_PATTERNS = [
    r"def __init__\(",
    r"logsumexp",
    r"NEG_INF",
    r"torch\.isfinite",
    r"torch\.isnan",
    r"torch\.clamp",
    r"def _score",
    r"def decode",
]


class Status(Enum):
    VERIFIED = "VERIFIED"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


@dataclass
class AnchorResult:
    name: str
    status: str  # VERIFIED, MISSING, DRIFT, AMBIGUOUS
    expected_line: int
    actual_line: int | None
    drift: int | None
    message: str


@dataclass
class AssumptionResult:
    id: str
    description: str
    passed: bool
    message: str


@dataclass
class TraceVerification:
    name: str
    commit_status: Status
    verified_commit: str
    current_commit: str | None
    uncommitted_changes: bool
    anchors: list[AnchorResult] = field(default_factory=list)
    assumptions: list[AssumptionResult] = field(default_factory=list)

    @property
    def overall_status(self) -> Status:
        if self.commit_status == Status.STALE:
            return Status.STALE
        if any(a.status != "VERIFIED" for a in self.anchors):
            return Status.DEGRADED
        if any(not a.passed for a in self.assumptions):
            return Status.DEGRADED
        return Status.VERIFIED


@dataclass
class AnchorImpact:
    """Impact assessment for an anchor after code changes."""

    anchor_name: str
    status: str  # unchanged, shifted, modified, deleted
    old_line: int
    new_line: int | None = None
    suggestion: str = ""


@dataclass
class FunctionInfo:
    """Extracted function/method information for init command."""

    name: str
    line: int
    importance: str  # critical, high, medium


class CodeSentinel:
    def __init__(self, sentinel_dir: Path | None = None):
        self.sentinel_dir = sentinel_dir or self._find_sentinel_dir()
        self.repo_root = self._find_repo_root()
        self.meta_path = self.sentinel_dir / ".sentinel-meta.yaml"
        self.anchors_path = self.sentinel_dir / "anchors" / "anchors.yaml"
        self.traces_dir = self.sentinel_dir / "traces"

    def _find_sentinel_dir(self) -> Path:
        """Find the code-sentinel skill directory."""
        # Check common locations
        candidates = [
            Path(__file__).parent,
            Path(".claude/skills/code-sentinel"),
            Path.cwd() / ".claude/skills/code-sentinel",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / ".sentinel-meta.yaml").exists():
                return candidate.resolve()
        raise FileNotFoundError("Could not find code-sentinel directory")

    def _find_repo_root(self) -> Path:
        """Find the git repository root."""
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=self.sentinel_dir,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        # Fallback: walk up from sentinel_dir
        current = self.sentinel_dir
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return self.sentinel_dir.parent.parent.parent.parent

    def load_meta(self) -> dict:
        """Load .sentinel-meta.yaml"""
        if not self.meta_path.exists():
            return {"version": "2.0", "sentinels": {}}
        return yaml.safe_load(self.meta_path.read_text())

    def load_anchors(self) -> dict:
        """Load anchors.yaml"""
        if not self.anchors_path.exists():
            return {}
        return yaml.safe_load(self.anchors_path.read_text())

    def get_current_commit(self, file_path: str) -> str | None:
        """Get the current commit hash for a file."""
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", file_path],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def has_uncommitted_changes(self, file_path: str) -> bool:
        """Check if file has uncommitted changes."""
        # Check staged changes
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", file_path],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        # Check unstaged changes
        unstaged = subprocess.run(
            ["git", "diff", "--name-only", "--", file_path],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        return bool(staged.stdout.strip() or unstaged.stdout.strip())

    def verify_anchor(
        self,
        pattern: str,
        expected_line: int,
        file_path: str,
        tolerance: int = 20,
        after: str | None = None,
    ) -> AnchorResult:
        """Verify a single anchor using verify-anchor.sh."""
        script = self.sentinel_dir / "anchors" / "verify-anchor.sh"
        cmd = [str(script), pattern, str(expected_line), file_path, str(tolerance)]
        if after:
            cmd.append(after)

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
        output = result.stdout.strip()

        # Parse output
        actual_line = None
        drift = None
        if "Line" in output:
            match = re.search(r"Line (\d+)", output)
            if match:
                actual_line = int(match.group(1))
        if "drift" in output:
            match = re.search(r"drift (\d+)", output)
            if match:
                drift = int(match.group(1))

        status_map = {0: "VERIFIED", 1: "MISSING", 2: "DRIFT", 3: "AMBIGUOUS"}
        return AnchorResult(
            name="",  # Will be set by caller
            status=status_map.get(result.returncode, "UNKNOWN"),
            expected_line=expected_line,
            actual_line=actual_line,
            drift=drift,
            message=output,
        )

    def verify_trace(self, trace_name: str, check_assumptions: bool = True) -> TraceVerification:
        """Run full verification for a trace."""
        meta = self.load_meta()
        anchors = self.load_anchors()

        trace_meta = meta.get("sentinels", {}).get(trace_name, {})
        if not trace_meta:
            return TraceVerification(
                name=trace_name,
                commit_status=Status.UNKNOWN,
                verified_commit="",
                current_commit=None,
                uncommitted_changes=False,
            )

        verified_commit = trace_meta.get("verified_commit", "")
        source_files = trace_meta.get("source_files", [])

        # Check commit status
        current_commit = None
        uncommitted = False
        commit_status = Status.VERIFIED

        for source_file in source_files:
            file_commit = self.get_current_commit(source_file)
            if file_commit:
                current_commit = file_commit
            if self.has_uncommitted_changes(source_file):
                uncommitted = True

        if current_commit and current_commit != verified_commit:
            commit_status = Status.STALE
        elif uncommitted:
            commit_status = Status.STALE

        # Verify anchors
        anchor_results = []
        trace_anchors = anchors.get(trace_name, {})
        for anchor_name, spec in trace_anchors.items():
            result = self.verify_anchor(
                pattern=spec["pattern"],
                expected_line=spec["expected_line"],
                file_path=str(self.repo_root / spec["file"]),
                tolerance=spec.get("drift_tolerance", 20),
                after=spec.get("after"),
            )
            result.name = anchor_name
            anchor_results.append(result)

        # Verify assumptions
        assumption_results = []
        if check_assumptions:
            assumption_results = self._verify_assumptions(trace_name, anchors)

        return TraceVerification(
            name=trace_name,
            commit_status=commit_status,
            verified_commit=verified_commit,
            current_commit=current_commit,
            uncommitted_changes=uncommitted,
            anchors=anchor_results,
            assumptions=assumption_results,
        )

    def _verify_assumptions(self, trace_name: str, anchors: dict) -> list[AssumptionResult]:
        """Verify mechanically-checkable assumptions for a trace."""
        trace_path = self.traces_dir / f"{trace_name}.md"
        if not trace_path.exists():
            return []

        content = trace_path.read_text()

        # Parse assumptions table (same regex as verify-assumptions.py)
        match = re.search(
            r"### Mechanically Verified\n\n"
            r".*?\n\n"
            r"\| ID \| Assumption \| Verification \|\n"
            r"\|[-|]+\|\n"
            r"((?:\| [A-Z][0-9]+ \|[^\n]+\|\n)+)",
            content,
            re.DOTALL,
        )
        if not match:
            return []

        results = []
        for line in match.group(1).strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3 and re.match(r"[A-Z]\d+", parts[0]):
                assumption_id = parts[0]
                description = parts[1]
                verification = parts[2]

                passed = True
                message = "Verified"

                if verification.startswith("anchor:"):
                    anchor_name = verification.split(":")[1].strip()
                    trace_anchors = anchors.get(trace_name, {})
                    if anchor_name in trace_anchors:
                        spec = trace_anchors[anchor_name]
                        anchor_result = self.verify_anchor(
                            pattern=spec["pattern"],
                            expected_line=spec["expected_line"],
                            file_path=str(self.repo_root / spec["file"]),
                            tolerance=spec.get("drift_tolerance", 20),
                            after=spec.get("after"),
                        )
                        passed = anchor_result.status == "VERIFIED"
                        message = anchor_result.message
                    else:
                        passed = False
                        message = f"Anchor {anchor_name} not defined"
                elif verification.startswith("`") and verification.endswith("`"):
                    # Shell command verification
                    cmd = verification[1:-1]
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, cwd=self.repo_root
                    )
                    passed = result.returncode == 0
                    message = "Command passed" if passed else "Command failed"

                results.append(
                    AssumptionResult(
                        id=assumption_id, description=description, passed=passed, message=message
                    )
                )

        return results

    def check_consistency(self) -> list[str]:
        """Check consistency between meta, anchors, and trace headers."""
        errors = []
        meta = self.load_meta()
        anchors = self.load_anchors()

        for trace_name, trace_meta in meta.get("sentinels", {}).items():
            # Check anchors exist in anchors.yaml
            for anchor_id in trace_meta.get("anchors", []):
                if trace_name not in anchors or anchor_id not in anchors.get(trace_name, {}):
                    errors.append(
                        f"{trace_name}: Anchor '{anchor_id}' in meta but not in anchors.yaml"
                    )

            # Check verified_commit matches trace header
            trace_path = self.traces_dir / f"{trace_name}.md"
            if trace_path.exists():
                trace_content = trace_path.read_text()
                match = re.search(
                    r"\*\*Verified against:\*\*.*?@ commit `([a-f0-9]+)`", trace_content
                )
                if match:
                    trace_commit = match.group(1)
                    meta_commit = trace_meta.get("verified_commit", "")
                    if trace_commit != meta_commit:
                        errors.append(
                            f"{trace_name}: Commit mismatch - meta='{meta_commit}', trace='{trace_commit}'"
                        )

        return errors

    def get_suggested_tests(self, modified_files: list[str]) -> list[str]:
        """Get suggested tests for modified files."""
        meta = self.load_meta()
        bindings = meta.get("test_bindings", {})
        tests = set()
        for f in modified_files:
            for pattern, test_list in bindings.items():
                if pattern in f:
                    tests.update(test_list)
        return sorted(tests)

    def get_modified_files(self) -> list[str]:
        """Get list of staged files."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []

    def extract_functions(self, source_path: Path) -> list[FunctionInfo]:
        """Extract function names and line numbers from a Python source file."""
        content = source_path.read_text()
        functions = []

        # Match function definitions with varying indentation
        pattern = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)

        for i, line in enumerate(content.splitlines(), 1):
            match = pattern.match(line)
            if match:
                indent = match.group(1)
                name = match.group(3)
                # Only include top-level or class methods (indent <= 4 spaces)
                if len(indent) <= 4:
                    # Determine importance
                    importance = "medium"
                    for pat in CRITICAL_PATTERNS:
                        if re.search(pat, line):
                            importance = "critical"
                            break
                    if importance == "medium":
                        for pat in HIGH_PATTERNS:
                            if re.search(pat, line):
                                importance = "high"
                                break
                    functions.append(FunctionInfo(name=name, line=i, importance=importance))

        return functions

    def extract_classes(self, source_path: Path) -> list[tuple[str, int]]:
        """Extract class names and line numbers from a Python source file."""
        content = source_path.read_text()
        classes = []

        pattern = re.compile(r"^class\s+(\w+)", re.MULTILINE)

        for i, line in enumerate(content.splitlines(), 1):
            match = pattern.match(line)
            if match:
                classes.append((match.group(1), i))

        return classes

    def get_files_changed_since(self, commit: str) -> set[str]:
        """Get set of files changed between commit and HEAD."""
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{commit}..HEAD"],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        if result.returncode == 0:
            return {f for f in result.stdout.strip().split("\n") if f}
        return set()

    def find_pattern_line(
        self, file_path: Path, pattern: str, after: str | None = None
    ) -> int | None:
        """Find the line number where a pattern occurs."""
        try:
            content = file_path.read_text()
            lines = content.splitlines()

            start_idx = 0
            if after:
                for i, line in enumerate(lines):
                    if after in line:
                        start_idx = i + 1
                        break

            for i, line in enumerate(lines[start_idx:], start_idx + 1):
                if pattern in line:
                    return i
            return None
        except Exception:
            return None

    def analyze_anchor_impacts(self, trace_name: str) -> list[AnchorImpact]:
        """Analyze how current HEAD affects trace anchors."""
        meta = self.load_meta()
        anchors = self.load_anchors()

        trace_meta = meta.get("sentinels", {}).get(trace_name, {})
        if not trace_meta:
            return []

        verified_commit = trace_meta.get("verified_commit", "")
        trace_anchors = anchors.get(trace_name, {})

        # Get changed files
        changed_files = self.get_files_changed_since(verified_commit)

        impacts = []
        for anchor_name, spec in trace_anchors.items():
            anchor_file = spec["file"]
            old_line = spec["expected_line"]

            # Check if the file was changed
            if anchor_file not in changed_files:
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="unchanged",
                        old_line=old_line,
                        new_line=old_line,
                    )
                )
                continue

            # Re-verify anchor at HEAD
            result = self.verify_anchor(
                pattern=spec["pattern"],
                expected_line=old_line,
                file_path=str(self.repo_root / anchor_file),
                tolerance=spec.get("drift_tolerance", 20),
                after=spec.get("after"),
            )

            if result.status == "MISSING":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="deleted",
                        old_line=old_line,
                        new_line=None,
                        suggestion="Pattern no longer found - manual review required",
                    )
                )
            elif result.status == "VERIFIED":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="unchanged",
                        old_line=old_line,
                        new_line=result.actual_line,
                    )
                )
            elif result.status == "DRIFT":
                # Check if this is just a line shift or semantic change
                # For now, treat all drifts as shifts (safe to auto-update)
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="shifted",
                        old_line=old_line,
                        new_line=result.actual_line,
                        suggestion=f"Update expected_line: {old_line} -> {result.actual_line}",
                    )
                )
            elif result.status == "AMBIGUOUS":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="modified",
                        old_line=old_line,
                        new_line=None,
                        suggestion="Pattern matches multiple lines - add 'after' disambiguator",
                    )
                )

        return impacts

    def save_anchors(self, anchors: dict) -> None:
        """Save anchors to anchors.yaml."""
        with open(self.anchors_path, "w") as f:
            yaml.dump(anchors, f, default_flow_style=False, sort_keys=False)


def cmd_status(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Show overall sentinel health."""
    meta = sentinel.load_meta()

    print("Code Sentinel Status")
    print("=" * 40)
    print(f"Last global verification: {meta.get('last_global_verification', 'Unknown')}")
    print()
    print("Sentinels:")

    for trace_name, trace_meta in meta.get("sentinels", {}).items():
        status = trace_meta.get("status", "UNKNOWN")
        symbol = "✓" if status == "VERIFIED" else "⚠" if status == "STALE" else "✗"
        print(f"  {symbol} {trace_name}: {status}")

    return EXIT_SUCCESS


def cmd_init(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Scaffold a new trace from a source file."""
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = sentinel.repo_root / source_path

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        return EXIT_GENERAL_ERROR

    # Determine trace name
    trace_name = args.name or source_path.stem

    # Check if trace already exists
    trace_path = sentinel.traces_dir / f"{trace_name}.md"
    if trace_path.exists() and not args.force:
        print(f"Error: Trace already exists: {trace_path}")
        print("Use --force to overwrite")
        return EXIT_GENERAL_ERROR

    # Extract functions and classes
    print(f"Analyzing {source_path}...")
    functions = sentinel.extract_functions(source_path)
    classes = sentinel.extract_classes(source_path)

    # Categorize by importance
    critical = [f for f in functions if f.importance == "critical"]
    high = [f for f in functions if f.importance == "high"]

    print(f"  Found {len(functions)} functions, {len(classes)} classes")
    print(f"  Critical: {len(critical)}, High: {len(high)}")

    # Compute relative path for the source file
    try:
        rel_source = source_path.relative_to(sentinel.repo_root)
    except ValueError:
        rel_source = source_path

    # Generate trace markdown
    trace_content = f"""# Sentinel: {trace_name}

**Verified against:** `{rel_source}` @ commit `TODO`

**Status:** DRAFT

**Linked tests:** TODO

## Summary

TODO: Describe the purpose and key functionality of this module.

## Active Assumptions

### Mechanically Verified

| ID | Assumption | Verification |
|----|------------|--------------|
"""

    # Add suggested assumptions for critical/high functions
    assumption_id = 1
    for func in critical[:5]:  # Limit to 5 critical functions
        trace_content += f"| A{assumption_id} | {func.name} exists at expected location | anchor: {func.name.upper()} |\n"
        assumption_id += 1

    trace_content += """
### Agent-Verified (on trace load)

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| TODO | TODO | TODO |

## Algorithm Flow

TODO: Document the key algorithm steps with line references.

"""

    # Add key functions section
    if critical or high:
        trace_content += "## Key Functions\n\n"
        trace_content += "| Function | Line | Importance |\n"
        trace_content += "|----------|------|------------|\n"
        for func in critical + high:
            trace_content += f"| `{func.name}` | {func.line} | {func.importance} |\n"
        trace_content += "\n"

    trace_content += """## Critical Invariants

- [ ] TODO: Document critical invariants

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| None documented | - | - |

## Version History

- **TODO**: Initial trace (DRAFT)
"""

    # Generate anchor suggestions
    anchors_content = f"\n# Anchors for {trace_name} (generated scaffold)\n{trace_name}:\n"
    for func in (critical + high)[:10]:  # Limit to 10 anchors
        anchor_name = func.name.upper()
        anchors_content += f"""  {anchor_name}:
    file: {rel_source}
    pattern: "def {func.name}("
    expected_line: {func.line}
    drift_tolerance: 30
"""

    # Generate meta entry suggestion
    meta_entry = f"""
# Add to .sentinel-meta.yaml under sentinels:
  {trace_name}:
    verified_commit: TODO
    source_files:
      - {rel_source}
    assumptions_mechanical: []
    assumptions_agent: []
    anchors: [{', '.join(f.name.upper() for f in (critical + high)[:10])}]
    linked_tests: []
    status: DRAFT
    depends_on: []
"""

    # Write trace file
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(trace_content)
    print(f"\nCreated trace: {trace_path}")

    # Show anchor suggestions
    print("\n=== Suggested Anchors ===")
    print("Add to anchors/anchors.yaml:")
    print(anchors_content)

    # Show meta entry suggestion
    print("\n=== Suggested Meta Entry ===")
    print(meta_entry)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Edit the trace file to complete TODO sections")
    print("  2. Add anchors to anchors/anchors.yaml")
    print("  3. Add entry to .sentinel-meta.yaml")
    print("  4. Update verified_commit and status when ready")
    print("  5. Run: ./sentinel.py verify --trace " + trace_name)
    print("=" * 60)

    return EXIT_SUCCESS


def cmd_verify(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Verify specific trace or all traces."""
    meta = sentinel.load_meta()
    exit_code = EXIT_SUCCESS

    traces_to_verify = []
    if args.all:
        traces_to_verify = list(meta.get("sentinels", {}).keys())
    elif args.trace:
        traces_to_verify = [args.trace]
    else:
        print("Error: Specify --trace NAME or --all")
        return EXIT_GENERAL_ERROR

    # Consistency check if requested
    if args.check_consistency:
        print("=== Consistency Check ===\n")
        errors = sentinel.check_consistency()
        if errors:
            for err in errors:
                print(f"  ✗ {err}")
            print(f"\nConsistency: {len(errors)} errors found\n")
            exit_code = EXIT_CONSISTENCY_FAILED
        else:
            print("  ✓ All metadata consistent\n")

    # Verify each trace
    for trace_name in traces_to_verify:
        result = sentinel.verify_trace(trace_name)
        _print_verification_result(result)

        if result.overall_status != Status.VERIFIED:
            if any(a.status == "MISSING" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_MISSING)
            elif any(a.status == "DRIFT" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)
            elif any(a.status == "AMBIGUOUS" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_AMBIGUOUS)
            elif any(not a.passed for a in result.assumptions):
                exit_code = max(exit_code, EXIT_ASSUMPTION_FAILED)

    return exit_code


def _print_verification_result(result: TraceVerification) -> None:
    """Print formatted verification result."""
    if result.overall_status == Status.VERIFIED:
        print(f"Grounded: {result.name} @ {result.verified_commit} ✓")
    else:
        print(f"Grounded: {result.name} @ {result.verified_commit}")

        if result.commit_status == Status.STALE:
            status_type = (
                "UNCOMMITTED_CHANGES" if result.uncommitted_changes else "COMMITTED_CHANGES"
            )
            if result.uncommitted_changes and result.current_commit != result.verified_commit:
                status_type = "BOTH"
            print(f"  Commit: STALE ({status_type})")
            print(
                f"    Verified: {result.verified_commit} | Current: {result.current_commit or 'unknown'}"
            )

        # Anchors summary
        verified = sum(1 for a in result.anchors if a.status == "VERIFIED")
        total = len(result.anchors)
        drifted = [a for a in result.anchors if a.status != "VERIFIED"]
        if drifted:
            print(f"  Anchors: {verified}/{total} verified, {len(drifted)} failed")
            for a in drifted:
                print(f"    ✗ {a.name}: {a.message}")

        # Assumptions summary
        if result.assumptions:
            failed = [a for a in result.assumptions if not a.passed]
            summary = ", ".join(f"{a.id} {'✓' if a.passed else '✗'}" for a in result.assumptions)
            print(f"  Assumptions: {summary}")
            if failed:
                for a in failed:
                    print(f"    ✗ {a.id}: {a.message}")

        print()
        print("⚠️  Cannot provide advice until sentinel is updated.")
    print()


def cmd_pipeline(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Run full pre-commit pipeline."""
    print("=" * 60)
    print("Code Sentinel Pipeline")
    print("=" * 60)
    print()

    exit_code = EXIT_SUCCESS
    meta = sentinel.load_meta()

    # Step 1: Get modified files
    if args.files:
        modified_files = args.files
    else:
        modified_files = sentinel.get_modified_files()

    if modified_files:
        print("Modified files:")
        for f in modified_files:
            print(f"  {f}")
        print()

    # Step 2: Consistency check
    print("=== Step 1: Consistency Check ===\n")
    errors = sentinel.check_consistency()
    if errors:
        for err in errors:
            print(f"  ✗ {err}")
        exit_code = EXIT_CONSISTENCY_FAILED
    else:
        print("  ✓ All metadata consistent")
    print()

    # Step 3: Verify all traces
    print("=== Step 2: Trace Verification ===\n")
    for trace_name in meta.get("sentinels", {}).keys():
        result = sentinel.verify_trace(trace_name)

        status_symbol = "✓" if result.overall_status == Status.VERIFIED else "✗"
        anchor_status = (
            f"{sum(1 for a in result.anchors if a.status == 'VERIFIED')}/{len(result.anchors)}"
        )
        assumption_status = (
            f"{sum(1 for a in result.assumptions if a.passed)}/{len(result.assumptions)}"
            if result.assumptions
            else "n/a"
        )

        print(f"  {status_symbol} {trace_name}")
        print(f"      Commit: {result.commit_status.value}")
        print(f"      Anchors: {anchor_status}")
        print(f"      Assumptions: {assumption_status}")

        if result.overall_status != Status.VERIFIED:
            if any(a.status != "VERIFIED" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)
            if any(not a.passed for a in result.assumptions):
                exit_code = max(exit_code, EXIT_ASSUMPTION_FAILED)
    print()

    # Step 4: Test advisory
    print("=== Step 3: Test Advisory ===\n")
    if modified_files:
        suggested_tests = sentinel.get_suggested_tests(modified_files)
        if suggested_tests:
            print("  Suggested tests for modified files:")
            print(f"    pytest {' '.join(suggested_tests)}")
            print()
            if os.environ.get("SENTINEL_RUN_TESTS") == "1":
                print("  Running tests...")
                test_result = subprocess.run(["pytest"] + suggested_tests, cwd=sentinel.repo_root)
                if test_result.returncode != 0:
                    exit_code = max(exit_code, EXIT_GENERAL_ERROR)
            else:
                print("  Run with: SENTINEL_RUN_TESTS=1 ./sentinel.py pipeline")
        else:
            print("  No sentinel-bound tests for modified files")
    else:
        print("  No staged files to check")
    print()

    # Summary
    print("=" * 60)
    if exit_code == EXIT_SUCCESS:
        print("Pipeline: PASSED ✓")
    else:
        print(f"Pipeline: FAILED (exit code {exit_code})")
    print("=" * 60)

    return exit_code


def cmd_retrace(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Interactive sentinel update."""
    trace_name = args.trace_name
    trace_path = sentinel.traces_dir / f"{trace_name}.md"

    if not trace_path.exists():
        print(f"Error: Trace not found: {trace_path}")
        return EXIT_GENERAL_ERROR

    # Get current state
    result = sentinel.verify_trace(trace_name)
    _print_verification_result(result)

    if args.diff_only:
        # Show git diff for source files
        meta = sentinel.load_meta()
        trace_meta = meta.get("sentinels", {}).get(trace_name, {})
        verified_commit = trace_meta.get("verified_commit", "")

        for source_file in trace_meta.get("source_files", []):
            print(f"\n=== Diff: {source_file} ===")
            print(f"From {verified_commit} to HEAD:\n")
            subprocess.run(
                ["git", "diff", f"{verified_commit}..HEAD", "--", source_file],
                cwd=sentinel.repo_root,
            )
        return EXIT_SUCCESS

    if args.auto:
        # Auto-detect anchor changes
        print("\n=== Auto-Retrace Analysis ===\n")
        impacts = sentinel.analyze_anchor_impacts(trace_name)

        if not impacts:
            print("No anchors found for this trace")
            return EXIT_SUCCESS

        # Group by status
        unchanged = [i for i in impacts if i.status == "unchanged"]
        shifted = [i for i in impacts if i.status == "shifted"]
        modified = [i for i in impacts if i.status == "modified"]
        deleted = [i for i in impacts if i.status == "deleted"]

        print("Anchor Impact Summary:")
        print(f"  Unchanged: {len(unchanged)}")
        print(f"  Shifted (auto-fixable): {len(shifted)}")
        print(f"  Modified (needs review): {len(modified)}")
        print(f"  Deleted (manual fix): {len(deleted)}")
        print()

        if shifted:
            print("Shifted anchors (safe to auto-update):")
            for impact in shifted:
                print(f"  {impact.anchor_name}: {impact.old_line} -> {impact.new_line}")

        if modified:
            print("\nModified anchors (manual review required):")
            for impact in modified:
                print(f"  ✗ {impact.anchor_name}: {impact.suggestion}")

        if deleted:
            print("\nDeleted anchors (pattern not found):")
            for impact in deleted:
                print(f"  ✗ {impact.anchor_name}: {impact.suggestion}")

        # Apply if requested
        if args.apply and shifted:
            print("\n=== Applying Safe Updates ===\n")
            anchors = sentinel.load_anchors()
            trace_anchors = anchors.get(trace_name, {})
            updated = 0

            for impact in shifted:
                if impact.anchor_name in trace_anchors and impact.new_line:
                    trace_anchors[impact.anchor_name]["expected_line"] = impact.new_line
                    print(f"  Updated {impact.anchor_name}: {impact.old_line} -> {impact.new_line}")
                    updated += 1

            if updated:
                sentinel.save_anchors(anchors)
                print(f"\nSaved {updated} anchor update(s) to anchors.yaml")
            else:
                print("\nNo updates applied")
        elif args.apply and not shifted:
            print("\nNo safe updates to apply")
        elif shifted:
            print(f"\nRun with --apply to update {len(shifted)} shifted anchor(s)")

        if modified or deleted:
            print("\n⚠️  Some anchors require manual attention")
            return EXIT_ANCHOR_DRIFT

        return EXIT_SUCCESS

    if args.anchors_only:
        # Auto-update anchor line numbers
        print("\nAuto-updating anchor line numbers...")
        anchors = sentinel.load_anchors()
        trace_anchors = anchors.get(trace_name, {})
        updated = 0

        for anchor_name, spec in trace_anchors.items():
            file_path = str(sentinel.repo_root / spec["file"])
            pattern = spec["pattern"]
            after = spec.get("after")

            # Find current line
            if after:
                cmd = f'grep -n "{after}" "{file_path}" | head -1 | cut -d: -f1'
                after_result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=sentinel.repo_root
                )
                if after_result.stdout.strip():
                    after_line = int(after_result.stdout.strip())
                    cmd = f'tail -n "+{after_line}" "{file_path}" | grep -n "{pattern}" | head -1 | cut -d: -f1'
                    grep_result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, cwd=sentinel.repo_root
                    )
                    if grep_result.stdout.strip():
                        actual_line = after_line + int(grep_result.stdout.strip()) - 1
                    else:
                        actual_line = None
                else:
                    actual_line = None
            else:
                cmd = f'grep -n "{pattern}" "{file_path}" | head -1 | cut -d: -f1'
                grep_result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=sentinel.repo_root
                )
                actual_line = (
                    int(grep_result.stdout.strip()) if grep_result.stdout.strip() else None
                )

            if actual_line and actual_line != spec["expected_line"]:
                print(f"  {anchor_name}: {spec['expected_line']} -> {actual_line}")
                spec["expected_line"] = actual_line
                updated += 1

        if updated:
            # Save updated anchors
            sentinel.save_anchors(anchors)
            print(f"\nUpdated {updated} anchor(s) in anchors.yaml")
        else:
            print("\nNo anchor updates needed")

        return EXIT_SUCCESS

    # Interactive mode
    print("\nInteractive retrace not yet implemented.")
    print("Please manually update:")
    print(f"  1. {trace_path}")
    print(f"  2. {sentinel.anchors_path}")
    print(f"  3. {sentinel.meta_path}")
    print("\nOr use:")
    print("  --auto         Analyze anchor impacts")
    print("  --auto --apply Apply safe updates")
    print("  --anchors-only Force update all anchor line numbers")

    return EXIT_SUCCESS


def cmd_install_hooks(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Install git pre-commit hooks via pre-commit framework."""
    # Check if pre-commit is installed
    result = subprocess.run(["which", "pre-commit"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: pre-commit not found. Install with: pip install pre-commit")
        return EXIT_GENERAL_ERROR

    # Check if .pre-commit-config.yaml exists
    config_path = sentinel.repo_root / ".pre-commit-config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        return EXIT_GENERAL_ERROR

    # Run pre-commit install
    result = subprocess.run(["pre-commit", "install"], cwd=sentinel.repo_root)
    if result.returncode == 0:
        print("Pre-commit hooks installed successfully")
        return EXIT_SUCCESS
    else:
        print("Failed to install pre-commit hooks")
        return EXIT_GENERAL_ERROR


def cmd_report(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Generate verification report."""
    meta = sentinel.load_meta()

    last_verification = meta.get("last_global_verification")
    if hasattr(last_verification, "isoformat"):
        last_verification = last_verification.isoformat()

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "version": meta.get("version", "unknown"),
        "last_global_verification": str(last_verification) if last_verification else None,
        "traces": {},
        "consistency_errors": sentinel.check_consistency(),
    }

    for trace_name in meta.get("sentinels", {}).keys():
        result = sentinel.verify_trace(trace_name)
        report["traces"][trace_name] = {
            "status": result.overall_status.value,
            "commit_status": result.commit_status.value,
            "verified_commit": result.verified_commit,
            "current_commit": result.current_commit,
            "uncommitted_changes": result.uncommitted_changes,
            "anchors": {
                "verified": sum(1 for a in result.anchors if a.status == "VERIFIED"),
                "total": len(result.anchors),
                "details": [
                    {
                        "name": a.name,
                        "status": a.status,
                        "expected": a.expected_line,
                        "actual": a.actual_line,
                    }
                    for a in result.anchors
                ],
            },
            "assumptions": {
                "passed": sum(1 for a in result.assumptions if a.passed),
                "total": len(result.assumptions),
                "details": [
                    {"id": a.id, "passed": a.passed, "message": a.message}
                    for a in result.assumptions
                ],
            },
        }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:  # markdown
        print("# Code Sentinel Report")
        print()
        print(f"Generated: {report['timestamp']}")
        print()
        print("## Summary")
        print()
        all_verified = all(t["status"] == "VERIFIED" for t in report["traces"].values())
        print(f"Overall Status: {'✓ VERIFIED' if all_verified else '✗ DEGRADED'}")
        print()
        print("## Traces")
        print()
        print("| Trace | Status | Commit | Anchors | Assumptions |")
        print("|-------|--------|--------|---------|-------------|")
        for name, data in report["traces"].items():
            status = "✓" if data["status"] == "VERIFIED" else "✗"
            anchors = f"{data['anchors']['verified']}/{data['anchors']['total']}"
            assumptions = f"{data['assumptions']['passed']}/{data['assumptions']['total']}"
            print(
                f"| {name} | {status} {data['status']} | {data['commit_status']} | {anchors} | {assumptions} |"
            )

        if report["consistency_errors"]:
            print()
            print("## Consistency Errors")
            print()
            for err in report["consistency_errors"]:
                print(f"- {err}")

    return EXIT_SUCCESS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Code Sentinel - Verification pipeline for torch-semimarkov",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./sentinel.py status                          Show sentinel health
  ./sentinel.py init src/module.py              Scaffold a new trace
  ./sentinel.py verify --trace triton-forward-k3plus
  ./sentinel.py verify --all --check-consistency
  ./sentinel.py pipeline                        Full pre-commit pipeline
  ./sentinel.py retrace NAME --auto             Analyze anchor impacts
  ./sentinel.py retrace NAME --auto --apply     Apply safe updates
  ./sentinel.py retrace NAME --anchors-only     Force update line numbers
  ./sentinel.py report --format json
        """,
    )
    parser.add_argument("--sentinel-dir", type=Path, help="Path to code-sentinel directory")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    subparsers.add_parser("status", help="Show sentinel health")

    # init
    init_parser = subparsers.add_parser("init", help="Scaffold a new trace from source file")
    init_parser.add_argument("source", help="Source file to analyze")
    init_parser.add_argument("--name", help="Trace name (default: source filename stem)")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing trace")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify traces")
    verify_parser.add_argument("--trace", help="Specific trace to verify")
    verify_parser.add_argument("--all", action="store_true", help="Verify all traces")
    verify_parser.add_argument(
        "--check-consistency", action="store_true", help="Check meta/anchor/trace consistency"
    )

    # pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pre-commit pipeline")
    pipeline_parser.add_argument(
        "--files", nargs="*", help="Specific files to check (default: staged files)"
    )
    pipeline_parser.add_argument("--ci", action="store_true", help="CI mode (non-interactive)")

    # retrace
    retrace_parser = subparsers.add_parser("retrace", help="Update a sentinel")
    retrace_parser.add_argument("trace_name", help="Name of trace to update")
    retrace_parser.add_argument("--auto", action="store_true", help="Auto-analyze anchor impacts")
    retrace_parser.add_argument(
        "--apply", action="store_true", help="Apply safe updates (use with --auto)"
    )
    retrace_parser.add_argument(
        "--anchors-only", action="store_true", help="Force update all anchor line numbers"
    )
    retrace_parser.add_argument(
        "--diff-only", action="store_true", help="Only show diff, don't update"
    )

    # install-hooks
    subparsers.add_parser("install-hooks", help="Install git pre-commit hooks")

    # report
    report_parser = subparsers.add_parser("report", help="Generate verification report")
    report_parser.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="Output format"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    try:
        sentinel = CodeSentinel(args.sentinel_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return EXIT_GENERAL_ERROR

    commands = {
        "status": cmd_status,
        "init": cmd_init,
        "verify": cmd_verify,
        "pipeline": cmd_pipeline,
        "retrace": cmd_retrace,
        "install-hooks": cmd_install_hooks,
        "report": cmd_report,
    }

    return commands[args.command](args, sentinel)


if __name__ == "__main__":
    sys.exit(main())
