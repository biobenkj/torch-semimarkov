#!/usr/bin/env python3
"""Verify mechanically-checkable assumptions for a trace.

Supports two sources:
1. assumptions.yaml (preferred) - structured verification specs
2. Trace markdown tables (fallback) - for backwards compatibility
"""
import re
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
TRACES_DIR = SCRIPT_DIR / "traces"
ANCHORS_DIR = SCRIPT_DIR / "anchors"
ASSUMPTIONS_PATH = ANCHORS_DIR / "assumptions.yaml"


def load_assumptions_yaml(trace_name: str) -> list[dict] | None:
    """Load assumptions from assumptions.yaml. Returns None if not found."""
    if not ASSUMPTIONS_PATH.exists():
        return None

    data = yaml.safe_load(ASSUMPTIONS_PATH.read_text())
    if not data or "assumptions" not in data:
        return None

    trace_assumptions = data["assumptions"].get(trace_name)
    if not trace_assumptions:
        return None

    return trace_assumptions


def parse_assumptions_markdown(trace_path: Path) -> list[dict]:
    """Parse Mechanically Verified assumptions table from trace markdown (fallback)."""
    content = trace_path.read_text()

    # Find the Mechanically Verified section
    match = re.search(
        r"### Mechanically Verified\n\n"
        r".*?\n\n"  # Description text
        r"\| ID \| Assumption \| Verification \|\n"
        r"\|[-|]+\|\n"
        r"((?:\| [A-Z][0-9]+ \|[^\n]+\|\n)+)",
        content,
        re.DOTALL,
    )
    if not match:
        return []

    assumptions = []
    for line in match.group(1).strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 3 and re.match(r"[A-Z]\d+", parts[0]):
            # Convert markdown format to YAML-like structure
            verification = parts[2]
            if verification.startswith("anchor:"):
                spec = {
                    "type": "anchor",
                    "ref": verification.split(":")[1].strip(),
                }
            elif verification.startswith("`") and verification.endswith("`"):
                spec = {
                    "type": "shell",
                    "command": verification[1:-1],
                }
            else:
                spec = {"type": "manual", "guidance": verification}

            assumptions.append(
                {
                    "id": parts[0],
                    "description": parts[1],
                    "verification": spec,
                    "mechanical": spec["type"] != "manual",
                }
            )

    return assumptions


def verify_assumption(assumption: dict, anchors: dict, trace_name: str) -> tuple[bool, str]:
    """Verify a single assumption. Returns (passed, message)."""
    if not assumption.get("mechanical", True):
        return True, "Manual verification required"

    verification = assumption["verification"]
    vtype = verification.get("type", "manual")

    if vtype == "anchor":
        anchor_name = verification["ref"]
        trace_anchors = anchors.get(trace_name, {})
        if anchor_name not in trace_anchors:
            return False, f"Anchor {anchor_name} not defined in anchors.yaml"

        spec = trace_anchors[anchor_name]
        cmd = [
            str(ANCHORS_DIR / "verify-anchor.sh"),
            spec["pattern"],
            str(spec["expected_line"]),
            str(REPO_ROOT / spec["file"]),
            str(spec.get("drift_tolerance", 20)),
        ]
        if spec.get("after"):
            cmd.append(spec["after"])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        return result.returncode == 0, result.stdout.strip()

    elif vtype == "pattern":
        file_path = REPO_ROOT / verification["file"]
        pattern = verification["regex"]
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        # Use grep -E for regex matching
        result = subprocess.run(
            ["grep", "-E", pattern, str(file_path)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            return True, f"Pattern found: {result.stdout.strip()[:60]}..."
        return False, "Pattern not found"

    elif vtype == "shell":
        cmd = verification["command"]
        result = subprocess.run(cmd, shell=True, capture_output=True, cwd=REPO_ROOT)
        return result.returncode == 0, (
            "Command passed" if result.returncode == 0 else "Command failed"
        )

    elif vtype == "test_passes":
        test = verification["test"]
        result = subprocess.run(
            ["pytest", "-xvs", test],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.returncode == 0, (
            "Test passed" if result.returncode == 0 else f"Test failed: {result.stdout[-200:]}"
        )

    elif vtype == "manual":
        guidance = verification.get("guidance", "No guidance provided")
        return True, f"Manual: {guidance}"

    return True, f"Unknown verification type: {vtype}"


def main(trace_name: str) -> int:
    trace_path = TRACES_DIR / f"{trace_name}.md"
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}")
        return 1

    anchors_path = ANCHORS_DIR / "anchors.yaml"
    if not anchors_path.exists():
        print(f"Anchors file not found: {anchors_path}")
        return 1

    anchors = yaml.safe_load(anchors_path.read_text())

    # Try YAML first, fall back to markdown
    assumptions = load_assumptions_yaml(trace_name)
    source = "assumptions.yaml"
    if assumptions is None:
        assumptions = parse_assumptions_markdown(trace_path)
        source = "trace markdown (fallback)"

    if not assumptions:
        print(f"No assumptions found for {trace_name}")
        return 0

    print(f"=== Assumption Verification: {trace_name} ===")
    print(f"Source: {source}\n")

    failed = 0
    mechanical_count = 0
    manual_count = 0

    for assumption in assumptions:
        is_mechanical = assumption.get("mechanical", True)
        if is_mechanical:
            mechanical_count += 1
        else:
            manual_count += 1

        passed, msg = verify_assumption(assumption, anchors, trace_name)
        status = "✓" if passed else "✗"
        mech_tag = "" if is_mechanical else " [manual]"
        print(f"  {status} {assumption['id']}: {assumption['description']}{mech_tag}")
        print(f"      {msg}")
        if not passed and is_mechanical:
            failed += 1

    print(f"\nMechanical: {mechanical_count - failed}/{mechanical_count} verified")
    print(f"Manual: {manual_count} (require agent judgment)")
    print(f"\nResult: {'VERIFIED' if failed == 0 else 'DEGRADED'}")
    if failed > 0:
        print(f"Failed mechanical assumptions: {failed}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify-assumptions.py <trace-name>")
        print("\nAvailable traces:")
        for trace in sorted(TRACES_DIR.glob("*.md")):
            print(f"  {trace.stem}")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
