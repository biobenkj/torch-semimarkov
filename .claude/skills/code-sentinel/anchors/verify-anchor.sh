#!/bin/bash
# Usage: verify-anchor.sh <pattern> <expected_line> <file> [drift_tolerance] [after_pattern]
# Returns: 0=verified, 1=missing, 2=drifted, 3=ambiguous

pattern="$1"
expected_line="$2"
file="$3"
drift_tolerance="${4:-20}"
after_pattern="${5:-}"

# If after_pattern provided, use two-stage matching
if [[ -n "$after_pattern" ]]; then
  # Find line number of after_pattern, then search for pattern after it
  after_line=$(grep -nF "$after_pattern" "$file" 2>/dev/null | head -1 | cut -d: -f1)
  if [[ -z "$after_line" ]]; then
    echo "ANCHOR_MISSING: Context pattern '$after_pattern' not found"
    exit 1
  fi
  # Search only after the context line
  actual_line=$(tail -n "+$after_line" "$file" | grep -nF "$pattern" | head -1 | cut -d: -f1)
  if [[ -n "$actual_line" ]]; then
    actual_line=$((after_line + actual_line - 1))
  fi
else
  # Check for ambiguous patterns (multiple matches)
  match_count=$(grep -cF "$pattern" "$file" 2>/dev/null)
  if [[ $match_count -gt 1 ]]; then
    echo "ANCHOR_AMBIGUOUS: Pattern matches $match_count lines"
    exit 3
  fi
  actual_line=$(grep -nF "$pattern" "$file" 2>/dev/null | head -1 | cut -d: -f1)
fi

if [[ -z "$actual_line" ]]; then
  echo "ANCHOR_MISSING: Pattern not found"
  exit 1
fi

drift=$((actual_line - expected_line))
drift=${drift#-}  # absolute value

if [[ $drift -gt $drift_tolerance ]]; then
  echo "ANCHOR_DRIFT: Expected ~$expected_line, found $actual_line (drift $drift > $drift_tolerance)"
  exit 2
fi

echo "ANCHOR_VERIFIED: Line $actual_line (drift $drift)"
exit 0
