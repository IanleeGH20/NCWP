#!/usr/bin/env python3
"""Check numeric/prose mismatches flagged by reviewers in LaTeX source."""
import re
import sys
from pathlib import Path

TEX = Path("/workspace/NCWP/ncwp_emnlp_final_revised (1).tex")

PATTERNS = [
    (r"\+15\.0", "Section 4.2: +15.0 claim vs Table 5 (+13.5)"),
    (r"\+14.?-.?21", "Introduction: +14-21 Spearman range"),
    (r"r\s*\\leq\s*80", "Section 4.2: r<=80 crossover claim"),
    (r"-0\.003-0\.008", "Anisotropy range notation"),
    (r"Grattafiori, Aaron and others", "Bibliography author list"),
    (r"Yang, An and others", "Bibliography author list"),
    (r"Random", "Tables 6/7: Random row mention"),
    (r"LPP", "Tables 6/7: LPP row mention"),
]


def main():
    if not TEX.exists():
        print(f"Missing: {TEX}")
        sys.exit(1)
    lines = TEX.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"Scanning {TEX} ({len(lines)} lines)\n")
    for pat, desc in PATTERNS:
        print(f"=== {desc} [{pat}] ===")
        found = False
        for i, line in enumerate(lines, 1):
            if re.search(pat, line):
                print(f"  L{i}: {line.strip()[:120]}")
                found = True
        if not found:
            print("  (no match)")
        print()


if __name__ == "__main__":
    main()
