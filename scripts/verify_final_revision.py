#!/usr/bin/env python3
"""
Verify the final revision-stage reconciliation artifacts.

This script does not rerun the complete EEG experiment. It checks the released
fresh-rerun audit ledgers that qualify the final manuscript after the complete
6,075-row revision-stage rerun.

Usage:
    python scripts/verify_final_revision.py --root .
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Publication repository root")
    args = ap.parse_args()
    root = Path(args.root)
    audit = root / "revision_audit"

    failures = []
    passed = 0

    def check(label, cond):
        nonlocal passed
        if cond:
            passed += 1
            print(f"PASS  {label}")
        else:
            failures.append(label)
            print(f"FAIL  {label}")

    p2 = read_csv(audit / "PHASE2_FRESH_AGGREGATE_CORRECTIONS.csv")
    p3 = read_csv(audit / "PHASE3_FINAL_MISMATCHES_29_FIELDS.csv")
    p3agg = read_csv(audit / "PHASE3_FRESH_SENSITIVITY_AGGREGATE_CORRECTIONS.csv")
    boot = read_csv(audit / "PHASE3_TABLE15_BOOTSTRAP_RECONCILIATION.csv")

    check("Phase-II display-correction ledger has 3 rows", len(p2) == 3)
    triples = {(r["quantity"], r["archived_display"], r["fresh_display"]) for r in p2}
    check("Phase-II 10% CRCS ASR is corrected 11.19 -> 11.18",
          ("CRCS targeted ASR (%)", "11.19", "11.18") in triples)
    check("Phase-II class-reference gain is corrected +1.80 -> +1.81",
          ("spectral-to-CRCS class-reference gain (pp)", "+1.80", "+1.81") in triples)

    check("Phase-III mismatch ledger has 29 fields", len(p3) == 29)
    keys = {
        (r["subject"], r["seed"], r["calibration_fraction"],
         r["scenario"], r["poison_rate"], r["removal_budget"])
        for r in p3
    }
    check("Phase-III mismatch ledger represents 8 scenario rows", len(keys) == 8)
    budgets = {round(float(r["removal_budget"]), 2) for r in p3}
    check("All Phase-III discrepancies are at B=0.20 or B=0.25",
          budgets == {0.20, 0.25})
    check("No Phase-III discrepancy is at primary B=0.15",
          all(not close(r["removal_budget"], 0.15) for r in p3))

    amap = {
        (int(r["calibration_percent"]), int(r["poison_percent"]),
         int(r["removal_budget_percent"])): (
            float(r["archived_asr_percent"]), float(r["fresh_asr_percent"])
        )
        for r in p3agg
    }
    expected = {
        (25,10,25):(10.987654320987653,11.028806584362137),
        (25,15,25):(14.499314128943755,14.55418381344307),
        (25,20,20):(69.28669410150891,69.31412894375858),
        (25,20,25):(25.48696844993141,25.404663923182436),
        (50,20,20):(59.87654320987654,59.89711934156379),
        (50,20,25):(18.60082304526749,18.559670781893008),
    }
    for key, vals in expected.items():
        check(f"Phase-III ASR correction {key}",
              key in amap and close(amap[key][0], vals[0]) and close(amap[key][1], vals[1]))

    check("Table-15 bootstrap ledger has 12 cells", len(boot) == 12)
    changed = [r for r in boot
               if r["ci_matches_at_2dp"].strip().lower() not in {"true","1","yes"}]
    check("Exactly 11 of 12 printed bootstrap CIs require correction", len(changed) == 11)
    unchanged = [r for r in boot
                 if r["ci_matches_at_2dp"].strip().lower() in {"true","1","yes"}]
    check("Exactly one printed bootstrap CI remains unchanged", len(unchanged) == 1)
    if unchanged:
        r = unchanged[0]
        check("Unchanged CI is calibration 25%, poison 5%",
              int(r["calibration_percent"]) == 25 and int(r["poison_percent"]) == 5
              and close(r["recomputed_ci_low_pp"], -84.91)
              and close(r["recomputed_ci_high_pp"], -34.68))

    print()
    print(f"Checks passed: {passed}")
    print(f"Checks failed: {len(failures)}")
    if failures:
        for item in failures:
            print("  -", item)
        return 1
    print("RESULT: FINAL REVISION AUDIT CONSISTENCY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
