"""run the parser over every real cv file in data/raw and see if anything breaks.

this is the quick "does it all still work" check. it parses everything, counts
the wins and losses, groups things by electrode and electrolyte, and runs a
couple of sanity checks. handy to run after you change the parser.
"""

from __future__ import annotations

import sys
import traceback
from collections import Counter
from pathlib import Path

# the package lives in src, so add that to the path before importing it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cv_agent import Electrolyte, parse_cv_file, parse_directory

# this script sits in scripts/, so go up one level to find data/raw
DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    files = sorted(DATA.glob("*.csv"))
    print(f"found {len(files)} files in {DATA}")
    print()

    # try to parse each one. keep the good ones, note why the bad ones failed.
    successes: list = []
    failures: list[tuple[str, str]] = []

    for fp in files:
        try:
            exp = parse_cv_file(fp)
            successes.append(exp)
        except Exception as exc:
            failures.append((fp.name, f"{type(exc).__name__}: {exc}"))

    print(f"parsed ok: {len(successes)}")
    print(f"failed:    {len(failures)}")
    print()

    # if anything failed, list it out and stop here with a nonzero exit code
    if failures:
        print("here's what broke")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1

    # sort everything into buckets, one by electrode and one by electrolyte
    by_electrode: dict[str, list] = {}
    electrolyte_counts: Counter[Electrolyte] = Counter()
    for e in successes:
        by_electrode.setdefault(e.file_meta.electrode_id, []).append(e)
        electrolyte_counts[e.file_meta.electrolyte] += 1

    print("by electrolyte")
    for el, n in electrolyte_counts.items():
        print(f"  {el.value:14s} {n} files")
    print()

    print("by electrode, with the scan rates we saw for each")
    for eid in sorted(by_electrode):
        rates = sorted({e.file_meta.scan_rate_mv_s for e in by_electrode[eid]})
        print(f"  {eid}: {len(by_electrode[eid])} files at {rates} mV/s")
    print()

    # sanity checks. first, does the scan rate in the filename match the one the
    # machine wrote in the header. they always should, so any mismatch is a flag.
    print("sanity checks")
    mismatches = 0
    for e in successes:
        expected = e.file_meta.scan_rate_mv_s / 1000.0
        if abs(e.instrument_meta.scan_rate_v_s - expected) / expected > 1e-3:
            mismatches += 1
    print(f"  scan rate mismatches between filename and header: {mismatches}")

    # second, did any file come back with no data rows
    empty = sum(1 for e in successes if e.n_points == 0)
    print(f"  files with no data rows: {empty}")

    # finally print out one full experiment so we can eyeball that the fields
    # actually got filled in right
    print()
    print("here's the first file in full, just to eyeball it")
    sample = successes[0]
    print(f"  file:        {sample.file_meta.source_path.name}")
    print(f"  electrode:   {sample.file_meta.electrode_id}")
    print(f"  electrolyte: {sample.file_meta.electrolyte.value}")
    print(f"  scan rate:   {sample.instrument_meta.scan_rate_v_s} V/s")
    print(f"  E range:     {sample.instrument_meta.low_e_v} to {sample.instrument_meta.high_e_v} V")
    print(f"  data points: {sample.n_points}")
    print(f"  segments:    {len(sample.segment_results)}")
    for seg in sample.segment_results:
        print(f"    segment {seg.segment}, Ep={seg.ep_v} V, ip={seg.ip_a} A")

    return 0


if __name__ == "__main__":
    # wrap the whole thing so if something unexpected blows up we still get a
    # clean traceback and a sensible exit code
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
