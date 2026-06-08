"""parses every cv file, runs the agent, writes reports.

  1. parse every cv file in data/raw
  2. group files by electrode
  3. for each electrode
       a. ask the agent which analyses to run (decide)
       b. run them
       c. ask the agent to write a short interpretation (interpret)
       d. plot everything and write an html report
  4. write a summary csv across all electrodes
  5. write an index.html linking to every electrode's report

uses claude if ANTHROPIC_API_KEY is set in the environment. 
without the key it falls back to deterministic rules so the script still works.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

# package is in src, so add it to the path before importing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cv_agent import parse_directory
from cv_agent.agent import compare, decide, interpret
from cv_agent.analysis import (
    CdlResult,
    LavironResult,
    NicholsonResult,
    RandlesSevcikResult,
    cdl,
    laviron,
    nicholson,
    randles_sevcik,
)
from cv_agent.models import CVExperiment, Electrolyte
from cv_agent.report import (
    plot_cdl,
    plot_cv_overlay,
    plot_laviron,
    plot_nicholson,
    plot_randles_sevcik,
    write_electrode_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"


def process_electrode(electrode_id: str, experiments: list[CVExperiment],
                      out_dir: Path
                      ) -> tuple[RandlesSevcikResult | None, CdlResult | None,
                                 LavironResult | None, str, list[str], int]:
    """run all the analyses the agent decides on for one electrode and write the report.

    returns the three result objects (any of which can be None), plus the
    agent's interpretation summary and flags.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # step 1, agent decides what to run
    decision = decide(experiments)
    print(f"  agent picked: {sorted(decision.analyses_to_run)}")
    if decision.reasoning:
        print(f"  because: {decision.reasoning}")

    # split experiments by electrolyte so we can pass the right ones to each analysis
    ferri = [e for e in experiments if e.file_meta.electrolyte == Electrolyte.FERRO_FERRI]
    pbs = [e for e in experiments if e.file_meta.electrolyte == Electrolyte.PBS]

    # always make a cv overlay plot of whatever we have. prefer ferro/ferri
    # since it's the more visually interesting one.
    cv_traces = ferri if ferri else pbs
    cv_plot = plot_cv_overlay(
        cv_traces,
        out_dir / f"{electrode_id}_cv_overlay.png",
        title=f"{electrode_id} CV at all scan rates",
    )

    # step 2, run whichever analyses the agent asked for
    rs_res: RandlesSevcikResult | None = None
    rs_plot_path: Path | None = None
    if "randles_sevcik" in decision.analyses_to_run and ferri:
        rs_res = randles_sevcik(ferri)
        rs_plot_path = plot_randles_sevcik(rs_res, out_dir / f"{electrode_id}_randles_sevcik.png")

    cdl_res: CdlResult | None = None
    cdl_plot_path: Path | None = None
    if "cdl" in decision.analyses_to_run and pbs:
        cdl_res = cdl(pbs)
        cdl_plot_path = plot_cdl(cdl_res, out_dir / f"{electrode_id}_cdl.png")

    lav_res: LavironResult | None = None
    lav_plot_path: Path | None = None
    if "laviron" in decision.analyses_to_run and ferri:
        lav_res = laviron(ferri)
        lav_plot_path = plot_laviron(lav_res, out_dir / f"{electrode_id}_laviron.png")
        
    nic_res: NicholsonResult | None = None
    nic_plot_path: Path | None = None
    if "nicholson" in decision.analyses_to_run and ferri:
        nic_res = nicholson(ferri)
        nic_plot_path = plot_nicholson(nic_res, out_dir / f"{electrode_id}_nicholson.png")

    # step 3, agent writes an interpretation
    # pull size from the first file (all files for one electrode share a size)
    size_mm = experiments[0].file_meta.electrode_size_mm
    interp = interpret(electrode_id, rs_res, cdl_res, lav_res, size_mm=size_mm)

    # step 4, stitch it all into one html page
    write_electrode_report(
        electrode_id=electrode_id,
        cv_plot=cv_plot,
        rs=rs_res, rs_plot=rs_plot_path,
        cdl_res=cdl_res, cdl_plot=cdl_plot_path,
        lav=lav_res, lav_plot=lav_plot_path,
        nic=nic_res, nic_plot=nic_plot_path,
        out_path=out_dir / f"{electrode_id}_report.html",
    )

    return rs_res, cdl_res, lav_res, interp.summary, interp.flags, size_mm

def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """one big csv across every electrode, for cross comparison in excel."""
    if not rows:
        return
    fieldnames = [
        "electrode_id",
        "n_files",
        "electrode_size_mm",
        "easa_mm2",
        "rs_r2_anodic",
        "rs_r2_cathodic",
        "cdl_uF",
        "cdl_r2",
        "laviron_k0_per_s",
        "laviron_alpha",
        "agent_summary",
        "agent_flags",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # tell the user up front whether the agent is on or off
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("agent mode: ON (ANTHROPIC_API_KEY is set, using claude)")
    else:
        print("agent mode: OFF (no ANTHROPIC_API_KEY, using deterministic rules)")
    print()

    print(f"reading cv files from {DATA_DIR}")
    experiments = parse_directory(DATA_DIR, skip_on_error=False)
    print(f"parsed {len(experiments)} files")
    print()

    # group by electrode so we can analyze one at a time
    by_electrode: dict[str, list[CVExperiment]] = defaultdict(list)
    for e in experiments:
        by_electrode[e.file_meta.electrode_id].append(e)

    summary_rows: list[dict] = []
    for electrode_id in sorted(by_electrode):
        exps = by_electrode[electrode_id]
        print(f"--- {electrode_id} ({len(exps)} files) ---")
        try:
            rs, cdl_res, lav, summary, flags, size_mm = process_electrode(
                electrode_id, exps, OUTPUT_DIR
            )
        except Exception:
            print(f"  error on {electrode_id}")
            traceback.print_exc()
            continue

        # collect for the cross electrode summary csv
        row: dict = {"electrode_id": electrode_id, "n_files": len(exps)}
        row["electrode_size_mm"] = size_mm
        if rs is not None:
            row["easa_mm2"] = rs.easa_mean_cm2 * 1e2
            row["rs_r2_anodic"] = rs.anodic_fit.r_squared
            row["rs_r2_cathodic"] = rs.cathodic_fit.r_squared
            print(f"  EASA: {rs.easa_mean_cm2*1e2:.4f} mm² "
                  f"(R² anodic={rs.anodic_fit.r_squared:.4f}, "
                  f"cathodic={rs.cathodic_fit.r_squared:.4f})")
        if cdl_res is not None:
            row["cdl_uF"] = cdl_res.cdl_microfarads
            row["cdl_r2"] = cdl_res.fit.r_squared
            print(f"  Cdl: {cdl_res.cdl_microfarads:.4f} µF "
                  f"(R²={cdl_res.fit.r_squared:.4f})")
        if lav is not None:
            row["laviron_k0_per_s"] = lav.k0_per_s
            row["laviron_alpha"] = lav.alpha
            if lav.k0_per_s is not None:
                print(f"  Laviron: k0 = {lav.k0_per_s:.3e} s⁻¹, alpha = {lav.alpha:.3f}")
            else:
                print(f"  Laviron: k0 unresolved (see report for notes)")

        row["agent_summary"] = summary
        row["agent_flags"] = " | ".join(flags) if flags else ""
        print(f"  agent says: {summary}")
        if flags:
            for flag in flags:
                print(f"  flag: {flag}")
        summary_rows.append(row)
        print()

    write_summary_csv(summary_rows, OUTPUT_DIR / "summary.csv")

    # cross electrode comparison. the agent looks at the whole batch at once,
    # spots patterns, calls out outliers, suggests what to look at next.
    print("--- batch comparison ---")
    comparison = compare(summary_rows)
    print(f"overview: {comparison.overview}")
    if comparison.patterns:
        print("patterns")
        for p in comparison.patterns:
            print(f"  - {p}")
    if comparison.outliers:
        print("outliers")
        for o in comparison.outliers:
            print(f"  - {o}")
    if comparison.recommendations:
        print("recommendations")
        for r in comparison.recommendations:
            print(f"  - {r}")
    print()

    # write the batch report html. its own page so it can be linked separately.
    batch_lines = ["<!doctype html><html><head><meta charset='utf-8'>",
                   "<title>Batch Comparison</title>",
                   "<style>body{font-family:system-ui;max-width:850px;margin:2em auto;padding:0 1em;color:#222;line-height:1.5;}",
                   "h1{border-bottom:2px solid #444;padding-bottom:0.3em;}",
                   "h2{margin-top:1.6em;color:#224;}",
                   ".overview{background:#f4f6fa;padding:1em 1.4em;border-radius:6px;border-left:4px solid #446;}",
                   "li{margin:0.4em 0;}",
                   ".outlier{background:#fff4dd;padding:0.6em 1em;border-radius:4px;border-left:3px solid #c80;margin:0.4em 0;}",
                   "</style></head><body>",
                   "<h1>Batch Comparison</h1>",
                   f"<div class='overview'>{comparison.overview}</div>"]
    if comparison.patterns:
        batch_lines.append("<h2>Patterns</h2><ul>")
        for p in comparison.patterns:
            batch_lines.append(f"<li>{p}</li>")
        batch_lines.append("</ul>")
    if comparison.outliers:
        batch_lines.append("<h2>Outliers</h2>")
        for o in comparison.outliers:
            batch_lines.append(f"<div class='outlier'>{o}</div>")
    if comparison.recommendations:
        batch_lines.append("<h2>Recommendations</h2><ul>")
        for r in comparison.recommendations:
            batch_lines.append(f"<li>{r}</li>")
        batch_lines.append("</ul>")
    batch_lines.append("<p><a href='index.html'>back to electrode index</a></p>")
    batch_lines.append("</body></html>")
    (OUTPUT_DIR / "batch.html").write_text("\n".join(batch_lines), encoding="utf-8")

    # the index page now also links to the batch comparison report
    index_html = ["<!doctype html><html><head><meta charset='utf-8'>",
                  "<title>CV-Agent Reports</title>",
                  "<style>body{font-family:system-ui;max-width:700px;margin:2em auto;padding:0 1em;}",
                  "li{margin:0.3em 0;}h1{border-bottom:2px solid #444;}",
                  ".batch{background:#eef4ff;padding:0.8em 1em;border-radius:6px;margin:1em 0;}</style>",
                  "</head><body><h1>Electrode Reports</h1>",
                  "<div class='batch'><b><a href='batch.html'>Batch Comparison →</a></b> "
                  "cross electrode patterns, outliers, and recommendations from the agent.</div>",
                  "<h2>Per electrode</h2><ul>"]
    for eid in sorted(by_electrode):
        index_html.append(f'<li><a href="{eid}_report.html">{eid}</a></li>')
    index_html.append("</ul><p><a href='summary.csv'>Download summary CSV</a></p></body></html>")
    (OUTPUT_DIR / "index.html").write_text("\n".join(index_html), encoding="utf-8")

    print(f"done. reports are in {OUTPUT_DIR}/")
    print(f"open {OUTPUT_DIR}/index.html in your browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
