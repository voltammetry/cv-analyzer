"""parses every cv file, runs the protocol's analyses, writes reports.

  1. parse every cv file in data/raw (experiment type read from the chi
     header's potential window; an optional manifest.csv sets electrode
     identity, size, and replicate grouping explicitly)
  2. per electrode: cv overlay, per-electrode randles sevcik / cdl, the
     protocol's peak table, and an html report. this is the qc view.
  3. per replicate group (the protocol's real unit of analysis): average the
     peak tables across electrodes, pick laviron vs nicholson from the mean
     peak separation (a fixed threshold rule, no llm), fit the averaged data,
     and write a group report with error bars.
  4. write a summary csv (per-electrode and per-group rows) and an index.html.

every number and every method choice is deterministic. if ANTHROPIC_API_KEY
is set, claude is used only to phrase plain-language summaries of numbers
that were already computed.
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
from cv_agent.agent import compare, interpret
from cv_agent.analysis import (
    CdlResult,
    RandlesSevcikResult,
    average_peak_tables,
    cdl,
    choose_kinetics_method,
    laviron,
    nicholson,
    peak_table,
    randles_sevcik,
    randles_sevcik_from_replicates,
)
from cv_agent.models import CVExperiment, Electrolyte, default_replicate_group
from cv_agent.report import (
    plot_cdl,
    plot_cv_overlay,
    plot_laviron,
    plot_nicholson,
    plot_randles_sevcik,
    write_electrode_report,
    write_group_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"


def process_electrode(electrode_id: str, experiments: list[CVExperiment],
                      out_dir: Path
                      ) -> tuple[RandlesSevcikResult | None, CdlResult | None,
                                 str, list[str], int]:
    """per-electrode qc: overlay plot, per-electrode fits, peak table, report.

    kinetics (laviron/nicholson) intentionally do not happen here anymore;
    the protocol runs them on replicate-averaged data, which happens at the
    group level in process_group().
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # split experiments by electrolyte (as classified from the file header)
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

    # the protocol's per-scan-rate peak table, shown in the report
    peaks = peak_table(ferri) if ferri else None

    rs_res: RandlesSevcikResult | None = None
    rs_plot_path: Path | None = None
    if ferri:
        rs_res = randles_sevcik(ferri)
        rs_plot_path = plot_randles_sevcik(rs_res, out_dir / f"{electrode_id}_randles_sevcik.png")

    cdl_res: CdlResult | None = None
    cdl_plot_path: Path | None = None
    if pbs:
        cdl_res = cdl(pbs)
        cdl_plot_path = plot_cdl(cdl_res, out_dir / f"{electrode_id}_cdl.png")

    # plain-language summary of the computed numbers (llm phrasing optional)
    size_mm = experiments[0].file_meta.electrode_size_mm
    interp = interpret(electrode_id, rs_res, cdl_res, None, size_mm=size_mm)

    write_electrode_report(
        electrode_id=electrode_id,
        cv_plot=cv_plot,
        rs=rs_res, rs_plot=rs_plot_path,
        cdl_res=cdl_res, cdl_plot=cdl_plot_path,
        lav=None, lav_plot=None,
        nic=None, nic_plot=None,
        out_path=out_dir / f"{electrode_id}_report.html",
        peaks=peaks,
    )

    return rs_res, cdl_res, interp.summary, interp.flags, size_mm


def process_group(group_id: str, ferri: list[CVExperiment],
                  out_dir: Path) -> dict | None:
    """the protocol's replicate-level analysis for one group of electrodes.

    builds each electrode's peak table, averages them (mean ± std at each
    scan rate), routes to laviron or nicholson from the mean peak separation,
    fits the averaged data, and writes the group html report. returns a
    summary row, or None if there was nothing to average.
    """
    by_electrode: dict[str, list[CVExperiment]] = defaultdict(list)
    for e in ferri:
        by_electrode[e.file_meta.electrode_id].append(e)
    tables = [peak_table(exps) for _, exps in sorted(by_electrode.items())]
    if not tables:
        return None

    rep = average_peak_tables(tables, group_id)
    method = choose_kinetics_method(rep.mean_delta_ep_mv)
    print(f"  {len(tables)} electrode(s), mean ΔEp = {rep.mean_delta_ep_mv:.0f} mV "
          f"-> kinetics method: {method}")

    rs_res = randles_sevcik_from_replicates(rep)
    rs_plot_path = plot_randles_sevcik(rs_res, out_dir / f"group_{group_id}_randles_sevcik.png")

    lav_res = nic_res = None
    lav_plot_path = nic_plot_path = None
    if method == "laviron":
        lav_res = laviron(rep)
        lav_plot_path = plot_laviron(lav_res, out_dir / f"group_{group_id}_laviron.png")
    elif method == "nicholson":
        nic_res = nicholson(rep)
        nic_plot_path = plot_nicholson(nic_res, out_dir / f"group_{group_id}_nicholson.png")

    write_group_report(
        group_id=group_id, rep=rep,
        rs=rs_res, rs_plot=rs_plot_path,
        lav=lav_res, lav_plot=lav_plot_path,
        nic=nic_res, nic_plot=nic_plot_path,
        method=method,
        out_path=out_dir / f"group_{group_id}_report.html",
    )

    row: dict = {
        "electrode_id": f"group:{group_id}",
        "n_files": len(ferri),
        "easa_mm2": rs_res.easa_mean_cm2 * 1e2,
        "rs_r2_anodic": rs_res.anodic_fit.r_squared,
        "rs_r2_cathodic": rs_res.cathodic_fit.r_squared,
        "kinetics_method": method,
    }
    print(f"  EASA (averaged, n={rs_res.n_replicates}): {row['easa_mm2']:.4f} mm² "
          f"(R² anodic={row['rs_r2_anodic']:.4f}, cathodic={row['rs_r2_cathodic']:.4f})")
    if lav_res is not None and lav_res.k0_per_s is not None:
        row["laviron_ks_per_s"] = lav_res.k0_per_s
        row["laviron_alpha"] = lav_res.alpha
        print(f"  Laviron: ks = {lav_res.k0_per_s:.3e} s⁻¹ "
              f"(α = {lav_res.alpha:.3f}, {lav_res.alpha_source}, "
              f"Vc = {lav_res.vc_v_s:.3g} V/s)")
    elif lav_res is not None:
        print("  Laviron: ks unresolved (see group report notes)")
    if nic_res is not None and nic_res.ks_mean_cm_s is not None:
        row["nicholson_ks_cm_s"] = nic_res.ks_mean_cm_s
        row["nicholson_ks_std_cm_s"] = nic_res.ks_std_cm_s
        print(f"  Nicholson: ks = {nic_res.ks_mean_cm_s:.3e} "
              f"± {nic_res.ks_std_cm_s:.1e} cm/s")
    return row


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """one big csv across every electrode and group, for excel."""
    if not rows:
        return
    fieldnames = [
        "electrode_id",
        "n_files",
        "electrode_size_mm",
        "replicate_group",
        "easa_mm2",
        "rs_r2_anodic",
        "rs_r2_cathodic",
        "cdl_uF",
        "cdl_r2",
        "kinetics_method",
        "laviron_ks_per_s",
        "laviron_alpha",
        "nicholson_ks_cm_s",
        "nicholson_ks_std_cm_s",
        "summary",
        "flags",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("llm phrasing: ON (summaries only; all numbers and routing are deterministic)")
    else:
        print("llm phrasing: OFF (rule-based summaries)")
    print()

    print(f"reading cv files from {DATA_DIR}")
    experiments = parse_directory(DATA_DIR, skip_on_error=True)
    print(f"parsed {len(experiments)} files")
    print()

    # group by electrode for the per-electrode qc pass
    by_electrode: dict[str, list[CVExperiment]] = defaultdict(list)
    for e in experiments:
        by_electrode[e.file_meta.electrode_id].append(e)

    summary_rows: list[dict] = []
    for electrode_id in sorted(by_electrode):
        exps = by_electrode[electrode_id]
        print(f"--- {electrode_id} ({len(exps)} files) ---")
        try:
            rs, cdl_res, summary, flags, size_mm = process_electrode(
                electrode_id, exps, OUTPUT_DIR
            )
        except Exception:
            print(f"  error on {electrode_id}")
            traceback.print_exc()
            continue

        group = (exps[0].file_meta.replicate_group
                 or default_replicate_group(electrode_id))
        row: dict = {"electrode_id": electrode_id, "n_files": len(exps),
                     "electrode_size_mm": size_mm, "replicate_group": group}
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
        row["summary"] = summary
        row["flags"] = " | ".join(flags) if flags else ""
        print(f"  summary: {summary}")
        for flag in flags:
            print(f"  flag: {flag}")
        summary_rows.append(row)
        print()

    # the protocol's replicate-group pass, over ferro/ferri data only
    ferri_by_group: dict[str, list[CVExperiment]] = defaultdict(list)
    for e in experiments:
        if e.file_meta.electrolyte == Electrolyte.FERRO_FERRI:
            group = (e.file_meta.replicate_group
                     or default_replicate_group(e.file_meta.electrode_id))
            ferri_by_group[group].append(e)

    group_ids: list[str] = []
    for group_id in sorted(ferri_by_group):
        print(f"--- replicate group {group_id} ---")
        try:
            row = process_group(group_id, ferri_by_group[group_id], OUTPUT_DIR)
        except Exception:
            print(f"  error on group {group_id}")
            traceback.print_exc()
            continue
        if row is not None:
            summary_rows.append(row)
            group_ids.append(group_id)
        print()

    write_summary_csv(summary_rows, OUTPUT_DIR / "summary.csv")

    # cross electrode comparison. all stats computed in code; llm may only
    # rephrase the overview paragraph.
    print("--- batch comparison ---")
    electrode_rows = [r for r in summary_rows
                      if not str(r["electrode_id"]).startswith("group:")]
    comparison = compare(electrode_rows)
    print(f"overview: {comparison.overview}")
    for label, items in (("patterns", comparison.patterns),
                         ("outliers", comparison.outliers),
                         ("recommendations", comparison.recommendations)):
        if items:
            print(label)
            for it in items:
                print(f"  - {it}")
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

    # index page: group reports first (they're the protocol's real output),
    # then per-electrode qc reports, then the batch comparison.
    index_html = ["<!doctype html><html><head><meta charset='utf-8'>",
                  "<title>CV-Agent Reports</title>",
                  "<style>body{font-family:system-ui;max-width:700px;margin:2em auto;padding:0 1em;}",
                  "li{margin:0.3em 0;}h1{border-bottom:2px solid #444;}",
                  ".batch{background:#eef4ff;padding:0.8em 1em;border-radius:6px;margin:1em 0;}</style>",
                  "</head><body><h1>CV Reports</h1>",
                  "<div class='batch'><b><a href='batch.html'>Batch Comparison →</a></b> "
                  "computed cross-electrode stats, outliers, and recommendations.</div>"]
    if group_ids:
        index_html.append("<h2>Replicate groups (protocol results)</h2><ul>")
        for gid in group_ids:
            index_html.append(f'<li><a href="group_{gid}_report.html">group {gid}</a></li>')
        index_html.append("</ul>")
    index_html.append("<h2>Per electrode (QC)</h2><ul>")
    for eid in sorted(by_electrode):
        index_html.append(f'<li><a href="{eid}_report.html">{eid}</a></li>')
    index_html.append("</ul><p><a href='summary.csv'>Download summary CSV</a></p></body></html>")
    (OUTPUT_DIR / "index.html").write_text("\n".join(index_html), encoding="utf-8")

    print(f"done. reports are in {OUTPUT_DIR}/")
    print(f"open {OUTPUT_DIR}/index.html in your browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())