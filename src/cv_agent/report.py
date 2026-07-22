"""generates the plots and the html reports that the user actually sees.

this is the presentation layer. no math happens here, everything has already
been computed in analysis.py. all we do is take those result objects and turn
them into pictures and a webpage.

structured as a bunch of pure functions, so the agent (or the runner script)
can call whatever it needs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non interactive backend, safe to use in scripts
import matplotlib.pyplot as plt
import numpy as np

from .analysis import CdlResult, LavironResult, NicholsonResult, RandlesSevcikResult, PeakTable, ReplicateTable
from .models import CVExperiment

# -----------------------------------------------------------------------------
# the plots
# -----------------------------------------------------------------------------


def plot_cv_overlay(experiments: list[CVExperiment], out_path: Path,
                    title: str | None = None) -> Path:
    """draw all the cvs for one electrode on top of each other, colored by scan rate."""
    experiments = sorted(experiments, key=lambda e: e.scan_rate_v_s)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # viridis colormap goes dark to light, perfect for ordered things like scan rates
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(experiments):
        color = cmap(i / max(1, len(experiments) - 1))
        # show current in microamps so the numbers are readable
        ax.plot(e.potential_v, e.current_a * 1e6,
                color=color, lw=0.9,
                label=f"{e.file_meta.scan_rate_mv_s:g} mV/s")
    ax.set_xlabel("Potential / V vs Ag/AgCl")
    ax.set_ylabel("Current / µA")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", ncol=2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_randles_sevcik(result: RandlesSevcikResult, out_path: Path) -> Path:
    """ip vs sqrt(nu) with the linear fit on top, for both anodic and cathodic peaks."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = result.sqrt_scan_rates
    # scatter the data points. when the result came from replicate-averaged
    # data we also have a std dev per point, drawn as error bars per protocol.
    yerr_a = result.ipa_std_a * 1e6 if result.ipa_std_a is not None else None
    yerr_c = result.ipc_std_a * 1e6 if result.ipc_std_a is not None else None
    ax.errorbar(x, np.abs(result.ipa_a) * 1e6, yerr=yerr_a, fmt="o",
                color="C3", label="|ipa| (anodic)", zorder=3, capsize=3)
    ax.errorbar(x, np.abs(result.ipc_a) * 1e6, yerr=yerr_c, fmt="o",
                color="C0", label="|ipc| (cathodic)", zorder=3, capsize=3)

    # draw the fit lines from 0 out past the last point
    x_line = np.linspace(0, x.max() * 1.05, 100)
    ya = (result.anodic_fit.slope * x_line + result.anodic_fit.intercept) * 1e6
    yc = (result.cathodic_fit.slope * x_line + result.cathodic_fit.intercept) * 1e6
    ax.plot(x_line, ya, color="C3", lw=1.0,
            label=f"anodic fit (R²={result.anodic_fit.r_squared:.4f})")
    ax.plot(x_line, yc, color="C0", lw=1.0, ls="--",
            label=f"cathodic fit (R²={result.cathodic_fit.r_squared:.4f})")

    ax.set_xlabel(r"$\sqrt{\nu}$ / (V/s)$^{1/2}$")
    ax.set_ylabel(r"$|i_p|$ / µA")
    ax.set_title(f"Randles–Ševčík: {result.electrode_id}  "
                 f"(EASA = {result.easa_mean_cm2*1e2:.3f} mm²)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_cdl(result: CdlResult, out_path: Path) -> Path:
    """current at 0.6 V vs scan rate, with the linear fit. slope is the capacitance."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nu = result.scan_rates_v_s
    i_uA = result.i_at_06v_a * 1e6
    ax.scatter(nu, i_uA, color="C2", zorder=3, label="i at 0.6 V (anodic sweep)")
    x_line = np.linspace(0, nu.max() * 1.05, 100)
    y_line = (result.fit.slope * x_line + result.fit.intercept) * 1e6
    ax.plot(x_line, y_line, color="C2", lw=1.0,
            label=f"fit (R²={result.fit.r_squared:.4f})")
    ax.set_xlabel(r"Scan rate $\nu$ / (V/s)")
    ax.set_ylabel("i at 0.6 V / µA")
    ax.set_title(f"C_dl: {result.electrode_id}  "
                 f"(C_dl = {result.cdl_microfarads:.3f} µF)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_laviron(result: LavironResult, out_path: Path) -> Path:
    """the protocol's laviron plot: averaged ΔEa and ΔEc vs ln(ν) with fits."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = result.ln_scan_rates
    ea_err = (result.delta_ea_std_v * 1000 if result.delta_ea_std_v is not None else None)
    ec_err = (result.delta_ec_std_v * 1000 if result.delta_ec_std_v is not None else None)
    ax.errorbar(x, result.delta_ea_v * 1000, yerr=ea_err, fmt="o", color="C3",
                capsize=3, zorder=3, label="ΔEa = Ep,a − E0'")
    ax.errorbar(x, result.delta_ec_v * 1000, yerr=ec_err, fmt="o", color="C0",
                capsize=3, zorder=3, label="ΔEc = Ep,c − E0'")

    if result.anodic_branch_fit is not None and result.cathodic_branch_fit is not None:
        # extend the fit lines to where they cross  delta E = 0 so the x-intercept
        # (the critical scan rate Vc) is visible on the plot
        x0s = [-f.intercept / f.slope
               for f in (result.anodic_branch_fit, result.cathodic_branch_fit)
               if f.slope != 0]
        x_min = min([x.min(), *x0s]) - 0.3
        x_line = np.linspace(x_min, x.max() + 0.3, 100)
        fa, fc = result.anodic_branch_fit, result.cathodic_branch_fit
        ax.plot(x_line, (fa.slope * x_line + fa.intercept) * 1000, color="C3",
                lw=1.0, label=f"anodic fit (R²={fa.r_squared:.3f})")
        ax.plot(x_line, (fc.slope * x_line + fc.intercept) * 1000, color="C0",
                lw=1.0, ls="--", label=f"cathodic fit (R²={fc.r_squared:.3f})")
        if result.vc_v_s is not None:
            ax.axvline(np.log(result.vc_v_s), color="k", lw=0.6, ls=":",
                       label=f"x-intercept → Vc = {result.vc_v_s:.3g} V/s")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)

    ax.set_xlabel(r"$\ln(\nu / \mathrm{V\,s^{-1}})$")
    ax.set_ylabel(r"$\Delta E$ / mV")
    title = f"Laviron: {result.electrode_id}"
    if result.k0_per_s is not None:
        title += f"  (ks ≈ {result.k0_per_s:.2e} s⁻¹, α = {result.alpha:.2f})"
    else:
        title += "  (ks unresolved, see notes)"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path

def plot_nicholson(result: NicholsonResult, out_path: Path) -> Path:
    """ks at each scan rate plus the average. the diagnostic plot for Nicholson."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nu = result.scan_rates_v_s
    if np.all(np.isnan(result.ks_per_scan_cm_s)):
        ax.text(0.5, 0.5, "ks not computed\n(ΔEp outside 61-212 mV)",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(nu, result.ks_per_scan_cm_s, color="C5", zorder=3,
                   label="ks per scan rate")
        if result.ks_mean_cm_s is not None:
            ax.axhline(result.ks_mean_cm_s, color="C5", lw=1.0, ls="--",
                       label=f"mean ks = {result.ks_mean_cm_s:.2e} cm/s")
    ax.set_xlabel(r"Scan rate $\nu$ / (V/s)")
    ax.set_ylabel(r"$k^0$ / cm s$^{-1}$")
    ax.set_title(f"Nicholson: {result.electrode_id}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path

# -----------------------------------------------------------------------------
# the html report. simple template with inline css so the file is self contained.
# -----------------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{electrode_id} report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
          margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.3em; }}
  h2 {{ margin-top: 1.8em; color: #224; }}
  .summary {{ background: #f4f6fa; padding: 1em 1.4em; border-radius: 6px;
              border-left: 4px solid #446; }}
  .metric {{ display: inline-block; margin-right: 2em; }}
  .metric .v {{ font-size: 1.3em; font-weight: bold; }}
  .metric .l {{ font-size: 0.85em; color: #555; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd;
         border-radius: 4px; margin: 0.5em 0; }}
  .note {{ background: #fff4dd; padding: 0.6em 1em; border-radius: 4px;
           border-left: 3px solid #c80; font-size: 0.9em; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; }}
  td, th {{ border: 1px solid #ccc; padding: 0.3em 0.7em; font-size: 0.9em; }}
  th {{ background: #f0f0f3; }}
  footer {{ margin-top: 3em; font-size: 0.8em; color: #888; }}
</style></head><body>
<h1>Electrode {electrode_id}</h1>
<div class="summary">{summary_html}</div>
{sections_html}
<footer>Generated by cv-agent. Analysis per Ye et al., 2024.</footer>
</body></html>
"""


def _summary_block(rs: RandlesSevcikResult | None,
                   cdl_res: CdlResult | None,
                   lav: LavironResult | None) -> str:
    # the headline numbers at the top of the report. just the highlights.
    parts: list[str] = []
    if rs is not None:
        parts.append(
            f'<div class="metric"><div class="v">{rs.easa_mean_cm2*1e2:.3f} mm²</div>'
            f'<div class="l">EASA (mean of branches)</div></div>'
        )
    if cdl_res is not None:
        parts.append(
            f'<div class="metric"><div class="v">{cdl_res.cdl_microfarads:.3f} µF</div>'
            f'<div class="l">Double-layer capacitance</div></div>'
        )
    if lav is not None and lav.k0_per_s is not None:
        parts.append(
            f'<div class="metric"><div class="v">{lav.k0_per_s:.2e} s⁻¹</div>'
            f'<div class="l">k⁰ (Laviron est.)</div></div>'
        )
    return "".join(parts) if parts else "<em>No results.</em>"


def _rs_section_html(rs: RandlesSevcikResult, rs_plot: Path) -> str:
    # the randles sevcik section. shows the plot, the numbers, and a table
    # of every scan rate's anodic and cathodic peak.
    rows = "".join(
        f"<tr><td>{nu*1000:g}</td><td>{ipa*1e6:.3f}</td><td>{ipc*1e6:.3f}</td></tr>"
        for nu, ipa, ipc in zip(rs.scan_rates_v_s, rs.ipa_a, rs.ipc_a)
    )
    return f"""
<h2>1 · Randles–Ševčík (EASA)</h2>
<p>Linear fit of |ip| vs √ν across ferro/ferri CVs.
Slope → electrochemically active surface area via
<em>ip = 2.69×10⁵ · n<sup>3/2</sup> · A · D<sup>1/2</sup> · C · √ν</em>.</p>
<img src="{rs_plot.name}" alt="Randles-Sevcik plot">
<p><b>EASA (anodic)</b> = {rs.easa_anodic_cm2*1e2:.3f} mm² &nbsp;&nbsp;
<b>EASA (cathodic)</b> = {rs.easa_cathodic_cm2*1e2:.3f} mm² <br>
<b>Anodic R²</b> = {rs.anodic_fit.r_squared:.4f} &nbsp;&nbsp;
<b>Cathodic R²</b> = {rs.cathodic_fit.r_squared:.4f}</p>
<table><tr><th>ν / mV·s⁻¹</th><th>ipa / µA</th><th>ipc / µA</th></tr>{rows}</table>
"""

def _peak_table_section_html(table: PeakTable) -> str:
    # the explicit per-scan-rate table the protocol requires (steps 16-18):
    # both peaks, formal potential, and the ΔE columns.
    rows = "".join(
        f"<tr><td>{nu*1000:g}</td><td>{epa:.4f}</td><td>{ipa*1e6:.3f}</td>"
        f"<td>{epc:.4f}</td><td>{ipc*1e6:.3f}</td><td>{e0:.4f}</td>"
        f"<td>{dea*1000:.1f}</td><td>{dec*1000:.1f}</td><td>{dep*1000:.1f}</td></tr>"
        for nu, epa, ipa, epc, ipc, e0, dea, dec, dep in zip(
            table.scan_rates_v_s, table.ep_a_v, table.ip_a_a, table.ep_c_v,
            table.ip_c_a, table.e0_prime_v, table.delta_ea_v,
            table.delta_ec_v, table.delta_ep_v)
    )
    return f"""
<h2>Peak table (protocol steps 16–18)</h2>
<p>Second-cycle peaks per scan rate, the formal potential
E⁰′ = (Ep,a + Ep,c)/2, and ΔEa/ΔEc = Ep − E⁰′.</p>
<table><tr><th>ν / mV·s⁻¹</th><th>Ep,a / V</th><th>ipa / µA</th>
<th>Ep,c / V</th><th>ipc / µA</th><th>E⁰′ / V</th>
<th>ΔEa / mV</th><th>ΔEc / mV</th><th>ΔEp / mV</th></tr>{rows}</table>
"""


def _cdl_section_html(cdl_res: CdlResult, cdl_plot: Path) -> str:
    # cdl section. plot, the capacitance and r squared, then a table of i at
    # 0.6 V for each scan rate.
    rows = "".join(
        f"<tr><td>{nu*1000:g}</td><td>{i*1e6:.4f}</td></tr>"
        for nu, i in zip(cdl_res.scan_rates_v_s, cdl_res.i_at_06v_a)
    )
    return f"""
<h2>2 · Double-Layer Capacitance (C_dl)</h2>
<p>Oxidation current at +0.6 V on the anodic sweep, plotted against scan rate.
For purely capacitive behavior, <em>i = C_dl · ν</em>, so the slope is C_dl.</p>
<img src="{cdl_plot.name}" alt="C_dl plot">
<p><b>C_dl</b> = {cdl_res.cdl_microfarads:.4f} µF &nbsp;&nbsp;
<b>Fit R²</b> = {cdl_res.fit.r_squared:.4f}</p>
<table><tr><th>ν / mV·s⁻¹</th><th>i at 0.6 V / µA</th></tr>{rows}</table>
"""


def _laviron_section_html(lav: LavironResult, lav_plot: Path) -> str:
    # laviron section per the protocol: ln(ν) fit, x-intercept, ks = nFαVc/RT
    if lav.k0_per_s is None:
        ks_str = "(unresolved)"
        vc_str = "(unresolved)"
    else:
        ks_str = f"{lav.k0_per_s:.3e} s⁻¹"
        vc_str = f"{lav.vc_v_s:.3g} V/s"
    alpha_str = ("(unresolved)" if lav.alpha is None
                 else f"{lav.alpha:.3f} ({lav.alpha_source})")
    return f"""
<h2>3 · Laviron (Electron Transfer Kinetics)</h2>
<p>Replicate-averaged ΔEa and ΔEc (n = {lav.n_replicates}) plotted against
ln(ν) and fit linearly. The x-intercept (ΔE → 0) gives the critical scan
rate Vc, and <em>ks = n·F·α·Vc / (R·T)</em> per protocol step 25. Applies
when the mean peak separation exceeds 212 mV.</p>
<img src="{lav_plot.name}" alt="Laviron plot">
<p><b>α</b> = {alpha_str} &nbsp;&nbsp;<b>Vc</b> = {vc_str} &nbsp;&nbsp;
<b>ks</b> = {ks_str}</p>
<div class="note"><b>Notes:</b> {lav.notes}</div>
"""

def _nicholson_section_html(nic: NicholsonResult, nic_plot: Path) -> str:
    if nic.ks_mean_cm_s is None:
        ks_str = "(not computed)"
    else:
        ks_str = f"{nic.ks_mean_cm_s:.3e} ± {nic.ks_std_cm_s:.1e} cm/s"
    rows = "".join(
        f"<tr><td>{nu*1000:g}</td><td>{dep*1000:.1f}</td><td>{ps:.3f}</td></tr>"
        for nu, dep, ps in zip(nic.scan_rates_v_s, nic.delta_ep_v, nic.psi)
    )
    return f"""
<h2>4 · Nicholson (Electron Transfer Kinetics)</h2>
<p>For peak separation in the 61-212 mV range, k⁰ is found per scan rate from
<em>k⁰ = Ψ · √(nπDFν / RT)</em>, Ψ from Nicholson's working curve.</p>
<img src="{nic_plot.name}" alt="Nicholson plot">
<p><b>mean k⁰</b> = {ks_str}</p>
<table><tr><th>ν / mV·s⁻¹</th><th>ΔEp / mV</th><th>Ψ</th></tr>{rows}</table>
<div class="note">{nic.notes}</div>
"""

def write_electrode_report(
    electrode_id: str,
    cv_plot: Path,
    rs: RandlesSevcikResult | None,
    rs_plot: Path | None,
    cdl_res: CdlResult | None,
    cdl_plot: Path | None,
    lav: LavironResult | None,
    lav_plot: Path | None,
    nic: NicholsonResult | None,
    nic_plot: Path | None,
    out_path: Path,
    peaks: PeakTable | None = None,
) -> Path:
    """stitch the plots and analysis results into one html page for one electrode."""
    summary = _summary_block(rs, cdl_res, lav)
    # always start with the raw cv overlay so the reader sees the data first
    sections: list[str] = [
        f'<h2>0 · Raw CV traces</h2><img src="{cv_plot.name}" alt="CV overlay">'
    ]
    # the protocol's per-scan-rate peak table comes right after the raw data
    if peaks is not None:                           
        sections.append(_peak_table_section_html(peaks))
    # then each analysis section, if we have it
    if rs is not None and rs_plot is not None:
        sections.append(_rs_section_html(rs, rs_plot))
    if cdl_res is not None and cdl_plot is not None:
        sections.append(_cdl_section_html(cdl_res, cdl_plot))
    if lav is not None and lav_plot is not None:
        sections.append(_laviron_section_html(lav, lav_plot))
    if nic is not None and nic_plot is not None:
        sections.append(_nicholson_section_html(nic, nic_plot))

    html = _HTML_TEMPLATE.format(
        electrode_id=electrode_id,
        summary_html=summary,
        sections_html="\n".join(sections),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path

def write_group_report(
    group_id: str,
    rep,  # ReplicateTable
    rs: RandlesSevcikResult | None,
    rs_plot: Path | None,
    lav: LavironResult | None,
    lav_plot: Path | None,
    nic: NicholsonResult | None,
    nic_plot: Path | None,
    method: str,
    out_path: Path,
) -> Path:
    """the replicate-group report: averaged data, the protocol's real output.

    shows the averaged peak table (mean ± std across the replicate
    electrodes), the randles sevcik fit on averaged currents, and whichever
    kinetics method the peak-separation rule selected.
    """
    summary = _summary_block(rs, None, lav)
    if nic is not None and nic.ks_mean_cm_s is not None:
        summary += (
            f'<div class="metric"><div class="v">{nic.ks_mean_cm_s:.2e} cm/s</div>'
            f'<div class="l">ks (Nicholson)</div></div>'
        )

    avg_rows = "".join(
        f"<tr><td>{nu*1000:g}</td>"
        f"<td>{ia*1e6:.2f} ± {sa*1e6:.2f}</td>"
        f"<td>{ic*1e6:.2f} ± {sc*1e6:.2f}</td>"
        f"<td>{dep*1000:.1f} ± {sdep*1000:.1f}</td></tr>"
        for nu, ia, sa, ic, sc, dep, sdep in zip(
            rep.scan_rates_v_s,
            rep.ip_a_mean_a, rep.ip_a_std_a,
            rep.ip_c_mean_a, rep.ip_c_std_a,
            rep.delta_ep_mean_v, rep.delta_ep_std_v)
    )
    sections: list[str] = [f"""
<h2>Replicate-averaged data (protocol steps 19–20)</h2>
<p>Mean ± std dev across electrodes: {", ".join(rep.electrode_ids)}.
Mean peak separation ΔEp = {rep.mean_delta_ep_mv:.0f} mV → kinetics
method selected by the protocol rule: <b>{method}</b>.</p>
<table><tr><th>ν / mV·s⁻¹</th><th>ipa / µA</th><th>ipc / µA</th>
<th>ΔEp / mV</th></tr>{avg_rows}</table>
"""]
    if rs is not None and rs_plot is not None:
        sections.append(_rs_section_html(rs, rs_plot))
    if lav is not None and lav_plot is not None:
        sections.append(_laviron_section_html(lav, lav_plot))
    if nic is not None and nic_plot is not None:
        sections.append(_nicholson_section_html(nic, nic_plot))

    html = _HTML_TEMPLATE.format(
        electrode_id=f"Replicate group {group_id}",
        summary_html=summary,
        sections_html="\n".join(sections),
    )
    # the shared template's h1 says "Electrode {id}"; fix the prefix for groups
    html = html.replace("<h1>Electrode Replicate group", "<h1>Replicate group")
    html = html.replace("<title>Replicate group", "<title>Group")
    out_path.write_text(html, encoding="utf-8")
    return out_path