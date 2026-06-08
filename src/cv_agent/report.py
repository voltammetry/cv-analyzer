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

from .analysis import CdlResult, LavironResult, NicholsonResult, RandlesSevcikResult
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
    # scatter the data points
    ax.scatter(x, np.abs(result.ipa_a) * 1e6, color="C3", label="|ipa| (anodic)", zorder=3)
    ax.scatter(x, np.abs(result.ipc_a) * 1e6, color="C0", label="|ipc| (cathodic)", zorder=3)

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
    """peak separation vs log scan rate. the diagnostic plot for laviron."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    valid = ~np.isnan(result.delta_ep_v)
    ax.scatter(result.log_scan_rates[valid],
               result.delta_ep_v[valid] * 1000,
               color="C4", zorder=3, label="ΔEp")
    # 100 mV is the rough threshold where the kinetic regime starts
    ax.axhline(100, color="k", lw=0.4, ls=":", alpha=0.5,
               label="~100 mV: kinetic regime threshold")
    ax.set_xlabel(r"$\log_{10}(\nu / \mathrm{V s^{-1}})$")
    ax.set_ylabel(r"$\Delta E_p$ / mV")
    title = f"Laviron: {result.electrode_id}"
    if result.k0_per_s is not None:
        title += f"  (k⁰ ≈ {result.k0_per_s:.2e} s⁻¹, α = {result.alpha:.2f})"
    else:
        title += "  (k⁰ unresolved, see notes)"
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
    # laviron section, including the caveat note at the bottom about when
    # k0 is reliable.
    if lav.k0_per_s is None:
        k0_str = "(unresolved)"
        alpha_str = "(unresolved)" if lav.alpha is None else f"{lav.alpha:.3f}"
    else:
        k0_str = f"{lav.k0_per_s:.3e} s⁻¹"
        alpha_str = f"{lav.alpha:.3f}"
    return f"""
<h2>3 · Laviron (Electron Transfer Kinetics)</h2>
<p>Peak separation ΔEp = Ep,a − Ep,c plotted against log(ν). In the kinetic
regime (ΔEp ≳ 100 mV at 25 °C, n=1), the anodic and cathodic branches go
linear with log(ν) and yield the transfer coefficient α and k⁰.</p>
<img src="{lav_plot.name}" alt="Laviron plot">
<p><b>α</b> = {alpha_str} &nbsp;&nbsp;<b>k⁰</b> = {k0_str}</p>
<div class="note"><b>Caveat:</b> {lav.notes}</div>
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
) -> Path:
    """stitch the plots and analysis results into one html page for one electrode."""
    summary = _summary_block(rs, cdl_res, lav)
    # always start with the raw cv overlay so the reader sees the data first
    sections: list[str] = [
        f'<h2>0 · Raw CV traces</h2><img src="{cv_plot.name}" alt="CV overlay">'
    ]
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
