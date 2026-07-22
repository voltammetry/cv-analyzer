"""interpretation layer: deterministic decisions, optional llm phrasing.

this file used to be "the agent": an llm decided which analyses to run and
eyeballed outliers. that was the wrong shape for this problem. every decision
in the protocol is a fixed numeric rule (the laviron/nicholson choice is just
a 212 mV threshold on peak separation), and stats should be computed, not
estimated by a language model. so:

  - decide() is now purely deterministic. it applies the protocol's rules and
    nothing else. same inputs, same outputs, every time.
  - compare() computes all of its stats (means, 1σ outliers, bad fits) in
    code, always. if ANTHROPIC_API_KEY is set, the llm may rewrite the
    *overview paragraph* in nicer prose from the already-computed numbers,
    but it never produces a number or an outlier call itself.
  - interpret() writes the plain-language summary of finished results. this
    is the one place an llm genuinely helps, and it only ever describes
    numbers that were already computed. without a key, a rule-based summary
    is built from the same numbers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
import math

from .analysis import CdlResult, LavironResult, RandlesSevcikResult, choose_kinetics_method
from .models import CVExperiment, Electrolyte


# -----------------------------------------------------------------------------
# what the agent returns
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentDecision:
    """what the agent decided to do for one electrode."""

    electrode_id: str
    analyses_to_run: set[str]    # subset of {"randles_sevcik", "cdl", "laviron"}
    reasoning: str               # short explanation of why these and not others


@dataclass(frozen=True)
class AgentInterpretation:
    """the agent's written summary after the analyses have been run."""

    electrode_id: str
    summary: str                 # one short paragraph for the report
    flags: list[str]             # any concerns worth flagging to the user


@dataclass(frozen=True)
class AgentComparison:
    """the agent's read on the whole batch, comparing electrodes to each other."""

    overview: str                # one paragraph summarizing the batch
    patterns: list[str]          # specific patterns or correlations the agent noticed
    outliers: list[str]          # electrodes that stand out from the rest, with why
    recommendations: list[str]   # what to look at next, what to redo, etc


# -----------------------------------------------------------------------------
# the deterministic fallback. used when no api key is set.
# -----------------------------------------------------------------------------

def _mean_peak_separation_mv(experiments: list[CVExperiment]) -> float | None:
    """mean (Ep,a - Ep,c) across scan rates, second cycle, in mV. None if unknown."""
    seps = []
    for e in experiments:
        epa = epc = None
        for seg in e.segment_results:
            if seg.ep_v is None or seg.ip_a is None:
                continue
            if seg.ip_a > 0:
                epa = seg.ep_v
            elif seg.ip_a < 0:
                epc = seg.ep_v
        if epa is not None and epc is not None:
            seps.append((epa - epc) * 1000.0)
    if not seps:
        return None
    return sum(seps) / len(seps)

def _deterministic_decide(experiments: list[CVExperiment]) -> AgentDecision:
    """the simple rule based version of decide(). used as a backup.

    no llm, just the same routing logic from run_analysis.py. ferro/ferri
    gets randles sevcik and laviron, pbs gets cdl, and that's it.
    """
    if not experiments:
        raise ValueError("decide, got an empty experiment list")
    electrode_id = experiments[0].file_meta.electrode_id
    electrolytes = {e.file_meta.electrolyte for e in experiments}

    analyses: set[str] = set()
    reasons: list[str] = []
    if Electrolyte.FERRO_FERRI in electrolytes:
        analyses.add("randles_sevcik")
        ff = [e for e in experiments if e.file_meta.electrolyte == Electrolyte.FERRO_FERRI]
        sep_mv = _mean_peak_separation_mv(ff)
        if sep_mv is None:
            analyses.add("laviron")
            reasons.append("ferro/ferri; ΔEp unknown, defaulting to randles sevcik + laviron")
        else:
            method = choose_kinetics_method(sep_mv)
            if method != "none":
                analyses.add(method)
                reasons.append(f"ferro/ferri; ΔEp {sep_mv:.0f} mV -> {method} (protocol thresholds)")
            else:
                reasons.append(f"ferro/ferri; ΔEp {sep_mv:.0f} mV < 61 mV, "
                               f"essentially reversible, neither kinetics method applies")
    if Electrolyte.PBS in electrolytes:
        analyses.add("cdl")
        reasons.append("pbs data is here, so cdl for double layer capacitance")

    return AgentDecision(
        electrode_id=electrode_id,
        analyses_to_run=analyses,
        reasoning=". ".join(reasons) if reasons else "no recognized electrolyte",
    )


def _deterministic_interpret(
    electrode_id: str,
    rs: RandlesSevcikResult | None,
    cdl_res: CdlResult | None,
    lav: LavironResult | None,
    size_mm: int = 1,
) -> AgentInterpretation:
    """the simple rule based version of interpret(). used as a backup."""
    parts: list[str] = []
    flags: list[str] = []

    if rs is not None:
        easa_mm2 = rs.easa_mean_cm2 * 1e2
        # geometric area of a disk = pi * r^2. for diameter d in mm,
        # area in mm² is pi * (d/2)² = pi * d² / 4
        geom_area_mm2 = math.pi * (size_mm / 2) ** 2
        roughness = easa_mm2 / geom_area_mm2
        parts.append(
            f"EASA is {easa_mm2:.2f} mm² (roughness factor about {roughness:.2f} "
            f"for a {size_mm} mm electrode)."
        )
        if rs.anodic_fit.r_squared < 0.99 or rs.cathodic_fit.r_squared < 0.99:
            flags.append(
                f"randles sevcik R² is below 0.99 "
                f"(anodic {rs.anodic_fit.r_squared:.3f}, "
                f"cathodic {rs.cathodic_fit.r_squared:.3f}). "
                f"worth checking the peak picks at low scan rates."
            )

    if cdl_res is not None:
        parts.append(f"double layer capacitance is {cdl_res.cdl_microfarads:.2f} µF.")
        if cdl_res.fit.r_squared < 0.95:
            flags.append(
                f"cdl fit R² is {cdl_res.fit.r_squared:.3f}, lower than usual. "
                f"the pbs sweep might have some drift."
            )

    if lav is not None:
        if lav.k0_per_s is None:
            parts.append("laviron k0 wasn't resolved (see the caveat in the report).")
        else:
            parts.append(f"laviron ks came out around {lav.k0_per_s:.2e} s⁻¹ "
                         f"(alpha = {lav.alpha:.2f}, {lav.alpha_source}; "
                         f"averaged over {lav.n_replicates} replicate(s)).")

    summary = " ".join(parts) if parts else "no results to summarize."
    return AgentInterpretation(
        electrode_id=electrode_id,
        summary=summary,
        flags=flags,
    )


# -----------------------------------------------------------------------------
# the llm version. uses the anthropic api.
# -----------------------------------------------------------------------------


# what the model is allowed to pick from
_ALLOWED_ANALYSES = {"randles_sevcik", "cdl", "laviron", "nicholson"}


def _build_decide_prompt(experiments: list[CVExperiment]) -> str:
    """build the prompt that asks claude which analyses to run."""
    # summarize the data we have, without dumping the raw arrays
    summary_lines = [
        f"electrode id: {experiments[0].file_meta.electrode_id}",
        f"number of files: {len(experiments)}",
        f"scan rates (mV/s): {sorted({e.file_meta.scan_rate_mv_s for e in experiments})}",
        f"electrolytes seen: {sorted({e.file_meta.electrolyte.value for e in experiments})}",
        f"data points per file: about {experiments[0].n_points}",
    ]
    # if there are any peak results from the chi, include a snapshot
    sample_peaks = []
    for e in experiments[:3]:
        peaks = [(s.ep_v, s.ip_a) for s in e.segment_results if s.ep_v is not None]
        if peaks:
            sample_peaks.append(f"  {e.file_meta.scan_rate_mv_s:g} mV/s: {peaks}")
    if sample_peaks:
        summary_lines.append("sample peaks from chi (Ep V, ip A):")
        summary_lines.extend(sample_peaks)

    data_summary = "\n".join(summary_lines)

    return f"""you are helping with cyclic voltammetry analysis.

here is the data for one electrode:

{data_summary}

the available analyses are:
- "randles_sevcik": EASA from ip vs sqrt(scan rate). requires ferro/ferri data
  with clear redox peaks across multiple scan rates.
- "cdl": double layer capacitance from i at 0.6 V vs scan rate. requires pbs
  data (no faradaic process).
- "laviron": k0 from peak shift vs log(scan rate). requires ferro/ferri data
  with enough peak separation (delta_Ep > about 100 mV at high scan rates)
  to be in the kinetic regime. note that laviron strictly applies to surface
  confined species, so for diffusing ferro/ferri the result is at best an
  order of magnitude estimate.

decide which analyses to run for this electrode. respond with valid json only,
no markdown, no extra prose. format:

{{"analyses": ["randles_sevcik", "cdl"], "reasoning": "one or two short sentences here"}}

only include analyses from {sorted(_ALLOWED_ANALYSES)}.
"""


def _build_interpret_prompt(
    electrode_id: str,
    rs: RandlesSevcikResult | None,
    cdl_res: CdlResult | None,
    lav: LavironResult | None,
    size_mm: float = 1.0,
) -> str:
    """build the prompt that asks claude to interpret the results."""
    lines = [f"electrode id: {electrode_id}", ""]

    if rs is not None:
        import math as _math
        easa_mm2 = rs.easa_mean_cm2 * 1e2
        geom_mm2 = _math.pi * (size_mm / 2) ** 2
        roughness = easa_mm2 / geom_mm2
        lines.append("randles sevcik:")
        lines.append(f"  EASA mean: {easa_mm2:.3f} mm² (geometric for a {size_mm:g} mm disk is {geom_mm2:.3f} mm²)")
        lines.append(f"  roughness factor: {roughness:.2f}")
        lines.append(f"  anodic R²: {rs.anodic_fit.r_squared:.4f}")
        lines.append(f"  cathodic R²: {rs.cathodic_fit.r_squared:.4f}")
        lines.append("")

    if cdl_res is not None:
        lines.append("cdl:")
        lines.append(f"  capacitance: {cdl_res.cdl_microfarads:.4f} µF")
        lines.append(f"  R²: {cdl_res.fit.r_squared:.4f}")
        lines.append("")

    if lav is not None:
        lines.append("laviron:")
        if lav.k0_per_s is None:
            lines.append("  k0: not resolved")
            lines.append(f"  notes: {lav.notes}")
        else:
            lines.append(f"  ks: {lav.k0_per_s:.3e} s⁻¹ (Vc = {lav.vc_v_s:.3g} V/s)")
            lines.append(f"  alpha: {lav.alpha:.3f} ({lav.alpha_source})")
        lines.append("")

    summary_data = "\n".join(lines)

    return f"""here are the analysis results for one electrode:

{summary_data}

write a short interpretation (3 to 5 sentences total) of these results.
mention the surface area, capacitance, and any other key numbers. flag
anything that looks unusual, like low R² values or roughness factors outside
the typical 1.2 to 2.0 range for gold electrodes. respond with valid json only,
no markdown. format:

{{"summary": "your interpretation paragraph here", "flags": ["any concerns, one per item"]}}

if there's nothing to flag, return an empty list for flags.
"""


def _call_claude(prompt: str, api_key: str) -> dict:
    """call the anthropic api and return the parsed json response.

    we ask for json out, then parse it. if parsing fails we raise, and the
    caller falls back to the deterministic version.
    """
    # imported here so the project doesn't need anthropic installed unless
    # the user actually wants the agent features
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    # claude returns a list of content blocks. for our prompts we just want
    # the text from the first block.
    text = message.content[0].text.strip()
    # sometimes claude wraps json in code fences even when we ask it not to.
    # strip those if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return json.loads(text)


# -----------------------------------------------------------------------------
# the public api. these are the two functions the runner calls.
# -----------------------------------------------------------------------------


def decide(experiments: list[CVExperiment],
           use_llm: bool | None = None) -> AgentDecision:
    """which analyses apply to this electrode. purely deterministic.

    this is the protocol's own routing rule and nothing else: ferro/ferri
    data gets randles sevcik plus one kinetics method chosen by the mean
    peak separation (>212 mV laviron, 61-212 mV nicholson), pbs data gets
    cdl. the use_llm argument is kept so old callers don't break, but it is
    ignored: method choice must be reproducible, so no llm is consulted.
    """
    return _deterministic_decide(experiments)


def interpret(
    electrode_id: str,
    rs: RandlesSevcikResult | None,
    cdl_res: CdlResult | None,
    lav: LavironResult | None,
    size_mm: float = 1.0,
    use_llm: bool | None = None,
) -> AgentInterpretation:
    """ask the agent to write a human readable summary of the results.

    same auto detection as decide(). uses the llm if we have a key, otherwise
    builds a deterministic summary from the same numbers.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if use_llm is None:
        use_llm = api_key is not None

    if not use_llm or api_key is None:
        return _deterministic_interpret(electrode_id, rs, cdl_res, lav)

    try:
        prompt = _build_interpret_prompt(electrode_id, rs, cdl_res, lav, size_mm)
        response = _call_claude(prompt, api_key)
        return AgentInterpretation(
            electrode_id=electrode_id,
            summary=response.get("summary", ""),
            flags=list(response.get("flags", [])),
        )
    except Exception as exc:
        print(f"heads up, agent interpret call failed ({exc}). falling back to rules.")
        return _deterministic_interpret(electrode_id, rs, cdl_res, lav)


# -----------------------------------------------------------------------------
# cross electrode comparison. runs once at the end, after every electrode has
# been individually analyzed. this is where the agent gets to be useful in a
# way that a hardcoded script can't, spotting patterns across the whole batch.
# -----------------------------------------------------------------------------


def _deterministic_compare(rows: list[dict]) -> AgentComparison:
    """rule based version of compare(). no llm, just simple stats."""
    if not rows:
        return AgentComparison(
            overview="no electrodes to compare.",
            patterns=[], outliers=[], recommendations=[],
        )

    # split into ferro/ferri and pbs electrodes based on what numbers they have
    ff_rows = [r for r in rows if r.get("easa_mm2") is not None]
    pbs_rows = [r for r in rows if r.get("cdl_uF") is not None]

    overview_parts: list[str] = [
        f"looked at {len(rows)} electrodes total."
    ]
    if ff_rows:
        easas = [r["easa_mm2"] for r in ff_rows]
        mean_easa = sum(easas) / len(easas)
        overview_parts.append(
            f"{len(ff_rows)} ferro/ferri electrodes, mean EASA {mean_easa:.2f} mm² "
            f"(range {min(easas):.2f} to {max(easas):.2f})."
        )
    if pbs_rows:
        cdls = [r["cdl_uF"] for r in pbs_rows]
        mean_cdl = sum(cdls) / len(cdls)
        overview_parts.append(
            f"{len(pbs_rows)} pbs electrodes, mean Cdl {mean_cdl:.2f} µF "
            f"(range {min(cdls):.2f} to {max(cdls):.2f})."
        )

    patterns: list[str] = []
    outliers: list[str] = []
    recommendations: list[str] = []

    # ferro/ferri outliers, more than one standard deviation from the mean
    if len(ff_rows) >= 3:
        easas = [r["easa_mm2"] for r in ff_rows]
        mean_easa = sum(easas) / len(easas)
        # simple population std dev
        var = sum((e - mean_easa) ** 2 for e in easas) / len(easas)
        std_easa = var ** 0.5
        for r in ff_rows:
            diff = r["easa_mm2"] - mean_easa
            if abs(diff) > std_easa and std_easa > 0:
                direction = "below" if diff < 0 else "above"
                outliers.append(
                    f"{r['electrode_id']}: EASA {r['easa_mm2']:.2f} mm² is "
                    f"more than 1σ {direction} the batch mean of {mean_easa:.2f} mm²."
                )

    # pbs outliers, same idea
    if len(pbs_rows) >= 3:
        cdls = [r["cdl_uF"] for r in pbs_rows]
        mean_cdl = sum(cdls) / len(cdls)
        var = sum((c - mean_cdl) ** 2 for c in cdls) / len(cdls)
        std_cdl = var ** 0.5
        for r in pbs_rows:
            diff = r["cdl_uF"] - mean_cdl
            if abs(diff) > std_cdl and std_cdl > 0:
                direction = "below" if diff < 0 else "above"
                outliers.append(
                    f"{r['electrode_id']}: Cdl {r['cdl_uF']:.2f} µF is "
                    f"more than 1σ {direction} the batch mean of {mean_cdl:.2f} µF."
                )

    # bad fits across the batch
    bad_fits = [r["electrode_id"] for r in ff_rows
                if r.get("rs_r2_anodic", 1.0) < 0.99 or r.get("rs_r2_cathodic", 1.0) < 0.99]
    if bad_fits:
        patterns.append(
            f"randles sevcik R² is below 0.99 for {', '.join(bad_fits)}. "
            f"could be peak picking, could be drift at low scan rates."
        )

    bad_cdl_fits = [r["electrode_id"] for r in pbs_rows
                    if r.get("cdl_r2", 1.0) < 0.95]
    if bad_cdl_fits:
        patterns.append(
            f"cdl fit R² is below 0.95 for {', '.join(bad_cdl_fits)}. "
            f"the pbs sweep may have drifted on those."
        )

    # roughness factor sanity, geometric area of a 1 mm disk is 0.785 mm²
    if ff_rows:
        import math
        roughnesses = {}
        for r in ff_rows:
            size = r.get("electrode_size_mm", 1)
            geom = math.pi * (size / 2) ** 2
            roughnesses[r["electrode_id"]] = r["easa_mm2"] / geom
        weird = [eid for eid, rf in roughnesses.items() if rf < 1.0 or rf > 2.5]

    if outliers:
        recommendations.append("revisit the outlier electrodes, either remeasure or check fabrication notes.")
    if bad_fits or bad_cdl_fits:
        recommendations.append("look at the cv overlay plots for the low R² electrodes to see if peaks are clean.")
    if not recommendations:
        recommendations.append("batch looks consistent. ready to move on to the next characterization step.")

    return AgentComparison(
        overview=" ".join(overview_parts),
        patterns=patterns,
        outliers=outliers,
        recommendations=recommendations,
    )




def compare(rows: list[dict], use_llm: bool | None = None) -> AgentComparison:
    """compare results across all electrodes, find patterns and outliers.

    all the stats (means, ranges, 1σ outliers, low-R² fits) are computed in
    code, every time. if an api key is set, the llm is allowed to rewrite the
    overview paragraph in friendlier prose from those computed numbers, but
    the patterns, outliers, and recommendations are always the computed ones.
    """
    det = _deterministic_compare(rows)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if use_llm is None:
        use_llm = api_key is not None
    if not use_llm or api_key is None or not rows:
        return det

    try:
        prompt = (
            "here is a computed statistical summary of a batch of cyclic "
            "voltammetry results:\n\n"
            f"overview: {det.overview}\n"
            f"patterns: {det.patterns}\n"
            f"outliers: {det.outliers}\n\n"
            "rewrite the overview as one friendly, readable paragraph for a "
            "lab notebook. do not add, change, or recompute any number, and "
            "do not add or remove outliers. respond with valid json only, "
            'no markdown: {"overview": "..."}'
        )
        response = _call_claude(prompt, api_key)
        overview = response.get("overview") or det.overview
        return AgentComparison(
            overview=overview,
            patterns=det.patterns,
            outliers=det.outliers,
            recommendations=det.recommendations,
        )
    except Exception as exc:
        print(f"heads up, llm phrasing failed ({exc}). using the computed text.")
        return det
