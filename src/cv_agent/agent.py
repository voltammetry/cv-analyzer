"""the actual ai agent layer.

so far the analyses are just plain functions. this file is what makes the
project an agent, not just a pipeline. there's two things the agent does that
a hardcoded script can't.

  1. it decides which analyses to run. given a list of cv files for one
     electrode, it looks at what we have (electrolyte, scan rates, peak
     separations, fit r squared) and picks the right analyses for that
     situation. for the simple cases this matches what a hardcoded rule would
     do (ferro/ferri gets randles sevcik and laviron, pbs gets cdl). but the
     llm can also catch weird cases, like flagging that laviron won't work on
     a given electrode because the peak separations are too small.

  2. it writes a short human readable interpretation of the results. things
     like "this electrode has a roughness factor of about 1.6, which is in
     the normal range for laser cut gold" or "the r squared on the cathodic
     fit is lower than the anodic one, worth a closer look at the lower scan
     rates." the kind of thing you'd write in your lab notebook after
     looking at the report.

the agent uses anthropic's claude api. if no api key is set, it falls back to
a deterministic routing function so the project still works end to end without
the agent, just without the smart parts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .analysis import CdlResult, LavironResult, RandlesSevcikResult
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
        if sep_mv is not None and sep_mv > 212:
            analyses.add("laviron")
            reasons.append(f"ferro/ferri; ΔEp {sep_mv:.0f} mV > 212, so randles sevcik + laviron")
        elif sep_mv is not None and sep_mv >= 61:
            analyses.add("nicholson")
            reasons.append(f"ferro/ferri; ΔEp {sep_mv:.0f} mV in 61-212, so randles sevcik + nicholson")
        else:
            analyses.add("laviron")
            reasons.append("ferro/ferri; ΔEp unknown, defaulting to randles sevcik + laviron")
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
        import math
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
            parts.append(f"laviron k0 came out around {lav.k0_per_s:.2e} s⁻¹ "
                         f"(alpha = {lav.alpha:.2f}), order of magnitude only.")

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
) -> str:
    """build the prompt that asks claude to interpret the results."""
    lines = [f"electrode id: {electrode_id}", ""]

    if rs is not None:
        easa_mm2 = rs.easa_mean_cm2 * 1e2
        roughness = easa_mm2 / 0.785
        lines.append("randles sevcik:")
        lines.append(f"  EASA mean: {easa_mm2:.3f} mm² (geometric for 1 mm disk is 0.785 mm²)")
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
            lines.append(f"  k0: {lav.k0_per_s:.3e} s⁻¹")
            lines.append(f"  alpha: {lav.alpha:.3f}")
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
    """ask the agent which analyses to run for this electrode.

    if use_llm is None, we auto detect. we use the llm if ANTHROPIC_API_KEY is
    set in the environment, otherwise we fall back to the deterministic rules.
    if use_llm is True or False, we honor it.
    """
    if not experiments:
        raise ValueError("decide, got an empty experiment list")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if use_llm is None:
        use_llm = api_key is not None

    if not use_llm or api_key is None:
        return _deterministic_decide(experiments)

    # try the llm. if anything goes wrong, fall back.
    try:
        prompt = _build_decide_prompt(experiments)
        response = _call_claude(prompt, api_key)
        analyses = set(response.get("analyses", [])) & _ALLOWED_ANALYSES
        reasoning = response.get("reasoning", "")
        return AgentDecision(
            electrode_id=experiments[0].file_meta.electrode_id,
            analyses_to_run=analyses,
            reasoning=reasoning,
        )
    except Exception as exc:
        print(f"heads up, agent decide call failed ({exc}). falling back to rules.")
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
        prompt = _build_interpret_prompt(electrode_id, rs, cdl_res, lav)
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


def _build_compare_prompt(rows: list[dict]) -> str:
    """build the prompt that asks claude to compare across all electrodes."""
    # serialize the rows as a clean readable table
    table_lines = []
    for r in rows:
        parts = [f"electrode {r['electrode_id']}"]
        if r.get("easa_mm2") is not None:
            roughness = r["easa_mm2"] / 0.785
            parts.append(f"EASA={r['easa_mm2']:.3f} mm² (roughness {roughness:.2f})")
            parts.append(f"RS R² anodic={r.get('rs_r2_anodic', '?'):.4f}")
            parts.append(f"cathodic={r.get('rs_r2_cathodic', '?'):.4f}")
        if r.get("cdl_uF") is not None:
            parts.append(f"Cdl={r['cdl_uF']:.4f} µF")
            parts.append(f"R²={r.get('cdl_r2', '?'):.4f}")
        if r.get("laviron_k0_per_s") is not None:
            parts.append(f"k0={r['laviron_k0_per_s']:.2e} s⁻¹")
        table_lines.append("  " + ", ".join(parts))

    table = "\n".join(table_lines)

    return f"""you are reviewing a batch of cyclic voltammetry results across multiple electrodes.

the electrodes were all fabricated using the same process (laser cut gold on
polyimide with kapton tape masking). for a 1 mm working electrode the
geometric area is 0.785 mm², so a roughness factor of 1.2 to 2.0 is normal.

here are the per electrode results:

{table}

give me a cross electrode read on this batch. look for patterns (e.g. are
some electrodes systematically worse?), outliers (which ones stand out and
why), and recommendations (what should be done next). respond with valid
json only, no markdown, no code fences. format:

{{
  "overview": "one short paragraph summarizing the batch as a whole",
  "patterns": ["specific observation 1", "specific observation 2"],
  "outliers": ["electrode_id: why it stands out", "..."],
  "recommendations": ["concrete next step 1", "..."]
}}

be specific. reference electrode ids by name. keep it concise."""


def compare(rows: list[dict], use_llm: bool | None = None) -> AgentComparison:
    """compare results across all electrodes, find patterns and outliers.

    takes the same list of dict rows that we'd write to summary.csv, one per
    electrode. each row has keys like electrode_id, easa_mm2, cdl_uF, etc.
    same auto detection as decide() and interpret().
    """
    if not rows:
        return AgentComparison(
            overview="no electrodes to compare.",
            patterns=[], outliers=[], recommendations=[],
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if use_llm is None:
        use_llm = api_key is not None

    if not use_llm or api_key is None:
        return _deterministic_compare(rows)

    try:
        prompt = _build_compare_prompt(rows)
        response = _call_claude(prompt, api_key)
        return AgentComparison(
            overview=response.get("overview", ""),
            patterns=list(response.get("patterns", [])),
            outliers=list(response.get("outliers", [])),
            recommendations=list(response.get("recommendations", [])),
        )
    except Exception as exc:
        print(f"heads up, agent compare call failed ({exc}). falling back to rules.")
        return _deterministic_compare(rows)
