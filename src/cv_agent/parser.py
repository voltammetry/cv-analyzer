"""reads the .csv files the chi potentiostat spits out (model CHI1040C and friends).

here's roughly what one of these files looks like top to bottom

    <timestamp>
    Cyclic Voltammetry
    File: <original path>
    Data Source:  Experiment
    Instrument Model:  CHI1040C
    Header:
    Note:

    Init E (V) = ...
    High E (V) = ...
    ...
    Sensitivity (A/V) = ...

    Results:

    Channel 1:
    Segment 1:
    Ep = 0.273V
    ip = 4.046e-6A
    Ah = 5.570e-5C

    Segment 2:
    ...

    Potential/V, Current/A

    -0.200, -3.964e-6
    -0.199, -3.815e-6
    ...

so there's three chunks. a header block with the settings, a results block
where the machine reports the peaks it found, and then the actual two column
data table at the bottom. we read all three.

one thing to know, the pbs files have empty segment blocks (no Ep/ip/Ah lines)
because there's no peak to find. we keep those as SegmentResult with None values
instead of dropping them, so the segment count still matches what the machine said.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .filename import parse_filename
from .models import (
    CVExperiment,
    FileMetadata,
    InstrumentMetadata,
    SegmentResult,
)


# the header lines all look like  Init E (V) = 0.5  so we have one regex per
# setting we care about. the \s* bits just mean we don't care about extra spaces.
_HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "init_e_v":          re.compile(r"^Init E\s*\(V\)\s*=\s*(-?[\d.]+)"),
    "high_e_v":          re.compile(r"^High E\s*\(V\)\s*=\s*(-?[\d.]+)"),
    "low_e_v":           re.compile(r"^Low E\s*\(V\)\s*=\s*(-?[\d.]+)"),
    "final_e_v":         re.compile(r"^Final E\s*\(V\)\s*=\s*(-?[\d.]+)"),
    "init_polarity":     re.compile(r"^Init P/N\s*=\s*([PN])"),
    "scan_rate_v_s":     re.compile(r"^Scan Rate\s*\(V/s\)\s*=\s*([\d.eE+-]+)"),
    "segments":          re.compile(r"^Segment\s*=\s*(\d+)"),
    "sample_interval_v": re.compile(r"^Sample Interval\s*\(V\)\s*=\s*([\d.eE+-]+)"),
    "quiet_time_s":      re.compile(r"^Quiet Time\s*\(sec\)\s*=\s*([\d.eE+-]+)"),
    "sensitivity_a_v":   re.compile(r"^Sensitivity\s*\(A/V\)\s*=\s*([\d.eE+-]+)"),
}

# the timestamp at the very top, looks like  May 12, 2026   14:21:22
_TIMESTAMP_RE = re.compile(
    r"^([A-Z][a-z]+ \d{1,2}, \d{4})\s+(\d{1,2}:\d{2}:\d{2})"
)

# the lines inside the results block
_SEGMENT_HEADER_RE = re.compile(r"^Segment\s+(\d+)\s*:")
_EP_RE = re.compile(r"^Ep\s*=\s*(-?[\d.eE+-]+)V")
_IP_RE = re.compile(r"^ip\s*=\s*(-?[\d.eE+-]+)A")
_AH_RE = re.compile(r"^Ah\s*=\s*(-?[\d.eE+-]+)C")

# the line that marks the start of the data table
_DATA_HEADER_RE = re.compile(r"^Potential/V\s*,\s*Current/A")


class CVParseError(ValueError):
    """thrown when we can't make sense of a file."""


def parse_cv_file(path: str | Path) -> CVExperiment:
    """read one chi .csv file and hand back a CVExperiment.

    raises CVParseError if the file is broken, or FilenameParseError if the
    name is off pattern.
    """
    path = Path(path)
    if not path.exists():
        raise CVParseError(f"can't find the file {path}")

    # name first, since that's quick and tells us electrode and electrolyte
    file_meta = parse_filename(path)

    # chi writes these with windows line endings and the odd trailing space, so
    # we read the whole thing as text, strip each line, then walk through it.
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines()]

    # pull each piece out one at a time
    timestamp = _extract_timestamp(lines)
    technique, instrument_model = _extract_technique_and_model(lines)
    header_values = _extract_header_values(lines)
    segment_results = _extract_segment_results(lines)
    potential, current = _extract_data_table(lines)

    instrument_meta = InstrumentMetadata(
        timestamp=timestamp,
        technique=technique,
        instrument_model=instrument_model,
        **header_values,
    )

    return CVExperiment(
        file_meta=file_meta,
        instrument_meta=instrument_meta,
        segment_results=tuple(segment_results),
        potential_v=potential,
        current_a=current,
    )


# ---------------------------------------------------------------------------
# the little helpers that each grab one part of the file
# ---------------------------------------------------------------------------


def _extract_timestamp(lines: list[str]) -> datetime | None:
    # the timestamp is in the first few lines. if we find it we parse it into a
    # real datetime, and if the format is weird we just give back None rather
    # than blowing up, since the timestamp isn't critical.
    for ln in lines[:5]:
        m = _TIMESTAMP_RE.match(ln)
        if m:
            try:
                return datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%B %d, %Y %H:%M:%S"
                )
            except ValueError:
                return None
    return None


def _extract_technique_and_model(lines: list[str]) -> tuple[str, str]:
    # grab the technique name and the instrument model from near the top. we
    # skip the File: line so we don't accidentally match the path text.
    technique = ""
    model = ""
    for ln in lines[:10]:
        if ln and not ln.startswith("File:") and "Cyclic Voltammetry" in ln:
            technique = "Cyclic Voltammetry"
        if ln.startswith("Instrument Model:"):
            model = ln.split(":", 1)[1].strip()
    return technique, model


def _extract_header_values(lines: list[str]) -> dict[str, object]:
    # walk every line and try each header regex on it. once we've found a value
    # we don't overwrite it, so the first match wins. polarity stays a string,
    # segments is an int, everything else is a float.
    values: dict[str, object] = {}
    for ln in lines:
        for key, pattern in _HEADER_PATTERNS.items():
            if key in values:
                continue
            m = pattern.match(ln)
            if m:
                raw = m.group(1)
                if key == "init_polarity":
                    values[key] = raw
                elif key == "segments":
                    values[key] = int(raw)
                else:
                    values[key] = float(raw)

    # if any setting is missing the file is probably not what we think it is,
    # so bail out and say which ones we couldn't find.
    missing = set(_HEADER_PATTERNS.keys()) - values.keys()
    if missing:
        raise CVParseError(
            f"the header is missing some fields we need. {sorted(missing)}"
        )
    return values


def _extract_segment_results(lines: list[str]) -> list[SegmentResult]:
    # this reads the results block, where the machine lists the peak it found
    # for each segment. empty blocks (the pbs case) come back as segments with
    # None values so we don't lose the count.

    # first find where the results block starts and where the data table starts,
    # since the results block sits between the two.
    results_start = None
    data_start = None
    for i, ln in enumerate(lines):
        if ln == "Results:" and results_start is None:
            results_start = i
        if _DATA_HEADER_RE.match(ln):
            data_start = i
            break

    if results_start is None or data_start is None:
        return []

    block = lines[results_start:data_start]

    segments: list[SegmentResult] = []
    current_seg: int | None = None
    ep: float | None = None
    ip: float | None = None
    ah: float | None = None

    # we read line by line and build up one segment at a time. when we hit a new
    # "Segment N:" line we flush whatever we'd collected for the previous one.
    def flush() -> None:
        nonlocal current_seg, ep, ip, ah
        if current_seg is not None:
            segments.append(SegmentResult(segment=current_seg, ep_v=ep, ip_a=ip, ah_c=ah))
        current_seg = None
        ep = ip = ah = None

    for ln in block:
        m_seg = _SEGMENT_HEADER_RE.match(ln)
        if m_seg:
            flush()
            current_seg = int(m_seg.group(1))
            continue
        m = _EP_RE.match(ln)
        if m and current_seg is not None:
            ep = float(m.group(1))
            continue
        m = _IP_RE.match(ln)
        if m and current_seg is not None:
            ip = float(m.group(1))
            continue
        m = _AH_RE.match(ln)
        if m and current_seg is not None:
            ah = float(m.group(1))
            continue

    flush()  # don't forget the last segment
    return segments


def _extract_data_table(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # grab the two column potential, current block at the bottom of the file.
    data_start = None
    for i, ln in enumerate(lines):
        if _DATA_HEADER_RE.match(ln):
            data_start = i + 1  # data starts on the line after the header
            break

    if data_start is None:
        raise CVParseError("couldn't find the Potential/V, Current/A line")

    potentials: list[float] = []
    currents: list[float] = []
    for ln in lines[data_start:]:
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 2:
            continue
        # anything that isn't two numbers we just skip quietly. covers blank
        # lines and any trailing junk.
        try:
            potentials.append(float(parts[0]))
            currents.append(float(parts[1]))
        except ValueError:
            continue

    if not potentials:
        raise CVParseError("found the data header but no actual rows under it")

    return np.asarray(potentials, dtype=float), np.asarray(currents, dtype=float)


def parse_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    skip_on_error: bool = True,
) -> list[CVExperiment]:
    """parse every cv file in a folder and return them as a list.

    by default if one file fails we print a warning and keep going, so one bad
    file doesn't stop the whole run. set skip_on_error to False if you'd rather
    have it stop on the first problem.
    """
    directory = Path(directory)
    experiments: list[CVExperiment] = []
    for fp in sorted(directory.glob(pattern)):
        try:
            experiments.append(parse_cv_file(fp))
        except (CVParseError, ValueError) as exc:
            if skip_on_error:
                print(f"heads up, skipping {fp.name} because {exc}")
                continue
            raise
    return experiments
