"""the data models for a parsed cv file.

basically these are the containers that hold everything once we've read a file
off disk. the parser fills these in, and every analysis later on (randles
sevcik, cdl, laviron) just reads from a CVExperiment. so this file is the
shared shape that the whole project agrees on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np


class Electrolyte(Enum):
    """which solution the experiment was run in.

    this matters because it decides what analysis makes sense.
    ferro/ferri is a redox couple floating in solution, so it gives peaks and
    we do randles sevcik on it. pbs has nothing reacting, so we just measure
    the charging current and get cdl out of it.
    """

    FERRO_FERRI = "ferro_ferri"
    PBS = "pbs"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FileMetadata:
    """everything we can figure out just from the filename, no need to open it.

    the naming pattern we follow is
        cv_sp1_<electrode_id>_<scan_rate>mvs_<MMDDYY>_<batch_code>_ff.csv
    so for example cv_sp1_ab2_5mvs_051226_0206_ff.csv tells us electrode ab2,
    5 mV/s, may 12 2026, batch 0206.
    """

    electrode_id: str           # like ab2 or hb3
    scan_rate_mv_s: float       # mV/s, straight from the filename
    experiment_date: str        # the MMDDYY chunk, left as a string on purpose
    batch_code: str             # like 0206 or 0507, tells us which electrolyte
    electrolyte: Electrolyte
    source_path: Path
    electrode_size_mm: int       # diameter in mm, from the sp prefix (sp1 = 1mm, etc)


@dataclass(frozen=True)
class InstrumentMetadata:
    """the stuff the chi machine writes in the header at the top of the file."""

    timestamp: datetime | None
    technique: str               # always cyclic voltammetry for us
    instrument_model: str        # the chi model, like CHI1040C
    init_e_v: float
    high_e_v: float
    low_e_v: float
    final_e_v: float
    init_polarity: str           # P or N
    scan_rate_v_s: float         # V/s, what the machine itself reports
    segments: int
    sample_interval_v: float
    quiet_time_s: float
    sensitivity_a_v: float


@dataclass(frozen=True)
class SegmentResult:
    """the peak the machine found for one segment of the sweep.

    the chi does its own peak picking and writes it in the results block. we
    hold onto it so later we can cross check it against our own peak finding.
    on pbs files these come back empty (all None) since there's no real peak.
    """

    segment: int
    ep_v: float | None           # peak potential, volts
    ip_a: float | None           # peak current, amps
    ah_c: float | None           # charge, coulombs


@dataclass(frozen=True)
class CVExperiment:
    """one fully parsed cv file. this is the thing every analysis eats."""

    file_meta: FileMetadata
    instrument_meta: InstrumentMetadata
    segment_results: tuple[SegmentResult, ...]
    potential_v: np.ndarray = field(repr=False)   # the voltage column, volts
    current_a: np.ndarray = field(repr=False)     # the current column, amps

    def __post_init__(self) -> None:
        # quick gut check. potential and current come from the same two column
        # table, so if they don't line up something went wrong in parsing.
        if self.potential_v.shape != self.current_a.shape:
            raise ValueError(
                f"potential and current arrays don't match. "
                f"got {self.potential_v.shape} vs {self.current_a.shape}"
            )
        if self.potential_v.ndim != 1:
            raise ValueError(f"expected a flat 1d array, got {self.potential_v.ndim}d")

    @property
    def n_points(self) -> int:
        # how many data points are in this sweep
        return int(self.potential_v.size)

    @property
    def scan_rate_v_s(self) -> float:
        # one place to ask for scan rate. we trust the machine's number over
        # the filename if they ever disagree.
        return self.instrument_meta.scan_rate_v_s
