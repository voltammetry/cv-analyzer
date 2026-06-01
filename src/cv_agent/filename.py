"""pull metadata out of a cv filename.

the naming convention we stick to is all lowercase and looks like
    cv_sp1_<electrode_id>_<scan_rate>mvs_<MMDDYY>_<batch_code>_ff.csv

couple examples
    cv_sp1_ab2_5mvs_051226_0206_ff.csv   means ab2, 5 mV/s, may 12 26, batch 0206, ferro/ferri
    cv_sp1_hb3_250mvs_051826_0507.csv    means hb3, 250 mV/s, may 18 26, batch 0507, pbs

heads up, the pbs files don't have the _ff on the end, so that part is optional.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Electrolyte, FileMetadata


# which batch code maps to which solution. add to this as you run new batches.
_BATCH_TO_ELECTROLYTE: dict[str, Electrolyte] = {
    "0206": Electrolyte.FERRO_FERRI,
    "0507": Electrolyte.PBS,
}


# this is the regex that actually picks the filename apart. we lowercase the
# name before matching so the odd uppercase file still goes through fine. the
# named groups (electrode_id, scan_rate, etc) are what we pull out after.
_FILENAME_RE = re.compile(
    r"""
    ^cv_sp
    (?P<size_mm>\d+)_                  # 1, 2, 3 = electrode diameter in mm
    (?P<electrode_id>[a-z]+\d+(?:-\d+)?) # ab2, hb3, a4-1, x6, etc
    _
    (?P<scan_rate>\d+(?:\.\d+)?)mvs    # 5mvs, 250mvs, etc
    _
    (?P<date>\d{6})                    # MMDDYY
    _
    -?                                 # optional minus sign before the batch code
    (?P<batch>\d{4})                   # the batch code, like 0206
    (?:_ff)?                           # optional _ff tag on the ferro/ferri ones
    \.csv$
    """,
    re.VERBOSE,
)

class FilenameParseError(ValueError):
    """thrown when a filename doesn't match the naming convention."""


def parse_filename(path: str | Path) -> FileMetadata:
    """take a file path and pull the metadata out of its name.

    we only look at the filename itself here, not the contents. returns a
    FileMetadata with electrode id, scan rate, electrolyte and so on. raises
    FilenameParseError if the name is off pattern.
    """
    path = Path(path)
    name = path.name.lower()  # lowercase first so an uppercase outlier still works

    match = _FILENAME_RE.match(name)
    if not match:
        raise FilenameParseError(
            f"the filename {path.name!r} doesn't match what we expect, "
            f"which is cv_sp1_<electrode>_<scan>mvs_<MMDDYY>_<batch>[_ff].csv"
        )

    # look up the electrolyte from the batch code. if it's a batch we haven't
    # seen before we just mark it unknown rather than guessing.
    batch_code = match.group("batch")
    electrolyte = _BATCH_TO_ELECTROLYTE.get(batch_code, Electrolyte.UNKNOWN)

    return FileMetadata(
        electrode_id=match.group("electrode_id"),
        scan_rate_mv_s=float(match.group("scan_rate")),
        experiment_date=match.group("date"),
        batch_code=batch_code,
        electrolyte=electrolyte,
        source_path=path,
        electrode_size_mm=int(match.group("size_mm")),
    )
