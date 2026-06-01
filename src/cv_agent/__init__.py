"""cv_agent, automated electrochemical analysis for cyclic voltammetry data.

this pulls the main things up to the top level so you can do
    from cv_agent import parse_cv_file, randles_sevcik, decide
instead of digging into submodules every time.
"""

from .agent import (
    AgentComparison,
    AgentDecision,
    AgentInterpretation,
    compare,
    decide,
    interpret,
)
from .analysis import (
    CdlResult,
    LavironResult,
    LinearFit,
    RandlesSevcikResult,
    cdl,
    laviron,
    randles_sevcik,
)
from .filename import FilenameParseError, parse_filename
from .models import (
    CVExperiment,
    Electrolyte,
    FileMetadata,
    InstrumentMetadata,
    SegmentResult,
)
from .parser import CVParseError, parse_cv_file, parse_directory
from .report import (
    plot_cdl,
    plot_cv_overlay,
    plot_laviron,
    plot_randles_sevcik,
    write_electrode_report,
)

__all__ = [
    # models
    "CVExperiment",
    "Electrolyte",
    "FileMetadata",
    "InstrumentMetadata",
    "SegmentResult",
    # parsing
    "CVParseError",
    "FilenameParseError",
    "parse_cv_file",
    "parse_directory",
    "parse_filename",
    # analysis
    "CdlResult",
    "LavironResult",
    "LinearFit",
    "RandlesSevcikResult",
    "cdl",
    "laviron",
    "randles_sevcik",
    # report
    "plot_cdl",
    "plot_cv_overlay",
    "plot_laviron",
    "plot_randles_sevcik",
    "write_electrode_report",
    # agent
    "AgentComparison",
    "AgentDecision",
    "AgentInterpretation",
    "compare",
    "decide",
    "interpret",
]
