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
    NicholsonResult,
    PeakTable,
    RandlesSevcikResult,
    ReplicateTable,
    average_peak_tables,
    cdl,
    choose_kinetics_method,
    laviron,
    nicholson,
    peak_table,
    randles_sevcik,
    randles_sevcik_from_replicates,
)
from .filename import FilenameParseError, parse_filename
from .models import (
    CVExperiment,
    Electrolyte,
    FileMetadata,
    InstrumentMetadata,
    SegmentResult,
    default_replicate_group,
)
from .parser import (CVParseError, classify_electrolyte, parse_cv_file,
                     parse_directory, read_manifest)
from .report import (
    plot_cdl,
    plot_cv_overlay,
    plot_laviron,
    plot_nicholson,
    plot_randles_sevcik,
    write_electrode_report,
    write_group_report,
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
    "read_manifest",
    "classify_electrolyte",
    "default_replicate_group",
    # analysis
    "CdlResult",
    "LavironResult",
    "LinearFit",
    "NicholsonResult",
    "PeakTable",
    "RandlesSevcikResult",
    "ReplicateTable",
    "average_peak_tables",
    "cdl",
    "choose_kinetics_method",
    "laviron",
    "nicholson",
    "peak_table",
    "randles_sevcik",
    "randles_sevcik_from_replicates",
    # report
    "plot_cdl",
    "plot_cv_overlay",
    "plot_laviron",
    "plot_nicholson",
    "plot_randles_sevcik",
    "write_electrode_report",
    "write_group_report",
    # agent
    "AgentComparison",
    "AgentDecision",
    "AgentInterpretation",
    "compare",
    "decide",
    "interpret",
]