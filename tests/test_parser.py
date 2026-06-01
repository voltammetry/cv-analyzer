"""tests for parser and filename modules. runs against the real data in data/raw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cv_agent import (
    CVParseError,
    Electrolyte,
    FilenameParseError,
    parse_cv_file,
    parse_directory,
    parse_filename,
)


# -----------------------------------------------------------------------------
# filename parsing
# -----------------------------------------------------------------------------


class TestFilenameParsing:
    def test_ferro_ferri_file(self) -> None:
        meta = parse_filename("cv_sp1_ab2_5mvs_051226_-0206_ff.csv")
        assert meta.electrode_id == "ab2"
        assert meta.scan_rate_mv_s == 5.0
        assert meta.experiment_date == "051226"
        assert meta.batch_code == "0206"
        assert meta.electrolyte == Electrolyte.FERRO_FERRI

    def test_pbs_file_no_ff_suffix(self) -> None:
        meta = parse_filename("cv_sp1_hb3_250mvs_051826_0507.csv")
        assert meta.electrode_id == "hb3"
        assert meta.scan_rate_mv_s == 250.0
        assert meta.batch_code == "0507"
        assert meta.electrolyte == Electrolyte.PBS

    def test_uppercase_outlier_is_normalized(self) -> None:
        # the one uppercase file should still parse the same as lowercase
        meta = parse_filename("CV_SP1_AB6_5mVs_051327_-0206_FF.csv")
        assert meta.electrode_id == "ab6"
        assert meta.scan_rate_mv_s == 5.0
        assert meta.electrolyte == Electrolyte.FERRO_FERRI

    def test_three_digit_scan_rate(self) -> None:
        meta = parse_filename("cv_sp1_ab4_250mvs_051226_-0206_ff.csv")
        assert meta.scan_rate_mv_s == 250.0

    def test_invalid_filename_raises(self) -> None:
        with pytest.raises(FilenameParseError):
            parse_filename("random_garbage.csv")

    def test_unknown_batch_code(self) -> None:
        meta = parse_filename("cv_sp1_xy1_50mvs_010125_9999_ff.csv")
        assert meta.electrolyte == Electrolyte.UNKNOWN
    
    def test_extracts_electrode_size(self) -> None:
        meta = parse_filename("cv_sp2_a4-1_100mvs_052026_0507.csv")
        assert meta.electrode_size_mm == 2
        meta = parse_filename("cv_sp1_ab2_5mvs_051226_-0206_ff.csv")
        assert meta.electrode_size_mm == 1


# -----------------------------------------------------------------------------
# full file parsing against real data
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class TestCVFileParsing:
    def test_parse_a_ferro_ferri_file(self) -> None:
        path = DATA_DIR / "cv_sp1_ab2_5mvs_051226_-0206_ff.csv"
        exp = parse_cv_file(path)

        # file metadata
        assert exp.file_meta.electrode_id == "ab2"
        assert exp.file_meta.electrolyte == Electrolyte.FERRO_FERRI

        # instrument metadata
        assert exp.instrument_meta.scan_rate_v_s == pytest.approx(0.005)
        assert exp.instrument_meta.segments == 4
        assert exp.instrument_meta.init_e_v == pytest.approx(-0.2)
        assert exp.instrument_meta.high_e_v == pytest.approx(0.6)

        # segment results
        assert len(exp.segment_results) == 4
        seg1 = exp.segment_results[0]
        assert seg1.segment == 1
        assert seg1.ep_v == pytest.approx(0.273)
        assert seg1.ip_a == pytest.approx(4.046e-6)

        # data table
        assert exp.n_points > 100
        assert exp.potential_v.shape == exp.current_a.shape
        assert exp.potential_v[0] == pytest.approx(-0.200)

    def test_parse_a_pbs_file(self) -> None:
        path = DATA_DIR / "cv_sp1_hb2_5mvs_051826_0507.csv"
        exp = parse_cv_file(path)

        assert exp.file_meta.electrode_id == "hb2"
        assert exp.file_meta.electrolyte == Electrolyte.PBS
        assert exp.instrument_meta.high_e_v == pytest.approx(0.7)
        assert exp.instrument_meta.low_e_v == pytest.approx(0.5)

        # pbs files have empty segment blocks. they should still come back as
        # SegmentResult objects with None values, not silently dropped.
        assert len(exp.segment_results) == 4
        for seg in exp.segment_results:
            assert seg.ep_v is None
            assert seg.ip_a is None

        assert exp.n_points > 100

    def test_scan_rate_consistency(self) -> None:
        # the scan rate in the filename should match the one in the header
        for fp in sorted(DATA_DIR.glob("*.csv")):
            exp = parse_cv_file(fp)
            expected_v_s = exp.file_meta.scan_rate_mv_s / 1000.0
            assert exp.instrument_meta.scan_rate_v_s == pytest.approx(
                expected_v_s, rel=1e-3
            ), f"scan rate mismatch in {fp.name}"

    def test_all_files_parse(self) -> None:
        # smoke test, every file in data/raw must parse cleanly
        files = sorted(DATA_DIR.glob("*.csv"))
        assert len(files) > 0, "no cv files found in data/raw"

        experiments = parse_directory(DATA_DIR, skip_on_error=False)
        assert len(experiments) == len(files)

        # every experiment should have finite, non empty data
        for exp in experiments:
            assert exp.n_points > 0
            assert np.isfinite(exp.potential_v).all()
            assert np.isfinite(exp.current_a).all()

    def test_electrolyte_split(self) -> None:
        # ab electrodes are ferro/ferri, hb electrodes are pbs
        experiments = parse_directory(DATA_DIR, skip_on_error=False)
        for exp in experiments:
            eid = exp.file_meta.electrode_id
            if eid.startswith("ab"):
                assert exp.file_meta.electrolyte == Electrolyte.FERRO_FERRI
            elif eid.startswith("hb"):
                assert exp.file_meta.electrolyte == Electrolyte.PBS


# -----------------------------------------------------------------------------
# agent cross electrode comparison
# -----------------------------------------------------------------------------


class TestAgentCompare:
    def test_compare_runs_on_empty(self) -> None:
        # empty input should not crash, just return an empty comparison
        from cv_agent import compare
        result = compare([])
        assert result.overview == "no electrodes to compare."
        assert result.patterns == []
        assert result.outliers == []

    def test_compare_finds_outliers(self) -> None:
        # one electrode far from the others should get flagged as an outlier
        from cv_agent import compare
        rows = [
            {"electrode_id": "a", "easa_mm2": 1.20, "rs_r2_anodic": 0.999, "rs_r2_cathodic": 0.999},
            {"electrode_id": "b", "easa_mm2": 1.22, "rs_r2_anodic": 0.999, "rs_r2_cathodic": 0.999},
            {"electrode_id": "c", "easa_mm2": 1.21, "rs_r2_anodic": 0.999, "rs_r2_cathodic": 0.999},
            {"electrode_id": "d", "easa_mm2": 0.50, "rs_r2_anodic": 0.999, "rs_r2_cathodic": 0.999},
        ]
        result = compare(rows, use_llm=False)  # force deterministic
        # d should land in the outliers list
        assert any("d" in o for o in result.outliers)

    def test_compare_runs_on_real_summary(self) -> None:
        # make sure compare works on the same shape of dict that run_analysis builds
        from cv_agent import compare
        rows = [
            {"electrode_id": "ab2", "easa_mm2": 1.27, "rs_r2_anodic": 0.9994,
             "rs_r2_cathodic": 0.9996},
            {"electrode_id": "ab4", "easa_mm2": 0.72, "rs_r2_anodic": 0.9965,
             "rs_r2_cathodic": 0.9850},
            {"electrode_id": "hb1", "cdl_uF": 0.84, "cdl_r2": 0.998},
            {"electrode_id": "hb2", "cdl_uF": 0.62, "cdl_r2": 0.99},
        ]
        result = compare(rows, use_llm=False)
        assert "9 electrodes" not in result.overview  # we sent 4 not 9
        assert "4 electrodes" in result.overview
        assert len(result.recommendations) > 0
