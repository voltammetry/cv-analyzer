"""plot a couple of parsed cvs so we can eyeball that the raw data came out right.

quick visual gut check. we plot one ferro/ferri file, which should show clear
redox peaks (the duck shape), and one pbs file, which should look roughly like
a rectangle since it's just charging current. if those two look right, the
parser is pulling the data table correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no gui needed, we just save to a file
import matplotlib.pyplot as plt

# package is in src, add it to the path before importing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cv_agent import parse_cv_file

ROOT = Path(__file__).resolve().parent.parent

# one ferro/ferri file and one pbs file to compare side by side
files = [
    ROOT / "data" / "raw" / "cv_sp1_ab2_100mvs_051226_-0206_ff.csv",
    ROOT / "data" / "raw" / "cv_sp1_hb2_100mvs_051826_0507.csv",
]

# two plots side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, fp in zip(axes, files):
    exp = parse_cv_file(fp)
    # current in microamps just so the numbers are easier to read
    ax.plot(exp.potential_v, exp.current_a * 1e6, lw=0.9)
    ax.set_xlabel("Potential / V")
    ax.set_ylabel("Current / µA")
    ax.set_title(
        f"{exp.file_meta.electrode_id}  "
        f"{exp.file_meta.scan_rate_mv_s:g} mV/s  "
        f"({exp.file_meta.electrolyte.value})"
    )
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.grid(alpha=0.25)

fig.tight_layout()
out = ROOT / "scripts" / "parsed_cv_check.png"
fig.savefig(out, dpi=130)
print(f"saved the plot to {out}")
