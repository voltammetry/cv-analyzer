# cv-agent

A tool that automatically turns raw potentiostat csv's into detailed reports.

Takes a folder of CHI Instruments csv files, parses them, runs the
electrochemical math the protocol calls for (Randles-Ševčík, C_dl, Laviron,
Nicholson), does replicate averaging, writes a clean report of the results,
and flags unusual results.

Built for wearable biosensor electrode cyclic voltammetry characterization
following the Ye et al. (2024) protocol.


## Motivation

Characterizing a fabricated electrode means running the same analysis
pipeline across many cyclic voltammetry csv files, involving multiple scan
rates for multiple electrodes and multiple electrolytes. Previously, this
was done by hand in Excel, which was tedious and liable to human error. This
tool automates the pipeline. Randles-Ševčík is used for diffusion-controlled
redox (ferro/ferricyanide), C_dl is used for capacitive charging current
(PBS/sweat), and Laviron or Nicholson is used for electron transfer
kinetics, chosen by a fixed threshold on peak separation (see below) — not
by a language model.


## Installation

```bash
git clone https://github.com/voltammetry/cv-agent.git
cd cv-agent
python -m venv .venv
source .venv/bin/activate      # <-- macos. windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,analysis,agent]"
```

drop `.csv` files in `data/raw/`.

## How to run

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # optional, see below
python scripts/run_analysis.py
```

Reports go to `output/`. `output/index.html` opens in a browser and links
to both the replicate-group reports (the protocol's actual output) and the
per-electrode QC reports.

`ANTHROPIC_API_KEY` is optional. Every number and every method choice in
this pipeline is computed deterministically whether or not it's set. If it
is set, Claude is used only to write the plain-language summary paragraphs
(per-electrode `interpret()` and the batch `compare()` overview) in nicer
prose over numbers that were already computed in code. Without a key, those
same summaries are still written, just from a simpler rule-based template.

---

## How it works

```
parse CSVs --> classify electrolyte from CHI header --> group by electrode (QC)
                                                             |
                                                             v
                                                  group by replicate set --> average peak table
                                                             |
                                                             v
                                        pick Laviron or Nicholson by mean deltaEp (fixed rule)
                                                             |
                                                             v
                                    run math --> interpret() summarizes --> HTML report
                                                             |
                                                             v
                                                    compare() finds batch outliers
```

Analyses, as functions in `analysis.py`:

- **`randles_sevcik()`** / **`randles_sevcik_from_replicates()`** — fits
  |ip| vs sqrt(nu), extracts EASA via the Randles-Sevcik equation.
  roughness factor = EASA / geometric area. the replicate version fits the
  mean peak currents across a group of electrodes and carries std devs
  through as error bars; this is the number the protocol asks you to
  report.
- **`cdl()`** — fits current at 0.6 V vs nu (linear in nu for capacitive
  charging), extracts C_dl.
- **`laviron()`** — fits deltaEa and deltaEc vs ln(nu), takes the
  x-intercept as the critical scan rate Vc, and computes
  ks = n*F*alpha*Vc / (R*T). applies when the mean peak separation exceeds
  212 mV. alpha is derived from the branch slopes, with a documented
  fallback to 0.5 when that comes out non-physical (the protocol never
  defines where alpha comes from).
- **`nicholson()`** — looks up psi from the peak separation against the
  protocol's Table 4 and computes ks per scan rate, then averages. applies
  for a mean peak separation between 61 and 212 mV.
- **`choose_kinetics_method()`** — the single, fixed rule that picks
  between the two above from the mean peak separation. this is the only
  "decision" anywhere in the pipeline, and it is not made by a model.

`agent.py` has three entry points, none of which choose a method or compute
a number:

- **`decide()`** — deterministically figures out which analyses apply to an
  electrode's data (randles sevcik + laviron/nicholson for ferro/ferri,
  cdl for pbs). always the same rule, same output for the same input.
- **`interpret()`** — writes a plain-language summary of one electrode's
  already-computed results.
- **`compare()`** — computes cross-batch stats (means, 1-sigma outliers,
  low-R^2 fits) in code, always. with an API key, Claude may rewrite the
  overview paragraph in nicer prose from those numbers; it never produces
  or edits a number or an outlier.

---

## How experiments are identified

Electrolyte (and therefore which analysis applies) is read from the CHI
file's own header — specifically the potential window it swept:

- **-0.2 V to +0.6 V** (a wide redox window) → ferro/ferricyanide
- **+0.5 V to +0.6/0.7 V** (a narrow window) → PBS / Cdl

This is the instrument's own record of what it ran, so it's the source of
truth. Scan rate is also always read from the header.

Electrode id, electrode size, and replicate group are the three things the
tool can't get from the header alone. By default they come from the
filename (see below), but you can set them explicitly and override the
filename guess with an optional `manifest.csv` dropped in `data/raw/`
alongside your CSVs:

```csv
filename,electrode_id,electrode_size_mm,replicate_group,electrolyte
cv_sp1_ab2_5mvs_051226_0206_ff.csv,ab2,1,batchA,ferro_ferri
```

Every column except `filename` is optional; only the ones you provide
override the filename/header guess. A file that isn't in the manifest and
doesn't match the filename convention still parses fine — its electrode id
just falls back to the filename's stem and its electrolyte comes from the
header.

### Replicate groups

The protocol's actual unit of analysis is a **set of replicate electrodes**,
not one electrode: run 3 (or more) electrodes, average their peak currents
and peak separations at each scan rate, and fit the averaged data. By
default, electrodes are grouped by stripping the trailing digits off the
electrode id — `ab2`, `ab3`, `ab4` all fall into group `"ab"`. Use
`replicate_group` in the manifest to set this explicitly if your naming
doesn't follow that pattern.

Per-electrode reports (`<electrode_id>_report.html`) are still written for
every electrode as a QC view — raw traces, the per-scan-rate peak table,
and an individual Randles-Sevcik/Cdl fit — but kinetics (Laviron/Nicholson)
and the EASA you should actually report run on the replicate-averaged data,
in `group_<group_id>_report.html`.

## File naming convention

```
cv_sp<size>_<electrode_id>_<scan_rate>mvs_<MMDDYY>_<batch_code>[_ff].csv
```

`sp<size>`, ex: `sp1`, electrode diameter in mm (sp1 = 1mm, sp2 = 2mm)
`electrode_id`, ex: `ab2`, electrode identifier
`scan_rate`, ex: `100mvs`, CV scan rate in mV/s
`batch_code`, ex: `0206`, a filename-level hint at electrolyte, used only
as a pre-fill; the CHI header's potential window always wins if the two
disagree
`_ff`, optional, another filename-level hint, same caveat as above

If a filename doesn't match this pattern, it no longer breaks anything —
the tool falls back to the header for electrolyte/scan rate and to the
filename's stem for the electrode id, and a manifest entry can fill in
anything still missing.

---

## License

MIT