# cv-agent

An LLM agent that automatically turns raw potentiostat csv's into detailed reports.

Takes a folder of CHI Instruments csv files, parses them, decides which analyses to run given the test (Randles-Ševčík, C_dl, Laviron), does the math, writes a clean report of the results, and flags unusual results. 

Built for wearable biosensor electrode cyclic voltammetry characterization following the Ye et al. (2024) protocol.


## Motivation

Characterizing a fabricated electrode means running the same analysis pipeline across many cyclic voltammetry csv files, involving multiple scan rates for multiple electrodes and multiple electrolytes. Previously, this was done by hand in Excel, which was tedious and liable to human error. This tool automates the pipeline and uses an LLM to decide which analysis is appropriate for which data. Randles-Ševčík is used for diffusion-controlled redox (ferro/ferricyanide), C_dl is used for capacitive charging current (PBS/sweat), and Laviron is used for surface-confined species.




## Installation

```bash
git clone https://github.com/<you>/cv-agent.git
cd cv-agent
python -m venv .venv
source .venv/bin/activate      # windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,analysis,agent]"
```

drop `.csv` files in `data/raw/`.

## How to un

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/run_analysis.py
```

Reports go to `output/`. `output/index.html` opens in a browser.

If no API key, reports will function without LLM analysis.

---

## How it works

```
parse CSVs --> group by electrode --> agent decides analyses --> run math --> agent interprets --> HTML report
                                                                                     |
                                                                                     V
                                                                            agent compares batch
```

Three analyses as functions:

- **`randles_sevcik()`** — fits |ip| vs √ν, extracts EASA via the Randles-Ševčík equation. roughness factor = EASA / geometric area.
- **`cdl()`** — fits current at 0.6 V vs ν (linear in ν for capacitive charging), extracts C_dl.
- **`laviron()`** — fits ΔEp vs log(ν) for electron transfer kinetics. returns k⁰ with a caveat block when applied to non-absorbed species.

The agent layer (`agent.py`) has three entry points: `decide()` decides which analysis to use depending on the test, `interpret()` writes the electrode summary, and `compare()` does cross-batch outlier detection at 1σ.

---

## File naming convention

```
cv_sp<size>_<electrode_id>_<scan_rate>mvs_<MMDDYY>_-<batch_code>[_ff].csv
```
note: file name formatting is very important, since the code will sometimes break if it is not followed strictly.


`sp<size>`, ex:  `sp1`, electrode diameter in mm (sp1 = 1mm, sp2 = 2mm) 
`electrode_id`, ex: `ab2`, electrode identifier 
`scan_rate`, ex: `100mvs`, CV scan rate in mV/s 
`batch_code`, ex: `-0206`, electrolyte (-0206 = ferro/ferri in KCl, 0507 = PBS) 
`_ff`, optional, present on ferro/ferricyanide runs 




## License

MIT
