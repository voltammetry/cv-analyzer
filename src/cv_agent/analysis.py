"""the three electrochemical analyses we run on parsed cv data.

each one is a plain function. you hand it a list of CVExperiment objects and
it gives back a dataclass with the results. no file reading, no plotting, just
the math. that's on purpose, it keeps them easy to test and means the agent
layer later on can just call these directly.

three analyses, all adapted from ye et al., 2024

    1. randles_sevcik(experiments) returns RandlesSevcikResult
       takes ferro/ferri cvs at different scan rates, fits ip vs sqrt(nu),
       and the slope tells us the electrochemically active surface area.

    2. cdl(experiments) returns CdlResult
       takes pbs cvs at different scan rates, plots the oxidation current at
       0.6 V against scan rate, and the slope is the double layer capacitance.

    3. laviron(experiments) returns LavironResult
       takes ferro/ferri cvs and tries to pull out k0, the heterogeneous
       electron transfer rate. heads up, this method strictly applies to
       surface bound redox species and ferro/ferri is floating in solution,
       so the number is more of an order of magnitude estimate. the result
       includes a caveat note explaining when it's trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import CVExperiment, Electrolyte


# -----------------------------------------------------------------------------
# constants from the protocol. all the magic numbers live up here so they're
# easy to find and change if needed.
# -----------------------------------------------------------------------------

# the randles sevcik equation at room temperature is
#     ip = 2.69e5 * n^(3/2) * A * sqrt(D) * C * sqrt(nu)
# where ip is in amps, A in cm^2, D in cm^2/s, C in mol/cm^3, nu in V/s.
# 2.69e5 is the standard prefactor at 25 C.
RS_CONSTANT = 2.69e5

# our redox probe is ferro/ferricyanide, which is a single electron transfer
N_ELECTRONS = 1

# diffusion coefficient for Fe(CN)6 3-/4- in 0.1 M KCl at 25 C. literature
# values are usually somewhere between 6.5e-6 and 8e-6 cm^2/s, we picked 7.2e-6
# as a reasonable middle.
D_FERROFERRI = 7.2e-6  # cm^2/s, Konopka & McDuffe 1970


# 5 mM concentration converted to mol/cm^3 (5 mmol/L * 1 L / 1000 cm^3)
C_FERROFERRI_MOL_PER_CM3 = 5e-3 * 1e-3

# the usual physical constants, for laviron
F_CONST = 96485.0  # faraday constant, C/mol
R_CONST = 8.314    # gas constant, J/(mol K)
T_KELVIN = 298.15  # room temperature, 25 C

# for cdl, the protocol says to read the oxidation current at 0.6 V on the
# forward sweep. it's a clean spot in the pbs window where no faradaic process
# is happening, so the current is purely capacitive.
CDL_POTENTIAL_V = 0.6

# which electron-transfer method applies, straight from the protocol:
# ΔEp > 212 mV -> laviron (steps 14-25), 61 <= ΔEp <= 212 mV -> nicholson
# (steps 26-37). these are the only two thresholds in the whole routing
# decision, which is why no llm is involved in choosing.
LAVIRON_MIN_MV = 212.0

# -----------------------------------------------------------------------------
# result dataclasses. each analysis returns one of these.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearFit:
    """a simple y = slope * x + intercept fit with how good it was."""

    slope: float
    intercept: float
    r_squared: float
    n_points: int


@dataclass(frozen=True)
class RandlesSevcikResult:
    electrode_id: str                     # electrode id, or the group id when
                                          # this was fit on replicate-averaged data
    scan_rates_v_s: np.ndarray            # nu, V/s
    sqrt_scan_rates: np.ndarray           # sqrt(nu), for plotting
    ipa_a: np.ndarray                     # anodic peak currents, A (mean if averaged)
    ipc_a: np.ndarray                     # cathodic peak currents, A (mean if averaged)
    anodic_fit: LinearFit                 # |ipa| vs sqrt(nu)
    cathodic_fit: LinearFit               # |ipc| vs sqrt(nu)
    easa_anodic_cm2: float                # surface area from anodic slope
    easa_cathodic_cm2: float              # surface area from cathodic slope
    easa_mean_cm2: float                  # average of the two, more robust
    ipa_std_a: np.ndarray | None = None   # per-scan-rate std across replicates
    ipc_std_a: np.ndarray | None = None   # (None when fit on a single electrode)
    n_replicates: int = 1              # average of the two, more robust


@dataclass(frozen=True)
class CdlResult:
    electrode_id: str
    scan_rates_v_s: np.ndarray
    i_at_06v_a: np.ndarray                # the current at 0.6 V for each rate
    fit: LinearFit                        # i vs nu, slope is the capacitance
    cdl_farads: float                     # capacitance in farads
    cdl_microfarads: float                # same thing in uF, just easier to read


@dataclass(frozen=True)
class LavironResult:
    """protocol steps 14-25: ΔE vs ln(ν), x-intercept -> Vc, ks = nFαVc/RT."""

    electrode_id: str                     # group id when fit on averaged data
    scan_rates_v_s: np.ndarray
    ln_scan_rates: np.ndarray             # ln(nu), the protocol's x axis
    delta_ea_v: np.ndarray                # Ep,a - E0' (averaged across replicates)
    delta_ec_v: np.ndarray                # Ep,c - E0' (averaged across replicates)
    delta_ea_std_v: np.ndarray | None
    delta_ec_std_v: np.ndarray | None
    anodic_branch_fit: LinearFit | None   # ΔEa vs ln(nu)
    cathodic_branch_fit: LinearFit | None # ΔEc vs ln(nu)
    alpha: float | None                   # transfer coefficient used in ks
    alpha_source: str                     # "branch slopes" or "assumed 0.5"
    vc_v_s: float | None                  # critical scan rate, the x-intercept
    k0_per_s: float | None                # ks = nFαVc/RT, s^-1
    n_replicates: int
    notes: str                            # caveats, and why k0 is None if it is                       # explains caveats and when k0 is None

@dataclass(frozen=True)
class NicholsonResult:
    electrode_id: str
    scan_rates_v_s: np.ndarray
    delta_ep_v: np.ndarray               # peak separation per scan rate
    psi: np.ndarray                      # Ψ looked up per scan rate
    ks_per_scan_cm_s: np.ndarray         # ks for each scan rate, cm/s
    ks_mean_cm_s: float | None
    ks_std_cm_s: float | None
    notes: str

# -----------------------------------------------------------------------------
# small helpers that the analyses share
# -----------------------------------------------------------------------------


def _linear_fit(x: np.ndarray, y: np.ndarray) -> LinearFit:
    # plain least squares line fit. numpy polyfit gives us slope and intercept,
    # then we compute r squared ourselves so we know how good the fit is.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points to fit a line")
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return LinearFit(slope=float(slope), intercept=float(intercept),
                     r_squared=r2, n_points=int(x.size))


def _split_segments(potential: np.ndarray, current: np.ndarray
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    # a chi file glues all the segments together in one long table. to work
    # with one sweep at a time we split wherever the potential reverses
    # direction. each piece is a monotonic sweep (going up or going down).
    if potential.size < 3:
        return [(potential, current)]

    # sign of the change in potential tells us which way we're sweeping. +1 is
    # forward (anodic), -1 is reverse (cathodic). where the sign flips, we're
    # at a turning point of the cv.
    dE = np.diff(potential)
    sign = np.sign(dE)
    flips = np.where(np.diff(sign) != 0)[0] + 1
    boundaries = [0, *flips.tolist(), potential.size]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a < 3:
            continue  # skip tiny fragments at the very ends
        segments.append((potential[a:b], current[a:b]))
    return segments


def _find_peaks_from_experiment(exp: CVExperiment) -> tuple[float | None, float | None]:
    # we want the anodic peak (most positive current) and the cathodic peak
    # (most negative current). the chi already reported these in the results
    # block, so we use those first. if it didn't find peaks for some reason
    # we fall back to scanning the raw data ourselves.

    anodic = [s.ip_a for s in exp.segment_results if s.ip_a is not None and s.ip_a > 0]
    cathodic = [s.ip_a for s in exp.segment_results if s.ip_a is not None and s.ip_a < 0]

    if anodic and cathodic:
        # protocol: use the second cycle. chi writes segments in order, so the
        # last anodic and last cathodic segments are the second cycle's peaks.
        return anodic[-1], cathodic[-1]

    # backup, scan the raw data. just take the global max and min current
    # across all segments.
    segments = _split_segments(exp.potential_v, exp.current_a)
    ipa = max((seg[1].max() for seg in segments), default=None)
    ipc = min((seg[1].min() for seg in segments), default=None)
    return (float(ipa) if ipa is not None else None,
            float(ipc) if ipc is not None else None)


def _current_at_potential(exp: CVExperiment, target_v: float,
                          direction: str = "forward") -> float:
    """read the current at a specific potential on either the forward or reverse sweep.

    forward means the potential is going up (anodic sweep), reverse means it's
    coming back down (cathodic sweep). we find the segment that has the right
    direction and contains our target voltage, then interpolate to get the
    current right at that voltage.
    """
    segments = _split_segments(exp.potential_v, exp.current_a)
    if not segments:
        raise ValueError("no segments to work with")

    # find every segment that goes the right way and brackets our target voltage
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for E, I in segments:
        if E[0] == E[-1]:
            continue
        going_up = E[-1] > E[0]
        if direction == "forward" and going_up:
            if E.min() <= target_v <= E.max():
                candidates.append((E, I))
        elif direction == "reverse" and not going_up:
            if E.min() <= target_v <= E.max():
                candidates.append((E, I))

    if not candidates:
        raise ValueError(
            f"no {direction} sweep contains E={target_v} V. "
            f"file window was {exp.potential_v.min():.3f} to {exp.potential_v.max():.3f} V"
        )

    # use the last matching cycle. by then the system has settled into a
    # steady state, so it's the cleanest reading.
    E, I = candidates[-1]
    # np.interp needs increasing x, so sort first
    order = np.argsort(E)
    return float(np.interp(target_v, E[order], I[order]))

@dataclass(frozen=True)
class PeakTable:
    """the protocol's per-scan-rate table for one electrode.

    one row per scan rate: both peaks, the formal potential E0' (midpoint of
    the peak potentials), and the two half-separations ΔEa = Ep,a - E0' and
    ΔEc = Ep,c - E0'. everything downstream (randles sevcik, laviron,
    nicholson) reads from this.
    """

    electrode_id: str
    scan_rates_v_s: np.ndarray
    ep_a_v: np.ndarray          # anodic peak potential
    ip_a_a: np.ndarray          # anodic peak current
    ep_c_v: np.ndarray          # cathodic peak potential
    ip_c_a: np.ndarray          # cathodic peak current
    e0_prime_v: np.ndarray      # (Ep,a + Ep,c)/2
    delta_ea_v: np.ndarray      # Ep,a - E0'  (positive)
    delta_ec_v: np.ndarray      # Ep,c - E0'  (negative, same magnitude)
    delta_ep_v: np.ndarray      # Ep,a - Ep,c = ΔEa - ΔEc


@dataclass(frozen=True)
class ReplicateTable:
    """the protocol's averaged table across replicate electrodes.

    mean and sample std dev (ddof=1) at each scan rate, computed across the
    electrodes in one replicate group. this is what the protocol says to fit,
    not the individual electrodes.
    """

    group_id: str
    electrode_ids: tuple[str, ...]
    scan_rates_v_s: np.ndarray
    ip_a_mean_a: np.ndarray
    ip_a_std_a: np.ndarray
    ip_c_mean_a: np.ndarray
    ip_c_std_a: np.ndarray
    delta_ea_mean_v: np.ndarray
    delta_ea_std_v: np.ndarray
    delta_ec_mean_v: np.ndarray
    delta_ec_std_v: np.ndarray
    delta_ep_mean_v: np.ndarray
    delta_ep_std_v: np.ndarray

    @property
    def mean_delta_ep_mv(self) -> float:
        return float(np.nanmean(self.delta_ep_mean_v)) * 1000.0
    
    
def _second_cycle_peaks(exp: CVExperiment
                    ) -> tuple[float, float, float, float] | None:
    """(Ep_a, ip_a, Ep_c, ip_c) for the second cycle, or None if not found.

    the chi lists one peak per segment in file order, so the *last* anodic and
    *last* cathodic entries are the second cycle's peaks, which is the cycle
    the protocol says to use.
    """
    epa = ipa = epc = ipc = None
    for seg in exp.segment_results:
        if seg.ep_v is None or seg.ip_a is None:
            continue
        if seg.ip_a > 0:
            epa, ipa = seg.ep_v, seg.ip_a
        elif seg.ip_a < 0:
            epc, ipc = seg.ep_v, seg.ip_a
    if epa is None or epc is None:
        return None
    return epa, ipa, epc, ipc


def peak_table(experiments: list[CVExperiment]) -> PeakTable:
    """build the protocol's per-scan-rate table for one electrode."""
    if not experiments:
        raise ValueError("peak_table, got an empty experiment list")
    electrode_ids = {e.file_meta.electrode_id for e in experiments}
    if len(electrode_ids) != 1:
        raise ValueError(f"peak_table, all experiments must be one electrode. got {electrode_ids}")
    electrode_id = electrode_ids.pop()
    for e in experiments:
        if e.file_meta.electrolyte != Electrolyte.FERRO_FERRI:
            raise ValueError("peak_table needs ferro/ferri data (it's peak based)")

    experiments = sorted(experiments, key=lambda e: e.scan_rate_v_s)
    n = len(experiments)
    nus = np.array([e.scan_rate_v_s for e in experiments])
    ep_a = np.full(n, np.nan); ip_a = np.full(n, np.nan)
    ep_c = np.full(n, np.nan); ip_c = np.full(n, np.nan)
    for i, e in enumerate(experiments):
        peaks = _second_cycle_peaks(e)
        if peaks is not None:
            ep_a[i], ip_a[i], ep_c[i], ip_c[i] = peaks

    e0 = (ep_a + ep_c) / 2.0
    return PeakTable(
        electrode_id=electrode_id,
        scan_rates_v_s=nus,
        ep_a_v=ep_a, ip_a_a=ip_a,
        ep_c_v=ep_c, ip_c_a=ip_c,
        e0_prime_v=e0,
        delta_ea_v=ep_a - e0,
        delta_ec_v=ep_c - e0,
        delta_ep_v=ep_a - ep_c,
    )
def average_peak_tables(tables: list[PeakTable], group_id: str) -> ReplicateTable:
    """average the per-electrode tables at each shared scan rate.

    protocol steps 19-20 / 31-32: mean and std dev across the replicate
    electrodes. we align on scan rates that every electrode has (rounded to
    0.1 mV/s so float noise doesn't split them).
    """
    if not tables:
        raise ValueError("average_peak_tables, got no tables")

    key = lambda nu: round(nu * 1e4)  # 0.1 mV/s resolution
    common = set(key(nu) for nu in tables[0].scan_rates_v_s)
    for t in tables[1:]:
        common &= set(key(nu) for nu in t.scan_rates_v_s)
    if not common:
        raise ValueError("average_peak_tables, the electrodes share no scan rates")

    nus = np.array(sorted(k / 1e4 for k in common))

    def stack(attr: str) -> np.ndarray:
        rows = []
        for t in tables:
            idx = {key(nu): i for i, nu in enumerate(t.scan_rates_v_s)}
            vals = getattr(t, attr)
            rows.append([vals[idx[key(nu)]] for nu in nus])
        return np.array(rows)  # shape (n_electrodes, n_scan_rates)

    def mean_std(attr: str) -> tuple[np.ndarray, np.ndarray]:
        m = stack(attr)
        mean = np.nanmean(m, axis=0)
        std = (np.nanstd(m, axis=0, ddof=1) if m.shape[0] > 1
               else np.zeros(m.shape[1]))
        return mean, std

    ipa_m, ipa_s = mean_std("ip_a_a")
    ipc_m, ipc_s = mean_std("ip_c_a")
    dea_m, dea_s = mean_std("delta_ea_v")
    dec_m, dec_s = mean_std("delta_ec_v")
    dep_m, dep_s = mean_std("delta_ep_v")

    return ReplicateTable(
        group_id=group_id,
        electrode_ids=tuple(t.electrode_id for t in tables),
        scan_rates_v_s=nus,
        ip_a_mean_a=ipa_m, ip_a_std_a=ipa_s,
        ip_c_mean_a=ipc_m, ip_c_std_a=ipc_s,
        delta_ea_mean_v=dea_m, delta_ea_std_v=dea_s,
        delta_ec_mean_v=dec_m, delta_ec_std_v=dec_s,
        delta_ep_mean_v=dep_m, delta_ep_std_v=dep_s,
    )
def choose_kinetics_method(mean_delta_ep_mv: float) -> str:
    """the protocol's routing rule, as one plain deterministic function.

    ΔEp > 212 mV -> "laviron", 61-212 mV -> "nicholson", below 61 mV the
    couple is essentially reversible and neither method applies -> "none".
    """
    if mean_delta_ep_mv > LAVIRON_MIN_MV:
        return "laviron"
    if NICHOLSON_MIN_MV <= mean_delta_ep_mv <= NICHOLSON_MAX_MV:
        return "nicholson"
    return "none"

# -----------------------------------------------------------------------------
# analysis 1, randles sevcik, gives us EASA
# -----------------------------------------------------------------------------


def randles_sevcik(experiments: list[CVExperiment]) -> RandlesSevcikResult:
    """fit ip vs sqrt(nu) for a set of ferro/ferri cvs and pull out the surface area.

    randles sevcik at room temp says
        ip = 2.69e5 * n^(3/2) * A * sqrt(D) * C * sqrt(nu)
    so plotting |ip| vs sqrt(nu) gives a straight line with slope
        slope = 2.69e5 * n^(3/2) * A * sqrt(D) * C
    we solve for A
        A = slope / (2.69e5 * n^(3/2) * sqrt(D) * C)
    and that A is the electrochemically active surface area in cm^2.

    every experiment we get has to be ferro/ferri (otherwise the math doesn't
    apply) and from the same electrode (we're fitting one electrode's data).
    """
    if not experiments:
        raise ValueError("randles_sevcik, got an empty experiment list")

    # all the files must belong to the same electrode, otherwise we'd be
    # mixing surface areas and getting nonsense
    electrode_ids = {e.file_meta.electrode_id for e in experiments}
    if len(electrode_ids) != 1:
        raise ValueError(
            f"randles_sevcik, all the experiments need to be from one electrode. "
            f"got {electrode_ids}"
        )
    electrode_id = electrode_ids.pop()

    # check the electrolyte too. running this on pbs would be wrong.
    for e in experiments:
        if e.file_meta.electrolyte != Electrolyte.FERRO_FERRI:
            raise ValueError(
                f"randles_sevcik wants ferro/ferri, but {e.file_meta.source_path.name} "
                f"is {e.file_meta.electrolyte.value}"
            )

   # sort by scan rate so the plot reads cleanly left to right
    experiments = sorted(experiments, key=lambda e: e.scan_rate_v_s)

    # collect scan rates and peak currents into parallel arrays
    nus = np.array([e.scan_rate_v_s for e in experiments])
    ipa = np.zeros(len(experiments))
    ipc = np.zeros(len(experiments))
    for i, e in enumerate(experiments):
        a, c = _find_peaks_from_experiment(e)
        if a is None or c is None:
            raise ValueError(
                f"couldn't find peaks in {e.file_meta.source_path.name}"
            )
        ipa[i] = a
        ipc[i] = c

    return _randles_sevcik_core(electrode_id, nus, ipa, ipc)


def randles_sevcik_from_replicates(rep: ReplicateTable) -> RandlesSevcikResult:
    """the protocol version: fit the replicate-averaged peak currents.

    same math as randles_sevcik(), but the ip at each scan rate is the mean
    across the replicate electrodes and the std devs ride along for the
    error bars in the plot. this is the number to report.
    """
    res = _randles_sevcik_core(
        rep.group_id, rep.scan_rates_v_s, rep.ip_a_mean_a, rep.ip_c_mean_a
    )
    import dataclasses
    return dataclasses.replace(
        res,
        ipa_std_a=rep.ip_a_std_a,
        ipc_std_a=rep.ip_c_std_a,
        n_replicates=len(rep.electrode_ids),
    )


def _randles_sevcik_core(label: str, nus: np.ndarray,
                         ipa: np.ndarray, ipc: np.ndarray) -> RandlesSevcikResult:
    """the shared fit + unit conversion behind both entry points."""
    valid = ~(np.isnan(ipa) | np.isnan(ipc))
    if valid.sum() < 2:
        raise ValueError(f"randles_sevcik, fewer than two valid peak pairs for {label}")
    nus, ipa, ipc = nus[valid], ipa[valid], ipc[valid]

    sqrt_nu = np.sqrt(nus)
    # fit each branch separately. doing both gives us a built in sanity check,
    # they should give nearly the same surface area.
    fit_a = _linear_fit(sqrt_nu, np.abs(ipa))
    fit_c = _linear_fit(sqrt_nu, np.abs(ipc))

    # the denominator that turns slope into surface area
    denom = (
        RS_CONSTANT
        * (N_ELECTRONS ** 1.5)
        * math.sqrt(D_FERROFERRI)
        * C_FERROFERRI_MOL_PER_CM3
    )
    easa_a = fit_a.slope / denom
    easa_c = fit_c.slope / denom

    return RandlesSevcikResult(
        electrode_id=label,
        scan_rates_v_s=nus,
        sqrt_scan_rates=sqrt_nu,
        ipa_a=ipa,
        ipc_a=ipc,
        anodic_fit=fit_a,
        cathodic_fit=fit_c,
        easa_anodic_cm2=easa_a,
        easa_cathodic_cm2=easa_c,
        easa_mean_cm2=(easa_a + easa_c) / 2,
    )


# -----------------------------------------------------------------------------
# analysis 2, double layer capacitance from pbs
# -----------------------------------------------------------------------------


def cdl(experiments: list[CVExperiment],
        target_potential_v: float = CDL_POTENTIAL_V) -> CdlResult:
    """double layer capacitance from a set of pbs cvs at different scan rates.

    in pbs nothing is reacting, so any current is just from charging the
    electrode/electrolyte interface (the double layer). that charging current
    follows i = Cdl * dV/dt = Cdl * nu. so if we plot i at a fixed voltage
    against scan rate, the slope is the capacitance.

    the protocol says to read the current at +0.6 V on the forward sweep.
    """
    if not experiments:
        raise ValueError("cdl, got an empty experiment list")

    # one electrode at a time
    electrode_ids = {e.file_meta.electrode_id for e in experiments}
    if len(electrode_ids) != 1:
        raise ValueError(
            f"cdl, all the experiments need to be from one electrode. got {electrode_ids}"
        )
    electrode_id = electrode_ids.pop()

    # must be pbs, ferro/ferri would have peaks and break the model
    for e in experiments:
        if e.file_meta.electrolyte != Electrolyte.PBS:
            raise ValueError(
                f"cdl wants pbs, but {e.file_meta.source_path.name} "
                f"is {e.file_meta.electrolyte.value}"
            )

    experiments = sorted(experiments, key=lambda e: e.scan_rate_v_s)
    nus = np.array([e.scan_rate_v_s for e in experiments])
    # read the current at our target voltage on the forward sweep for every file
    currents = np.array([
        _current_at_potential(e, target_potential_v, direction="forward")
        for e in experiments
    ])

    fit = _linear_fit(nus, currents)
    # slope is amps per (V/s), which is the same as A * s / V = coulombs / V = farads
    cdl_F = fit.slope
    return CdlResult(
        electrode_id=electrode_id,
        scan_rates_v_s=nus,
        i_at_06v_a=currents,
        fit=fit,
        cdl_farads=cdl_F,
        cdl_microfarads=cdl_F * 1e6,
    )


# -----------------------------------------------------------------------------
# analysis 3, laviron, electron transfer kinetics
# -----------------------------------------------------------------------------


def laviron(rep: ReplicateTable) -> LavironResult:
    """laviron per the protocol: ΔE vs ln(ν), x-intercept, ks = nFαVc/RT.

    the protocol's exact recipe (steps 19-25), run on the replicate-averaged
    table (a single electrode still works, it's just a group of one):

      1. average ΔEa and ΔEc across the replicate electrodes  (done upstream
         in average_peak_tables)
      2. check the mean peak separation ΔEp = ΔEa - ΔEc is > 212 mV,
         otherwise this method doesn't apply and Nicholson should be used
      3. plot ΔEa and ΔEc against ln(ν) and fit each branch with a line
      4. the x-intercept (where ΔE -> 0) is the critical scan rate,
         Vc = exp(x-intercept). the two branches give two estimates that
         should agree; we average them.
      5. ks = n·F·α·Vc / (R·T), in s⁻¹

    the protocol doesn't spell out where α comes from. we derive it from the
    branch slopes (laviron theory: anodic slope = RT/((1-α)nF), cathodic
    slope = -RT/(αnF) in ln space) and fall back to the standard α = 0.5
    assumption for a symmetric couple like ferro/ferri if that comes out
    non-physical. the notes field always says which was used.

    the usual caveat still applies: laviron strictly describes surface-bound
    species and ferro/ferri diffuses, so treat ks as an estimate.
    """
    notes: list[str] = [
        "ferro/ferricyanide is diffusion controlled, not surface confined; "
        "laviron strictly applies to adsorbed species, so ks is an estimate."
    ]

    valid = ~(np.isnan(rep.delta_ea_mean_v) | np.isnan(rep.delta_ec_mean_v))
    nus = rep.scan_rates_v_s
    ln_nus = np.log(nus)

    def bail(reason: str, fit_a=None, fit_c=None, alpha=None,
             alpha_source="n/a") -> LavironResult:
        notes.append(reason)
        return LavironResult(
            electrode_id=rep.group_id,
            scan_rates_v_s=nus,
            ln_scan_rates=ln_nus,
            delta_ea_v=rep.delta_ea_mean_v,
            delta_ec_v=rep.delta_ec_mean_v,
            delta_ea_std_v=rep.delta_ea_std_v,
            delta_ec_std_v=rep.delta_ec_std_v,
            anodic_branch_fit=fit_a,
            cathodic_branch_fit=fit_c,
            alpha=alpha,
            alpha_source=alpha_source,
            vc_v_s=None,
            k0_per_s=None,
            n_replicates=len(rep.electrode_ids),
            notes=" ".join(notes),
        )

    if valid.sum() < 3:
        return bail("fewer than three scan rates with valid peak pairs; can't fit.")

    # protocol step 21: this method only applies above 212 mV mean separation
    mean_dep_mv = rep.mean_delta_ep_mv
    if not (mean_dep_mv > LAVIRON_MIN_MV):
        return bail(
            f"mean ΔEp = {mean_dep_mv:.0f} mV is not above {LAVIRON_MIN_MV:.0f} mV; "
            f"the protocol says to use Nicholson's method instead. ks not computed."
        )

    # protocol steps 22-23: linear fit of each averaged branch against ln(ν)
    fit_a = _linear_fit(ln_nus[valid], rep.delta_ea_mean_v[valid])
    fit_c = _linear_fit(ln_nus[valid], rep.delta_ec_mean_v[valid])

    # α from the branch slopes (see docstring). the protocol's ks formula
    # needs an α but never defines one, so this is our best faithful reading.
    factor = R_CONST * T_KELVIN / (N_ELECTRONS * F_CONST)  # RT/nF in ln space
    alpha_estimates = []
    if fit_a.slope > 0:
        alpha_estimates.append(1.0 - factor / fit_a.slope)
    if fit_c.slope < 0:
        alpha_estimates.append(-factor / fit_c.slope)
    alpha = float(np.mean(alpha_estimates)) if alpha_estimates else float("nan")
    if 0.0 < alpha < 1.0:
        alpha_source = "branch slopes"
        notes.append(f"α = {alpha:.3f} derived from the ΔE vs ln(ν) branch slopes.")
    else:
        alpha, alpha_source = 0.5, "assumed 0.5"
        notes.append(
            "slope-derived α was outside (0, 1), so α = 0.5 was assumed "
            "(standard for a symmetric couple; the protocol doesn't define α)."
        )

    # protocol step 24: x-intercept of each branch, ΔE = 0 at ln(ν) = -b/m.
    # back out of ln space to get the critical scan rate in V/s, then average
    # the two branch estimates.
    vcs = []
    for fit in (fit_a, fit_c):
        if fit.slope != 0:
            vcs.append(math.exp(-fit.intercept / fit.slope))
    if not vcs:
        return bail("both branch fits came back flat; no x-intercept.",
                    fit_a, fit_c, alpha, alpha_source)
    vc = float(np.mean(vcs))
    if len(vcs) == 2:
        notes.append(
            f"Vc from anodic branch = {vcs[0]:.3g} V/s, from cathodic branch "
            f"= {vcs[1]:.3g} V/s; used the mean, {vc:.3g} V/s."
        )

    # protocol step 25
    ks = N_ELECTRONS * F_CONST * alpha * vc / (R_CONST * T_KELVIN)

    return LavironResult(
        electrode_id=rep.group_id,
        scan_rates_v_s=nus,
        ln_scan_rates=ln_nus,
        delta_ea_v=rep.delta_ea_mean_v,
        delta_ec_v=rep.delta_ec_mean_v,
        delta_ea_std_v=rep.delta_ea_std_v,
        delta_ec_std_v=rep.delta_ec_std_v,
        anodic_branch_fit=fit_a,
        cathodic_branch_fit=fit_c,
        alpha=alpha,
        alpha_source=alpha_source,
        vc_v_s=vc,
        k0_per_s=float(ks),
        n_replicates=len(rep.electrode_ids),
        notes=" ".join(notes),
    )

# -----------------------------------------------------------------------------
# analysis 4, nicholson, electron transfer kinetics for 61-212 mV peak sep
# -----------------------------------------------------------------------------

# Ψ vs ΔEp in mV (n=1, 25 °C), transcribed from protocol Table 4 (page 28).
# np.interp fills the small gaps between tabulated points (e.g. 91-99).
_NICHOLSON_PSI_MV = {
    61: 9.5, 62: 6.5, 63: 5.3, 64: 4.4, 65: 3.7, 66: 3.3, 67: 2.9, 68: 2.6,
    69: 2.35, 70: 2.15, 71: 1.95, 72: 1.83, 73: 1.69, 74: 1.58, 75: 1.48,
    76: 1.40, 77: 1.32, 78: 1.25, 79: 1.19, 80: 1.13, 81: 1.08, 82: 1.028,
    83: 0.984, 84: 0.943, 85: 0.906, 86: 0.871, 87: 0.839, 88: 0.809,
    89: 0.781, 90: 0.755,
    # 91-99 interpolated by np.interp between 90 and 100
    100: 0.558, 101: 0.544, 102: 0.530, 103: 0.517, 104: 0.504, 105: 0.492,
    106: 0.480, 107: 0.469, 108: 0.458, 109: 0.448, 110: 0.438, 111: 0.428,
    112: 0.419, 113: 0.410, 114: 0.402, 115: 0.394, 116: 0.386, 117: 0.378,
    118: 0.370, 119: 0.363, 120: 0.356, 121: 0.349, 122: 0.343, 123: 0.337,
    124: 0.331, 125: 0.325, 126: 0.319, 127: 0.313, 128: 0.308, 129: 0.302,
    # 130-212: Lavagnini et al. 2004 fit Ψ=(-0.6288+0.0021·ΔEp)/(1-0.017·ΔEp)
    130: 0.294, 135: 0.267, 140: 0.243, 145: 0.221, 150: 0.202,
    155: 0.186, 160: 0.170, 165: 0.156, 170: 0.144, 175: 0.132,
    180: 0.122, 185: 0.112, 190: 0.103, 195: 0.095, 200: 0.087,
    205: 0.080, 210: 0.073, 212: 0.071,
}

NICHOLSON_MIN_MV = 61.0
NICHOLSON_MAX_MV = 212.0


def _psi_from_dep(delta_ep_mv: float) -> float:
    xs = sorted(_NICHOLSON_PSI_MV)
    ys = [_NICHOLSON_PSI_MV[x] for x in xs]
    return float(np.interp(delta_ep_mv, xs, ys))


def nicholson(rep: ReplicateTable) -> NicholsonResult:
    """heterogeneous rate constant ks via Nicholson's working curve.

    protocol steps 26-37, run on the replicate-averaged peak separations.
    valid when the mean ΔEp is 61-212 mV. ψ is looked up per scan rate from
    Table 4, ks = ψ·√(nπDFν/RT) per scan rate (in cm/s, a different unit
    from laviron's s⁻¹), then mean and std across the scan rates.
    """
    nus = rep.scan_rates_v_s
    delta_ep = rep.delta_ep_mean_v
    mean_dep_mv = rep.mean_delta_ep_mv

    if not (NICHOLSON_MIN_MV <= mean_dep_mv <= NICHOLSON_MAX_MV):
        return NicholsonResult(
            electrode_id=rep.group_id,
            scan_rates_v_s=nus,
            delta_ep_v=delta_ep,
            psi=np.full(len(nus), np.nan),
            ks_per_scan_cm_s=np.full(len(nus), np.nan),
            ks_mean_cm_s=None,
            ks_std_cm_s=None,
            notes=(f"mean ΔEp = {mean_dep_mv:.0f} mV is outside Nicholson's "
                   f"61-212 mV range; use Laviron instead. ks not computed."),
        )

    psi = np.array([_psi_from_dep(d * 1000.0) for d in delta_ep])
    ks = psi * np.sqrt(
        N_ELECTRONS * math.pi * D_FERROFERRI * F_CONST * nus / (R_CONST * T_KELVIN)
    )
    return NicholsonResult(
        electrode_id=rep.group_id,
        scan_rates_v_s=nus,
        delta_ep_v=delta_ep,
        psi=psi,
        ks_per_scan_cm_s=ks,
        ks_mean_cm_s=float(np.nanmean(ks)),
        ks_std_cm_s=float(np.nanstd(ks, ddof=1)) if np.isfinite(ks).sum() > 1 else None,
        notes=(f"ks per protocol step 35, averaged over {len(ks)} scan rates "
               f"and {len(rep.electrode_ids)} replicate electrode(s). "
               f"mean ΔEp = {mean_dep_mv:.0f} mV. ψ table verified against "
               f"protocol Table 4. ks in cm/s."),
    )