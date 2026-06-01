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
# values are usually somewhere between 6.5e-6 and 8e-6 cm^2/s, we picked 7.6e-6
# as a reasonable middle.
D_FERROFERRI = 7.6e-6  # cm^2/s

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
    electrode_id: str
    scan_rates_v_s: np.ndarray            # nu, V/s
    sqrt_scan_rates: np.ndarray           # sqrt(nu), for plotting
    ipa_a: np.ndarray                     # anodic peak currents, A
    ipc_a: np.ndarray                     # cathodic peak currents, A
    anodic_fit: LinearFit                 # |ipa| vs sqrt(nu)
    cathodic_fit: LinearFit               # |ipc| vs sqrt(nu)
    easa_anodic_cm2: float                # surface area from anodic slope
    easa_cathodic_cm2: float              # surface area from cathodic slope
    easa_mean_cm2: float                  # average of the two, more robust


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
    electrode_id: str
    scan_rates_v_s: np.ndarray
    delta_ep_v: np.ndarray                # peak to peak separation, Ep_a - Ep_c
    log_scan_rates: np.ndarray            # log10(nu)
    anodic_branch_fit: LinearFit | None
    cathodic_branch_fit: LinearFit | None
    alpha: float | None                   # transfer coefficient, 0 to 1
    k0_per_s: float | None                # the rate constant we're after, s^-1
    notes: str                            # explains caveats and when k0 is None


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

    # gather the machine's own picks by segment. positive ip means anodic.
    instrument_ip_a: list[float] = []
    instrument_ip_c: list[float] = []
    for seg in exp.segment_results:
        if seg.ip_a is None:
            continue
        if seg.ip_a > 0:
            instrument_ip_a.append(seg.ip_a)
        else:
            instrument_ip_c.append(seg.ip_a)

    if instrument_ip_a and instrument_ip_c:
        # take the biggest anodic and most negative cathodic across all cycles
        return max(instrument_ip_a), min(instrument_ip_c)

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
        electrode_id=electrode_id,
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


def laviron(experiments: list[CVExperiment]) -> LavironResult:
    """laviron's method for the heterogeneous electron transfer rate, k0.

    laviron showed that when nF*delta_Ep/RT is bigger than about 10, the peak
    potentials shift linearly with log(nu). the slopes of those shifts give us
    the transfer coefficient alpha, and from that we get k0.

        Ep_a - E0' goes like (2.303 RT / ((1-alpha) n F)) * log(nu)
        Ep_c - E0' goes like -(2.303 RT / (alpha n F)) * log(nu)

    big caveat. laviron strictly applies to surface bound redox species, but
    ferro/ferricyanide is freely diffusing in solution. so the k0 we get out is
    an order of magnitude guess at best. we mention this in the notes field of
    the result so it's clear what we're looking at.

    also, we only fit the high overpotential branch where delta_Ep is bigger
    than about 100 mV. that's where the kinetic regime kicks in. if not enough
    of our scan rates land in that regime we just return k0 = None and explain
    in the notes.
    """
    if not experiments:
        raise ValueError("laviron, got an empty experiment list")

    electrode_ids = {e.file_meta.electrode_id for e in experiments}
    if len(electrode_ids) != 1:
        raise ValueError(
            f"laviron, all the experiments need to be from one electrode. "
            f"got {electrode_ids}"
        )
    electrode_id = electrode_ids.pop()

    notes_parts: list[str] = []
    if any(e.file_meta.electrolyte != Electrolyte.FERRO_FERRI for e in experiments):
        raise ValueError("laviron needs ferro/ferri data")

    # always include the caveat up front
    notes_parts.append(
        "ferro/ferricyanide is diffusion controlled, not surface confined. "
        "laviron strictly applies to adsorbed species, so the k0 here is an "
        "order of magnitude estimate from the high overpotential region of "
        "dEp vs log(nu)."
    )

    experiments = sorted(experiments, key=lambda e: e.scan_rate_v_s)
    nus = np.array([e.scan_rate_v_s for e in experiments])
    log_nus = np.log10(nus)

    # pull anodic and cathodic peak potentials from the segment results.
    # segments with positive ip are anodic, negative ip are cathodic.
    Ep_a = np.full(len(experiments), np.nan)
    Ep_c = np.full(len(experiments), np.nan)
    for i, e in enumerate(experiments):
        for seg in e.segment_results:
            if seg.ep_v is None or seg.ip_a is None:
                continue
            if seg.ip_a > 0 and np.isnan(Ep_a[i]):
                Ep_a[i] = seg.ep_v
            elif seg.ip_a < 0 and np.isnan(Ep_c[i]):
                Ep_c[i] = seg.ep_v

    # need at least three valid peak pairs to even attempt a fit
    valid = ~(np.isnan(Ep_a) | np.isnan(Ep_c))
    if valid.sum() < 3:
        notes_parts.append("not enough valid peak pairs to fit.")
        return LavironResult(
            electrode_id=electrode_id,
            scan_rates_v_s=nus,
            delta_ep_v=Ep_a - Ep_c,
            log_scan_rates=log_nus,
            anodic_branch_fit=None,
            cathodic_branch_fit=None,
            alpha=None,
            k0_per_s=None,
            notes=" ".join(notes_parts),
        )

    delta_Ep = Ep_a - Ep_c  # peak separation

    # the kinetic regime needs nF*delta_Ep/RT > about 4, which means
    # delta_Ep needs to be bigger than 4*RT/(nF), or about 100 mV at room temp.
    threshold_v = 4 * R_CONST * T_KELVIN / (N_ELECTRONS * F_CONST)
    kinetic_mask = valid & (delta_Ep > threshold_v)
    if kinetic_mask.sum() < 3:
        notes_parts.append(
            f"fewer than three scan rates in the kinetic region "
            f"(delta_Ep > {threshold_v*1000:.0f} mV). can't fit reliably, "
            f"returning k0 = None."
        )
        return LavironResult(
            electrode_id=electrode_id,
            scan_rates_v_s=nus,
            delta_ep_v=delta_Ep,
            log_scan_rates=log_nus,
            anodic_branch_fit=None,
            cathodic_branch_fit=None,
            alpha=None,
            k0_per_s=None,
            notes=" ".join(notes_parts),
        )

    # E0' is the formal potential. for ferro/ferri, the average of Ep_a and
    # Ep_c is a fine estimate. we take the mean across all the kinetic region points.
    E0_prime = float(np.mean((Ep_a[kinetic_mask] + Ep_c[kinetic_mask]) / 2))
    notes_parts.append(f"used E0' = {E0_prime*1000:.1f} mV (midpoint average).")

    # fit each branch in the kinetic region. these slopes are what laviron's
    # equations relate to alpha.
    fit_a = _linear_fit(log_nus[kinetic_mask], Ep_a[kinetic_mask] - E0_prime)
    fit_c = _linear_fit(log_nus[kinetic_mask], Ep_c[kinetic_mask] - E0_prime)

    # solve for alpha from each branch and average them.
    # anodic slope = 2.303*RT/((1-alpha)*nF), so (1-alpha) = 2.303*RT/(slope*nF).
    # cathodic slope = -2.303*RT/(alpha*nF), so alpha = -2.303*RT/(slope*nF).
    factor = 2.303 * R_CONST * T_KELVIN / (N_ELECTRONS * F_CONST)
    one_minus_alpha = factor / fit_a.slope if fit_a.slope != 0 else math.nan
    alpha_from_c = -factor / fit_c.slope if fit_c.slope != 0 else math.nan
    alpha = float(np.mean([1 - one_minus_alpha, alpha_from_c]))

    # alpha has to be between 0 and 1. if it's not, the model isn't valid here.
    if not (0 < alpha < 1):
        notes_parts.append(
            f"computed alpha = {alpha:.3f} is outside (0, 1), so the kinetic "
            f"model isn't valid for this data. returning k0 = None."
        )
        return LavironResult(
            electrode_id=electrode_id,
            scan_rates_v_s=nus,
            delta_ep_v=delta_Ep,
            log_scan_rates=log_nus,
            anodic_branch_fit=fit_a,
            cathodic_branch_fit=fit_c,
            alpha=alpha,
            k0_per_s=None,
            notes=" ".join(notes_parts),
        )

    # final step, k0. laviron's closed form at one reference scan rate is
    #   log(k0) = alpha*log(1-alpha) + (1-alpha)*log(alpha)
    #           - log(RT/(nF*nu)) - alpha*(1-alpha)*nF*dEp/(2.303*RT)
    # we plug in the smallest nu in the kinetic region and its dEp.
    nu_ref = float(nus[kinetic_mask][0])
    dEp_ref = float(delta_Ep[kinetic_mask][0])

    log_k0 = (
        alpha * math.log10(1 - alpha)
        + (1 - alpha) * math.log10(alpha)
        - math.log10(R_CONST * T_KELVIN / (N_ELECTRONS * F_CONST * nu_ref))
        - alpha * (1 - alpha) * N_ELECTRONS * F_CONST * dEp_ref / (2.303 * R_CONST * T_KELVIN)
    )
    k0 = 10 ** log_k0

    return LavironResult(
        electrode_id=electrode_id,
        scan_rates_v_s=nus,
        delta_ep_v=delta_Ep,
        log_scan_rates=log_nus,
        anodic_branch_fit=fit_a,
        cathodic_branch_fit=fit_c,
        alpha=alpha,
        k0_per_s=float(k0),
        notes=" ".join(notes_parts),
    )
