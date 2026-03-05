"""
General purpose timeseries functions.

@author: Felix Teutloff
@date: 09-2025
@version: 0.1

"""

from astropy import units as u, time as t, coordinates as c
import numpy as np


def phasefold(
    t_values: t.Time, period: u.Quantity, t0: t.Time | u.Quantity = 0 * u.second
) -> u.Quantity:
    """
    Method that returns a pandas dataframe with the datapoints phase-folded
    along a given period, with an offset of t0.

    Paramters:
    ----------
        t_values: t.Time; Time value(s) that are to be turned to phase values.
        period: astropy.unit.Quantity; The period around which the data is
        to be folded.
        t0: astropy.time.Time or astropy.unit.Quantity = 1 * u.second;
        (optional) Time object by which the zero-phase is offset. If it is
        a Quantity, the value of t0 is checked to be convertible to
        seconds. If it is not possible, it raises a ValueError.

    Returns:
    --------
        u.Quantity; Inputs, shifted by t0 and divided by input period (modulo 1).
    """

    try:
        period.to(u.second)
    except ValueError as ve:
        raise ValueError(
            "The period value cannot be converted to "
            "seconds and is thus not likely a time."
        ) from ve

    if isinstance(t0, u.Quantity):
        try:
            t0.to(u.second)
            period.to(u.second)
        except ValueError as ve:
            raise ValueError(
                "The t0 value cannot be converted to "
                "seconds and is thus not likely a time."
            ) from ve

    def phase(x):
        return (((x - t0).jd) * u.day / period) % 1

    return phase(t_values)


def generate_fspace(
    t_values: t.Time,
    f_min: u.Quantity | None = None,
    f_max: u.Quantity | None = None,
    oversample: float = 1.0,
) -> u.Quantity:
    """
    Creates a frequency space grid, based on the lightcurve data.

    Parameters:
    -----------
        t: t.Time; Time points of the observations.
        f_min: u.Quantity; minimum frequency you wish to include.
        f_max: u.Quantity; maximum frequency you wish to include.
        oversample: float; oversampling factor that is included in the
        resolution calculation.

    Returns:
    --------
        f_grid: u.Quantity; A frequency grid, with resolution based on the inputs.
    """

    t_min = np.min(t_values)
    t_max = np.max(t_values)
    if t_min == t_max:
        raise ValueError("t_range is 0.")

    t_range = t_max - t_min
    t_cadence = np.nanmin(np.diff(np.sort(t_values)))

    f_resolution = 1.0 / t_range / oversample
    if f_max is None:
        f_max = 0.5 / t_cadence
    if f_min is None:
        f_min = f_resolution

    f_resolution_unit = f_resolution.unit

    f_grid = (
        np.arange(
            f_min.to(f_resolution_unit).value,
            f_max.to(f_resolution_unit).value,
            f_resolution.value,
        )
        * f_resolution_unit
    )
    return f_grid


def barycentric_correction(t_values: t.Time, object_coords: c.SkyCoord) -> t.Time:
    """
    Converts time objects from earth to barycenric time.

    Parameters:
    -----------

        t_values: t.Time; Time objects at which the data has been taken. These
            need to have the location set, otherwise it will not be possible to
            perform a correct BJD correction!
        object_coords: c.SkyCoord; Object coordinates in the night sky.

    Returns:
    --------

        t_bjd: t.Time; Time values converted to BJD.
    """
    if t_values.location is None:
        raise AttributeError(
            "Cannot perform Barycentric correction without knowing where the observatory is!"
        )

    t_bjd = t_values + t_values.light_travel_time(object_coords)
    return t_bjd


def weighted_binning(
    phase_start: float, phase_stop: float, bins: int
) -> tuple[list[int], list[float]]:
    """
    Digitizes a phase range into bins, weighted by the length of the phase range.

    Parameters:
    -----------

        phase_start: float; The beginning of the phase range.
        phase_stop: float; The end of the phase range. Has to be strictly larger than phase_start.
        bins: int; Number of bins per full phase (from 0-1)

    Returns:
    --------

        (covered_bins, bin_weights): (list[int], list[float]); List of bins that are covered by the phase range, and their weights
    """

    if phase_start > phase_stop:
        raise ValueError(
            "End of phase range is smaller or equal to beginning of phase range."
        )

    # Adjust phase space to bin space
    phase_bin_start = phase_start * bins
    phase_bin_stop = phase_stop * bins

    phase_baseline = phase_bin_stop - phase_bin_start

    bin_lowest = int(phase_bin_start)
    bin_highest = int(phase_bin_stop)

    # If both values are in the same bin, we can exit early.
    if bin_lowest == bin_highest:
        return ([bin_lowest], [1.0])

    covered_bins = []
    bin_weights = []

    for i in range(
        bin_lowest, bin_highest
    ):
        # Calculate weights per phase bin by checking if the bin edges are
        # nearer than the phase range edges.
        omega_i = min(phase_bin_stop, i + 1) - max(phase_bin_start, i)
        w_i = omega_i / phase_baseline

        covered_bins.append(i)
        bin_weights.append(w_i)

    return (covered_bins, bin_weights)
