"""
Base Lightcurve package. Specific LC classes inherit from this one.

@author: Felix Teutloff
@date: 09-2025
@version: 0.2
"""

from abc import ABC, abstractmethod
from warnings import warn

import matplotlib.axes as axes
import matplotlib.pyplot as plt

from astropy import units as u, time as t
from astropy.table import QTable
from astropy.timeseries import LombScargle
from astropy.io.ascii import write

from lightcurves import timeseries as ts

import numpy as np


class BaseLightcurve(ABC):
    """
    Base Lightcurve object, specific LC objects inherit from this one.
    """

    # Locaton of observatory is used in conversion from MJD measurements to BJD.
    OBSERVATORY: str

    # Warn the user that the light curve has too little datapoints.
    LOWWARNING: int = 10

    @abstractmethod
    def __init__(
        self,
        lc_data: dict | list | QTable,
        low_warn: bool,
        sig_clip: float | None = None,
    ):
        pass

    @classmethod
    @abstractmethod
    def lomb_scargle(cls, data: QTable, **ls_kwargs) -> LombScargle:
        pass

    @classmethod
    @abstractmethod
    def lomb_scargle_multiband(cls, data: QTable, **ls_kwargs) -> LombScargle:
        pass

    @abstractmethod
    def plot_periodogram(
        self,
        freq_space: u.Quantity,
        power_space: u.Quantity,
        band: str = "",
        ax: axes.Axes | None = None,
        mark_maximum: bool = False,
        fal: float | None = None,
        **plot_kwargs,
    ) -> axes.Axes:
        pass

    @classmethod
    def _plot_periodogram(
        cls,
        freq_space: u.Quantity,
        power_space: u.Quantity,
        fal: float | None,
        ax: axes.Axes,
        mark_maximum: bool,
        draw_period_axis: bool,
        color: str,
        label_prefix: str,
        ** plot_kwargs
    ) -> axes.Axes:

        ax.step(freq_space, power_space, c=color, **plot_kwargs)

        # Mark the maximum value
        if mark_maximum:
            pmax_idx = np.argmax(power_space)
            ax.scatter(
                freq_space[pmax_idx],
                1.1 * power_space[pmax_idx],  # x, y
                marker="v",
                c=color,
                edgecolors="black",  # marker specifications
                label=label_prefix
                + r"$f(p_{max})$"
                + f" = {freq_space[pmax_idx]:.2f}",  # label
            )

        # Include a False alarm level line
        if fal is not None:
            ax.axhline(fal, ls="--", color=color, alpha=0.5)

        ax.set_xlabel(f"Frequency $f$ [{freq_space.unit}]")
        ax.set_ylabel(r"Power $p$ [1]")
        ax.set_xlim(np.min(freq_space).value, np.max(freq_space).value)

        # Return early
        if not draw_period_axis:
            return ax

        # Draw second axis at top with Period ticks.
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ticklabels = np.pow(ax.get_xticks() * freq_space.unit, -1).to(u.minute)
        ax_top.set_xticks(ax.get_xticks())
        ax_top.set_xticklabels([f"{i:.2f}" for i in ticklabels.value])
        ax_top.set_xlabel(f"Period $P$ [{ticklabels.unit}]")

        return ax

    @classmethod
    def _periodogram_color_and_labelprefix(cls, band, bands_info):
        if band in bands_info.keys():
            plot_color = bands_info[band]["color"]
            label_prefix = f"{band}: "
        else:
            # NOTE: Maybe put a warning here.
            plot_color = "k"
            label_prefix = ""

        return plot_color, label_prefix

    @abstractmethod
    def plot_lightcurve(
        self,
        bands: list | str,
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        **plot_kwargs,
    ) -> axes.Axes:
        pass

    @classmethod
    def _plot_lightcurve(
        cls,
        time: t.Time,
        flux: u.Quantity,
        fluxerr: u.Quantity | None,
        ax: axes.Axes,
        color: str,
        label_prefix: str,
        **plot_kwargs
    ) -> axes.Axes:

        n_points = f"{label_prefix} ({len(flux) - np.count_nonzero(flux.mask)} points)"

        ax.errorbar(
            time.mjd,
            flux,
            yerr=fluxerr,
            c=color,
            fmt="o",
            label=n_points,
            **plot_kwargs,
        )
        return ax

    @abstractmethod
    def plot_folded(
        self,
        period: u.Quantity,
        bands: list | str = "",
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        n_periods: int = 2,
        normalize: bool = True,
        **plot_kwargs,
    ) -> axes.Axes:
        pass

    @classmethod
    def _plot_band_folded(
        cls,
        time: t.Time,
        period: u.Quantity,
        t0: u.Quantity | t.Time,
        flux: u.Quantity,
        fluxerr: u.Quantity | None,
        ax: axes.Axes,
        color: str,
        n_periods: int,
        **plot_kwargs
    ) -> axes.Axes:

        phase = ts.phasefold(time, period, t0)

        for offset in range(n_periods):
            ax.errorbar(
                phase + offset,
                flux,
                yerr=fluxerr,
                color=color,
                fmt="o",
                **plot_kwargs,
            )
        ax.set_xlabel(r"Phase $\Phi$ [$2\pi$]")
        return ax

    @abstractmethod
    def generate_fspace(
        self,
        f_min: u.Quantity | None = None,
        f_max: u.Quantity | None = None,
        oversample: float = 1.0,
    ) -> u.Quantity:
        pass

    @classmethod
    def write_lcurve_file(
        cls,
        t_values: u.Quantity,
        t_exp: u.Quantity,
        flux_values: u.Quantity,
        flux_unc: u.Quantity,
        weight_1: u.Quantity,
        weight_2: u.Quantity,
        **write_kwargs
    ) -> None:
        """
        Writes contents of a lightcurve to a file which can be used by lcurve
        to analyse.

        Parameters:
        -----------

            t_values: u.Quantity; Values in time- (or phase-) space.
            t_exp: u.Quantity; Exposure times in the same unit as t_values. If
                both t_values and t_exp are in time space and have units, the
                function will handle the conversion by itself!
            flux_values: u.Quantity; Single band flux values
            flux_unc: u.Quantity; Flux uncertainties, corresponding to the
                values for flux_values.
            weight_1: u.Quantity; Weighting column 1. # TODO: Find out what this is specifically for
            weight_2: u.Quantity; Weighting column 2. # TODO: Find out what this is specifically for
            **write_kwargs; Any other keyword arguments passed to the function
                call are passed into the `astropy.io.ascii.write()` call.
        """

        try:
            t_exp = t_exp.to(t_values.unit)
        except u.UnitConversionError:
            warn(
                "Unit conversion Error occured when converting exposure time" +
                " units to input time units. Be careful with your outputs!"
            )

        out_table = QTable(
            [t_values, t_exp, flux_values, flux_unc, weight_1, weight_2],
            names=["TIME", "T_EXP", "FLUX", "FLUX_UNC", "W1", "W2"]
        )

        write(table=out_table, **write_kwargs)

        return None

    @classmethod
    def _verify_ax(
        cls,
        ax: axes.Axes | None
    ) -> axes.Axes:
        return plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax
