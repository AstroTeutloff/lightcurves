"""
Base Lightcurve package. Specific LC classes inherit from this one.

@author: Felix Teutloff
@date: 09-2025
@version: 0.2
"""

from abc import ABC, abstractmethod
from warnings import warn

import matplotlib.axes as axes

from astropy import units as u
from astropy.table import QTable
from astropy.timeseries import LombScargle
from astropy.io.ascii import write


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

    @abstractmethod
    def plot_lightcurve(
        self,
        bands: list | str,
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        **plot_kwargs,
    ) -> axes.Axes:
        pass

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
