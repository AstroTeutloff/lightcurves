"""
Base Lightcurve package. Specific LC classes inherit from this one.

@author: Felix Teutloff
@date: 09-2025
@version: 0.2
"""

from abc import ABC, abstractmethod

import matplotlib.axes as axes

from astropy import units as u
from astropy.table import QTable
from astropy.timeseries import LombScargle


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
