"""
Package for BlackGEM lightcurve analysis.

@author: Felix Teutloff
@date: 09-2025
@version: 0.3
"""

from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes

from astropy import units as u, coordinates as c, time as t
from astropy.timeseries import LombScargle, LombScargleMultiband
from astropy.table import QTable
from astropy.stats import sigma_clip

import lightcurves.timeseries as ts
from lightcurves.statistics import statistics
from lightcurves.BaseLightcurve import BaseLightcurve


class BGLightcurve(BaseLightcurve):
    """
    Blackgem lightcurve object. Holds a QTable of data.
    """

    OBSERVATORY = c.EarthLocation.of_site("lasilla")
    LOWWARNING = 10

    FLUX_UNIT = u.erg * (u.s**-1) * (u.cm**-2)

    BANDS_INFO = {
        "u": {"color": "cyan"},
        "g": {"color": "g"},
        "q": {"color": "orange"},
        "r": {"color": "r"},
        "i": {"color": "brown"},
        "z": {"color": "k"},
    }

    def __init__(
        self,
        lc_data: dict | list | QTable,
        low_warn: bool = True,
        sig_clip: float | None = None,
    ):
        """
        Constructor for a BGLightcurve.

        Parameters:
        -----------

            lc_data: Data, that comprises the Lightcurve. It can be of type
                dictionary, or astropy (Q)Table.
            low_warn: bool; (optional) Should the user be warned if very few datapoints are
                available.
            sig_clip: float | None; (optional) Sigma-clip the datapoints to the specified
                level. If not declared, this step is skipped.
        """

        lc_data = QTable(lc_data, masked=True)

        self.all = lc_data

        self.all["FNU_OPT"].unit = BGLightcurve.FLUX_UNIT
        self.all["FNUERR_OPT"].unit = BGLightcurve.FLUX_UNIT

        self.all["MJD_OBS"] = t.Time(
            self.all["MJD_OBS"],
            format="mjd",
            location=BGLightcurve.OBSERVATORY
        )

        self.all["COORD"] = coords = c.SkyCoord(
            self.all["RA"], self.all["DEC"], unit=(u.deg, u.deg), frame="icrs"
        )
        # Check if BJD times are all there, if not, calculate them.
        if any(np.isnan(self.all["BJD_OBS"])):
            self.all["BJD_OBS"] = ts.barycentric_correction(
                self.all["MJD_OBS"],
                coords
            )
        else:
            self.all["BJD_OBS"] = t.Time(self.all["BJD_OBS"], format="mjd")

        # Set up LC by filter.
        self.all = self.all.group_by("FILTER")

        # Sigma clip the data, if wished.
        for band, data in zip(self.all.groups.keys, self.all.groups):
            if sig_clip is not None:
                sc = sigma_clip(
                    data=data["FNU_OPT"].data,
                    sigma=sig_clip,
                )
                data["FNU_OPT"][sc.mask] = np.ma.masked
                data["FNUERR_OPT"][sc.mask] = np.ma.masked
                data["MJD_OBS"][sc.mask] = np.ma.masked
                data["BJD_OBS"][sc.mask] = np.ma.masked

        if not low_warn:
            return

        for band in BGLightcurve.BANDS_INFO.keys():
            try:
                band_len = len(self[band])
            except KeyError:
                continue

            if 0 < band_len < BGLightcurve.LOWWARNING:
                warn(
                    f"WARNING: {band}-band has less than "
                    f"{BGLightcurve.LOWWARNING} datapoints. ({band_len})"
                )

    @classmethod
    def lomb_scargle(
        cls,
        data: QTable,
        **ls_kwargs,
    ) -> LombScargle:
        """
        Perform a LombScargle (single band) analysis on `BlackGEM-data shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `BJD_OBS`,
                `FNU_OPT`, `FNUERR_OPT`.
            **ls_kwargs; Any additional keyword-arguments are passed to the
                constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.
        """

        data = data[~data["FNU_OPT"].mask]

        if len(set(data["FILTER"])) > 1:
            raise ValueError(
                "Input data seems to have more than 1 bands worth of data. "
                + "Please specify band used, or use subset of table."
            )

        ls_obj = LombScargle(
            data["BJD_OBS"], data["FNU_OPT"], data["FNUERR_OPT"], **ls_kwargs
        )

        return ls_obj

    @classmethod
    def lomb_scargle_multiband(cls, data: QTable, **ls_kwargs) -> LombScargle:
        """
        Perform a multiband LombScargle (multiband) analysis on
        `BlackGEM-data shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `BJD_OBS`,
                `FNU_OPT`, `FNUERR_OPT`, and `FILTER`.
            **ls_kwargs; Any additional keyword-arguments are passed to the
            constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.
        """

        data = data[~data["FNU_OPT"].mask]

        ls_obj = LombScargleMultiband(
            data["BJD_OBS"],
            data["FNU_OPT"],
            data["FILTER"],
            data["FNUERR_OPT"],
            **ls_kwargs,
        )

        return ls_obj

    def plot_folded(
        self,
        period: u.Quantity,
        bands: list | str = ["u", "g", "q", "r", "i", "z"],
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        n_periods: int = 2,
        normalize: bool = True,
        **plot_kwargs,
    ) -> axes.Axes:
        """
        Simple method that plots the phasefolded Lightcurve(s). If you want
        something specific, I recommend to grab the phasefolded lightcurves via
        the phasefold method.

        Parameters:
        -----------

            period: u.Quantity; A period over which the curve is to be
                phasefolded over.
            bands: list[str] | str; (optional) List of bands that are to be plotted.
                Options are `u`, `g`, `q`, `r`, `i`, `z`.
                Alternatively, a string can be used. Syntax is the same, without
                spaces. By default all bands are plotted.
            ax: axes.Axes object; (optional) The plotting axis to use. If not declared in
                call, a new figure object is created.
            show_uncertainty: bool; (optional) Whether or not to plot with errorbars.
            n_periods : int; (optional) Amount of periods the phasefolded lc should be
                plotted over.
            normalize : bool; (optional) Should each band be normalized to the mean of the
                bands flux.

        Returns:
        --------

            axes.Axes; The axes object that was either put in, or created for the plot.
        """

        ax = super()._verify_ax(ax)

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in bands_info (see constructor)
                band_info = BGLightcurve.BANDS_INFO[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    "Please use only the bands `u`,`g`, `r`, `i` and/or `z`!"
                ) from ke

            # This is able to fail, because not all of the Filters will be
            # available in all of the Lightcurves.
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            if normalize:
                # TODO: Calculate Error correctly!!
                flux = field["FNU_OPT"] / np.nanmean(field["FNU_OPT"])
                fluxerr = field["FNUERR_OPT"] / np.nanmean(field["FNU_OPT"])
            else:
                flux = field["FNU_OPT"]
                fluxerr = field["FNUERR_OPT"]

            yerr = fluxerr if show_uncertainty else None

            ax = super()._plot_band_folded(
                time=field["BJD_OBS"],
                period=period,
                t0=np.max(self.all["BJD_OBS"]),
                flux=flux,
                fluxerr=yerr,
                ax=ax,
                color=band_info["color"],
                label=band,
                n_periods=n_periods,
                **plot_kwargs
            )

        if normalize:
            ylabel = r"Mean weighted flux $F/\bar{F}$ [1]"
        else:
            ylabel = rf"Flux $F$[{BGLightcurve.FLUX_UNIT}]"

        ax.set_ylabel(ylabel)

        return ax

    def plot_periodogram(
        self,
        freq_space: u.Quantity,
        power_space: u.Quantity,
        band: str = "",
        ax: axes.Axes | None = None,
        mark_maximum: bool = False,
        fal: float | None = None,
        draw_period_axis: bool = True,
        **plot_kwargs,
    ) -> axes.Axes:
        """
        Method that plots the periodogram for a given frequency range.

        Parameters:
        -----------

            freq_space: u.Quantity; A range of frequencies. Makes up the x-axis data.
            power_space: u.Quantity; A corresponding list of power data. Makes up y-axis data.
            band: str; (optional) ID of the band that is plotted, options are
                `ugqriz`. This is just used for colouring the plot.
            ax: axes.Axes object; (optional) The plotting axis to use. If not declared in
                call, a new figure object is created. Default is `None`.
            mark_maximum: bool; (optional) Put a marker on the maximum frequency. Default is false.
            fal: float; (optional) A false alarm level. An axhline is created at its' height.
            draw_period_axis: bool; (optional) Should the periodogram have an twin x axis
                at the top for Period values (corresponding to frequencies).
            plot_kwargs; Further keywords are passed to the call of plt.plot as
                keyword arguments

        Returns:
        --------

            axes.Axes; The axes object that was either put in, or created for the plot.
        """

        ax = super()._verify_ax(ax)

        plot_color, label_prefix = super()._periodogram_color_and_labelprefix(
            band,
            self.BANDS_INFO
        )

        ax = super()._plot_periodogram(
            freq_space=freq_space,
            power_space=power_space,
            fal=fal,
            ax=ax,
            mark_maximum=mark_maximum,
            draw_period_axis=draw_period_axis,
            color=plot_color,
            label_prefix=label_prefix,
            **plot_kwargs
        )

        return ax

    def plot_lightcurve(
        self,
        bands: list | str = ["u", "g", "q", "r", "i", "z"],
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        **plot_kwargs,
    ) -> axes.Axes:
        """
        Plots the lightcurve data.

        Parameters:
        -----------
            bands: list[str] | str; (optional) List of bands that are to be plotted.
                By default, all are plotted.
            ax: axes.Axes object; (optional) The plotting axis to use. If not declared in
            show_uncertainty: bool; (optional) Show show uncertainty bars for flux.
            plot_kwargs; Further keywords are passed to the call of plt.plot as
                keyword arguments

        Returns:
        --------
            axes.Axes; The axes object that was either put in, or created for the plot.
        """

        ax = super()._verify_ax(ax)

        # NOTE: Could bands be a set?
        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in BANDS_INFO
                band_info = BGLightcurve.BANDS_INFO[band.lower()]
            except KeyError as ke:
                raise KeyError("Please use only the bands in `ugqiz`") from ke

            # This can fail because not every lightcurve has all Filters worth of data
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            yerr = field["FNUERR_OPT"] if show_uncertainty else None

            ax = super()._plot_lightcurve(
                time=field["BJD_OBS"],
                flux=field["FNUERR_OPT"],
                fluxerr=yerr,
                ax=ax,
                color=band_info["color"],
                label_prefix=band,
                **plot_kwargs
            )

            ax.set_xlabel("Time (BJD) [d]")
            ax.set_ylabel(rf"Flux $F$ [{BGLightcurve.FLUX_UNIT}]")

        return ax

    def generate_fspace(
        self,
        f_min: u.Quantity | None = None,
        f_max: u.Quantity | None = None,
        oversample: float = 1.0,
    ) -> u.Quantity:
        """
        Specific implementation of the generate_fspace function of the timeseries
        module. I.e.: Generates a frequency space with resolution based on the
        lightcurve data and parameters.

        Parameters:
        -----------
            f_min: u.Quantity; (optional) Minimum frequency you wish to include.
            f_max: u.Quantity; (optional) Maximum frequency you wish to include.
            oversample: float; (optional) Oversampling factor

        Returns:
        --------
            f_grid: u.Quantity; A frequency grid
        """

        return ts.generate_fspace(
            t_values=self.all["BJD_OBS"],
            f_min=f_min,
            f_max=f_max,
            oversample=oversample,
        )

    def __getitem__(self, filter_id: str) -> QTable:
        """
        Convenience method for getting the Subtable for a specific filter.

        Parameters:
        -----------
            filter_id: str; The ID of the filter you want to get data for.

        Returns:
        --------
            QTable; Sub-table with the data, corresponding to the filter.

        Raises:
        -------
            KeyError; If Filter ID is not included in table.
        """

        for group_id, key in enumerate(self.all.groups.keys):
            if key[0] == filter_id:
                return self.all.groups[group_id]

        raise KeyError(
            f"Filter `{filter_id}` is not available. "
            + f"Available filters are: {[i[0] for i in self.all.groups.keys]}"
        )

    def flux_statistics(
        self,
        band: str
    ) -> (statistics, statistics):
        """
        Creates a statistics object for the flux and the flux errors.

        Parameters:
        -----------

            band: str; Filter band for which the statistics is to be created.

        Returns:
        --------

            (statistics, statistics); Two statistics objects, one for the flux,
                one for the flux errors.
        """
        band_data = self[band]
        return statistics(band_data["FNU_OPT"]), statistics(band_data["FNUERR_OPT"])
