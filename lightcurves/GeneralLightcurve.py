"""
Package for GeneralLightcurve analysis.

@author: Felix Teutloff
@date: 02-2026
@version: 0.1
"""

from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes

from astropy import units as u, coordinates as c, time as t
from astropy.timeseries import LombScargle, LombScargleMultiband
from astropy.table import QTable
from astropy.stats import sigma_clip
from astropy.utils.masked import Masked

import lightcurves.timeseries as ts
from lightcurves.statistics import statistics
from lightcurves.BaseLightcurve import BaseLightcurve


class GeneralLightcurve(BaseLightcurve):
    """
    General lightcurve object. Holds a QTable of data.
    """

    LOWWARNING = 10

    def __init__(
        self,
        time: t.Time,
        brightness: u.Quantity,
        brightness_unc: u.Quantity,
        filter: list[str],
        t_exposure: u.Quantity = 0*u.second,
        obj_coordinates: c.SkyCoord | None = None,
        do_barycorr: bool = False,
        bands_info: dict = {},
        low_warn: bool = True,
        sig_clip: float | None = None,
    ):
        """
        Constructor for a general light curve object.

        Parameters:
        -----------

            time: t.Time; Observation times. Can be geocentric, or
                helio/barycentric. If `do_barycorr` is set to true, the times
                are expected to have a location associated.
            brightness: u.Quantity; Some brightnesses, corresponding to the
                observation times. The length of `brightness` is to be the same
                length as `time`.
            brightness_unc: u.Quantity; Brightness uncertainties, corresponding
                to `brightness`. The length of `brightness_unc` is to be the
                same length as `brightness`.
            filter: list[str]; List of filters, the individual brightnesses are
                in. The length of `filters` has to be the same as `brightness`,
                or be 1.
            t_exposure: u.Quantity | None; (optional) Exposure times in time
                units. Length has to be the same as `time` or 1. Default
                is 0 seconds.
            obj_coordinates; c.SkyCoord | None; (optional) Coordinates of the
                observed object.
            do_barycorr: bool; (optional) Perform a barycentric correction for
                the time data. If this is set to `True`, observartory, and
                obj_coordinates have to be set! If not declared, this step is
                skipped.
            bands_info: dict; (optional) A dictionary of supplemental
                information. Keys are expected to be the same as for `filters`.
                A specific sub-key that is expected is `color`.
            low_warn: bool; (optional) Should the user be warned if very few
                datapoints are available.
            sig_clip: float | None; (optional) Sigma-clip the datapoints to the
                specified level. If not declared, this step is skipped.

        """

        # Set up bands_info
        self.bands_info = bands_info

        # If the list of filters has values that are not in `bands_info`s keys,
        # raise an error. But only if `bands_info` is not an empty dictionary.
        if len(set(filter) - set(bands_info.keys())) > 0 and len(bands_info.keys()) > 0:
            raise ValueError(
                "Values in `filter` with no corresponding `bands_info` value"
            )

        # Perform checks if input lengths are the same.
        len_time = len(time)
        len_bright = len(brightness)
        len_bright_unc = len(brightness_unc)
        len_filter = len(filter)
        try:
            len_t_exp = len(t_exposure)
        except TypeError:
            len_t_exp = 1

        if len_filter == 1:
            filter = len_bright * filter
        elif len_filter != len_bright:
            raise ValueError(
                "Incompatible lengths of `brightness` and `filter`.")

        if len_time != len_bright:
            raise ValueError(
                "Inconsistent lengths of `time` and `brightness`.")
        if len_bright != len_bright_unc:
            raise ValueError(
                "Inconsistent lengths of `brightness` and `brightness_unc`."
            )

        if len_t_exp == 1:
            t_exposure = np.full(len_time, fill_value=t_exposure)
        elif len_t_exp != len_time:
            raise ValueError(
                "Incompatible lengths of `time` and `t_exposure`."
            )

        # If wanted, perform barycentric correction
        if do_barycorr:
            barycentric_time = ts.barycentric_correction(time, obj_coordinates)
        else:
            barycentric_time = time

        # Assemble Table
        self.all = QTable(
            data=[
                time,
                barycentric_time,
                Masked(brightness),
                Masked(brightness_unc),
                filter,
                t_exposure
            ],
            names=[
                "TIME", "TIME_BARY",
                "BRIGHTNESS", "BRIGHTNESS_UNC",
                "FILTER",
                "T_EXP"
            ],
            masked=True,
        )

        # Set up LC by filter.
        self.all = self.all.group_by("FILTER")

        # Sigma clip the data, if wished.
        for band, data in zip(self.available_filters(), self.all.groups):
            if sig_clip is not None:
                sc = sigma_clip(
                    data=data["BRIGHTNESS"].data,
                    sigma=sig_clip,
                )
                data["BRIGHTNESS"][sc.mask] = np.ma.masked
                data["BRIGHTNESS_UNC"][sc.mask] = np.ma.masked
                data["TIME"][sc.mask] = np.ma.masked
                data["TIME_BARY"][sc.mask] = np.ma.masked

        if (
            # If the set of used filters - the set of filters with color
            # information is not the emtpy set, warn the user.
            set([str(k[0]) for k in self.available_filters()]) -
                set(self.bands_info.keys())
        ):
            warn(
                "`bands_info` does not have the same keys as there are filters." +
                " Expect the plotting to fail!")
        if not low_warn:
            return

        for band, data in zip(self.available_filters(), self.all.groups):
            try:
                band_len = len(data)
            except KeyError:
                continue

            if 0 < band_len < GeneralLightcurve.LOWWARNING:
                warn(
                    f"WARNING: {band}-band has less than "
                    f"{GeneralLightcurve.LOWWARNING} datapoints. ({band_len})"
                )

    @classmethod
    def lomb_scargle(
        cls,
        data: QTable,
        **ls_kwargs,
    ) -> LombScargle:
        """
        Perform a LombScargle (single band) analysis on the `GeneralLightcurve-shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `TIME_BARY`,
                `BRIGHTNESS`, `BRIGHTNESS_UNC`.
            **ls_kwargs; Any additional keyword-arguments are passed to the
                constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.
        """

        data = data[~data["BRIGHTNESS"].mask]

        if len(set(data["FILTER"])) > 1:
            raise ValueError(
                "Input data seems to have more than 1 bands worth of data. "
                + "Please specify band used, or use subset of table."
            )

        ls_obj = LombScargle(
            data["TIME_BARY"], data["BRIGHTNESS"], data["BRIGHTNESS_UNC"], **ls_kwargs
        )

        return ls_obj

    @classmethod
    def lomb_scargle_multiband(cls, data: QTable, **ls_kwargs) -> LombScargle:
        """
        Perform a multiband LombScargle (multiband) analysis on
        `GeneralLightcurve-shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `TIME_BARY`,
                `BRIGHTNESS`, `BRIGHTNESS_UNC`, and `FILTER`.
            **ls_kwargs; Any additional keyword-arguments are passed to the
            constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.
        """

        data = data[~data["BRIGHTNESS"].mask]

        ls_obj = LombScargleMultiband(
            data["TIME_BARY"],
            data["BRIGHTNESS"],
            data["FILTER"],
            data["BRIGHTNESS_UNC"],
            **ls_kwargs,
        )

        return ls_obj

    def plot_folded(
        self,
        period: u.Quantity,
        bands: list | str = "",
        ephemeris: t.Time | None = None,
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
                Alternatively, a string can be used. Syntax is the same, without
                spaces. By default all bands are plotted.
            ephemeris: t.Time; (optional) An ephemeris (T_0) zerophase for the
                phasefold. Default is `None`, then the latest datapoint is
                taken as the ephemeris.
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

        if bands == "":
            bands = [band for band in self.available_filters()]

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in bands_info (see constructor)
                band_info = self.bands_info[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    f"Please use only the bands {self.available_filters()}!"
                ) from ke

            field = self[band]
            # Start plotting
            if normalize:
                # TODO: Calculate Error correctly!!
                flux = field["BRIGHTNESS"] / np.nanmean(field["BRIGHTNESS"])
                fluxerr = field["BRIGHTNESS_UNC"] / \
                    np.nanmean(field["BRIGHTNESS"])
            else:
                flux = field["BRIGHTNESS"]
                fluxerr = field["BRIGHTNESS_UNC"]

            yerr = fluxerr if show_uncertainty else None

            ax = super()._plot_band_folded(
                time=field["TIME_BARY"],
                period=period,
                t0=ephemeris if ephemeris is not None else np.max(
                    self.all["TIME_BARY"]),
                flux=flux,
                fluxerr=yerr,
                ax=ax,
                color=band_info["color"],
                n_periods=n_periods,
                **plot_kwargs
            )
        if normalize:
            ylabel = r"Mean weighted flux $F/\bar{F}$ [1]"
        else:
            ylabel = "Brightness [TODO: Set unit]"

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

        plot_color, label_prefix = super()._periodogram_color_and_labelprefix(
            band,
            self.bands_info
        )

        ax = super()._verify_ax(ax)

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
        bands: list | str = "",
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

        if bands == "":
            bands = self.available_filters()

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in BANDS_INFO
                d = self.bands_info[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    f"Please use only the bands in {self.available_filters()}"
                ) from ke

            field = self[band]
            yerr = field["BRIGHTNESS_UNC"] if show_uncertainty else None

            ax = super()._plot_lightcurve(
                time=field["TIME_BARY"],
                flux=field["BRIGHTNESS"],
                fluxerr=yerr,
                ax=ax,
                color=d["color"],
                label_prefix=band,
                ** plot_kwargs
            )

        ax.set_xlabel("Time (BJD) [d]")
        ax.set_ylabel("Brightness [TODO: Set unit]")

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
            t_values=self.all["TIME_BARY"],
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

        for group_id, key in enumerate(self.available_filters()):
            if key[0] == filter_id:
                return self.all.groups[group_id]

        raise KeyError(
            f"Filter `{filter_id}` is not available. "
            + f"Available filters are: {self.available_filters()}"
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
        return statistics(band_data["BRIGHTNESS"]), statistics(band_data["BRIGHTNESS_UNC"])

    def available_filters(self) -> tuple[str]:
        """
        Returns the names of filters available for that light curve.

        Returns:
        --------

            list_filters: tuple[str]; Names of filters the light curve has data
                for. In the case that no filters are available (how?) an empty
                tuple is returned.
        """

        return (filter[0] for filter in self.all.groups.keys)
