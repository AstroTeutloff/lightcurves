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

import lightcurves.timeseries as ts
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

        if len_filter == 1:
            filter = len_bright * filter
        elif len_filter != len_bright:
            raise ValueError("Incompatible lengths of `brightness` and `filter`.")

        if len_time != len_bright:
            raise ValueError("Inconsistent lengths of `time` and `brightness`.")
        if len_bright != len_bright_unc:
            raise ValueError(
                "Inconsistent lengths of `brightness` and `brightness_unc`."
            )

        # If wanted, perform barycentric correction
        if do_barycorr:
            barycentric_time = ts.barycentric_correction(time, obj_coordinates)
        else:
            barycentric_time = time

        # Assemble Table
        self.all = QTable(
            data=[time, barycentric_time, brightness, brightness_unc, filter],
            names=["TIME", "TIME_BARY", "BRIGHTNESS", "BRIGHTNESS_UNC", "FILTER"],
            masked=True,
        )

        # Set up LC by filter.
        self.all = self.all.group_by("FILTER")

        # Sigma clip the data, if wished.
        for band, data in zip(self.all.groups.keys, self.all.groups):
            if sig_clip is not None:
                sc = sigma_clip(
                    data=data["BRIGHTNESS"].data,
                    sigma=sig_clip,
                )
                data["BRIGHTNESS"][sc.mask] = np.ma.masked
                data["BRIGHTNESS_UNC"][sc.mask] = np.ma.masked
                data["TIME"][sc.mask] = np.ma.masked
                data["TIME_BARY"][sc.mask] = np.ma.masked

        if not low_warn:
            return

        for band, data in zip(self.all.groups.keys, self.all.groups):
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

        # Creating the axes object if it is not declared.
        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax
        if bands == "":
            bands = self.all.groups.keys

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in bands_info (see constructor)
                band_info = self.bands_info[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    f"Please use only the bands {self.all.groups.keys}!"
                ) from ke

            # This is able to fail, because not all of the Filters will be
            # available in all of the Lightcurves.
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            time_pf = ts.phasefold(
                field["TIME_BARY"], period, t0=np.max(self.all["TIME_BARY"])
            )

            # Start plotting
            if normalize:
                # TODO: Calculate Error correctly!!
                flux = field["BRIGHTNESS"] / np.nanmean(field["BRIGHTNESS"])
                fluxerr = field["BRIGHTNESS_UNC"] / np.nanmean(field["BRIGHTNESS"])
            else:
                flux = field["BRIGHTNESS"]
                fluxerr = field["BRIGHTNESS_UNC"]

            yerr = fluxerr if show_uncertainty else None

            for offset in range(n_periods):
                ax.errorbar(
                    time_pf + offset,
                    flux,
                    yerr=yerr,
                    label=band,
                    color=band_info["color"],
                    fmt="o",
                    **plot_kwargs,
                )

        if normalize:
            ylabel = r"Mean weighted flux $F/\bar{F}$ [1]"
        else:
            ylabel = "Brightness [TODO: Set unit]"

        ax.set_xlabel(r"Phase $\Phi$ [$2\pi$]")
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

        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        # Draw the periodogram
        if band in self.bands_info.keys():
            plot_color = self.bands_info[band]["color"]
            label_prefix = f"{band.lower()}: "
        else:
            plot_color = "k"
            label_prefix = ""

        ax.step(freq_space, power_space, c=plot_color, **plot_kwargs)

        # Mark the maximum value
        if mark_maximum:
            pmax_idx = np.argmax(power_space)
            ax.scatter(
                freq_space[pmax_idx],
                1.1 * power_space[pmax_idx],  # x, y
                marker="v",
                c=plot_color,
                edgecolors="black",  # marker specifications
                label=label_prefix
                + r"$f(p_{max})$"
                + f" = {freq_space[pmax_idx]:.2f}",  # label
            )

        # Include a False alarm level line
        if fal is not None:
            ax.axhline(fal, ls="--", color=plot_color, alpha=0.5)

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

        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        if bands == "":
            bands = self.all.groups.keys

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in BANDS_INFO
                d = self.bands_info[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    f"Please use only the bands in {self.all.groups.keys}"
                ) from ke

            # This can fail because not every lightcurve has all Filters worth of data
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            n_points = f"{band} ({len(field) - np.count_nonzero(field['BRIGHTNESS'].mask)} points)"

            yerr = field["BRIGHTNESS_UNC"] if show_uncertainty else None

            ax.errorbar(
                field["TIME"].mjd,
                field["BRIGHTNESS"],
                yerr=yerr,
                c=d["color"],
                fmt="o",
                label=n_points,
                **plot_kwargs,
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

        for group_id, key in enumerate(self.all.groups.keys):
            if key[0] == filter_id:
                return self.all.groups[group_id]

        raise KeyError(
            f"Filter `{filter_id}` is not available. "
            + f"Available filters are: {[i[0] for i in self.all.groups.keys]}"
        )
