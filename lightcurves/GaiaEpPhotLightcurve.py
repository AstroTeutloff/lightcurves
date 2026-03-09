"""
Package for Gaia epoch photometry lightcurve analysis.

@author: Felix Teutloff
@date: 09-2025
@version: 0.2
"""

from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes

from astropy import units as u, coordinates as c, time as t
from astropy.timeseries import LombScargle, LombScargleMultiband
from astropy.table import QTable, vstack
from astropy.stats import sigma_clip

import lightcurves.timeseries as ts
from lightcurves.BaseLightcurve import BaseLightcurve


class GaiaEpPhotLightcurve(BaseLightcurve):
    """
    Gaia epoch photometry lightcurve object. Holds a QTable of data.
    """

    OBSERVATORY = "GAIA"  # Not really an earth location so, so no coordinates.
    LOWWARNING = 10

    FLUX_UNIT = u.electron * (u.s**-1)
    GAIA_EPOCH_OFFSET = (
        2455197.5  # Time data is given as JD - offset, this is that offset.
    )
    FILTERS = ["G", "BP", "RP"]  # Available Filter bands in Gaia.

    BANDS_INFO = {
        "G": {"color": "g"},
        "BP": {"color": "blue"},
        "RP": {"color": "red"},
    }

    def __init__(
        self,
        lc_data: dict | list | QTable,
        low_warn: bool = True,
        sig_clip: float | None = None,
    ):
        """
        Constructor for a GaiaEpPhotLightcurve.

        Parameters:
        -----------

            lc_data: dict | list | QTable; Data, that comprises the Lightcurve.
                It needs to be castable into a QTable and needs the proper column
                names for Gaia Epoch photometry data.
            low_warn: bool; (optional) Should the user be warned if very few datapoints are
                available.
            sig_clip: float | None; (optional) Sigma-clip the datapoints to the specified
                level. If not declared, this step is skipped. Default is None.
        """

        lc_data = QTable(lc_data, masked=True)

        self.all = lc_data

        # Sigma clip the data, if wished.
        for band in GaiaEpPhotLightcurve.FILTERS:
            data = self.all[f"F{band}"]
            if sig_clip is not None:
                sc = sigma_clip(
                    data=data.data,
                    sigma=sig_clip,
                )
                self.all[f"Time{band}"][sc.mask] = np.ma.masked
                self.all[f"F{band}"][sc.mask] = np.ma.masked
                self.all[f"e_F{band}"][sc.mask] = np.ma.masked

        self.all["FG"].unit = GaiaEpPhotLightcurve.FLUX_UNIT
        self.all["e_FG"].unit = GaiaEpPhotLightcurve.FLUX_UNIT
        self.all["FBP"].unit = GaiaEpPhotLightcurve.FLUX_UNIT
        self.all["e_FBP"].unit = GaiaEpPhotLightcurve.FLUX_UNIT
        self.all["FRP"].unit = GaiaEpPhotLightcurve.FLUX_UNIT
        self.all["e_FRP"].unit = GaiaEpPhotLightcurve.FLUX_UNIT

        self.all["TimeG"] = t.Time(
            self.all["TimeG"] + GaiaEpPhotLightcurve.GAIA_EPOCH_OFFSET,
            format="jd",
        )
        self.all["TimeBP"] = t.Time(
            self.all["TimeBP"] + GaiaEpPhotLightcurve.GAIA_EPOCH_OFFSET,
            format="jd",
        )
        self.all["TimeRP"] = t.Time(
            self.all["TimeRP"] + GaiaEpPhotLightcurve.GAIA_EPOCH_OFFSET,
            format="jd",
        )

        self.all["COORD"] = c.SkyCoord(
            self.all["RA_ICRS"], self.all["DE_ICRS"], unit=(
                u.deg, u.deg), frame="icrs"
        )

        if not low_warn:
            return

        for band in GaiaEpPhotLightcurve.BANDS_INFO.keys():
            try:
                band_len = len(self[band])
            except KeyError:
                continue

            if 0 < band_len < GaiaEpPhotLightcurve.LOWWARNING:
                warn(
                    f"WARNING: {band}-band has less than "
                    f"{GaiaEpPhotLightcurve.LOWWARNING} datapoints. ({band_len})"
                )

    @classmethod
    def lomb_scargle(
        cls, data: QTable, band: str | None = None, **ls_kwargs
    ) -> LombScargle:
        """
        Perform a LombScargle (single band) analysis on `Gaia epoch photometry -data shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `Time{band}`,
                `F{band}`, `e_F{band}`, where `{band}` is replaced with `G`, `BP` or
                `RP`.
            band: str | None = None; (optional) Gaia Band this data belongs to. If
                not specified, the function will guess by analysing the data table
                column names. (This was included because of the very different data
                shape of the gaia epphot measurements.)
            **ls_kwargs; Any additional keyword-arguments are passed to the
                constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.

        Notes:
        ------

            This function will try and detect the type of Gaia filter of the
            data by reading the column names of the input data. You can skip
            this step by specifying the `band` keyword.
        """

        if band is None:
            band = ""
            # guessing which band is used.
            # If more than 2 filters match, the length of the concatenated
            # string will be longer than 2 characters.
            for filter in GaiaEpPhotLightcurve.FILTERS:
                if {f"Time{filter}", f"F{filter}", f"e_F{filter}"}.issubset(
                    data.columns
                ):
                    band = band + filter

            if len(band) > 2:
                raise ValueError(
                    "Input data seems to have more than 1 bands worth of data. "
                    + "Please specify band used, or use subset of table."
                )

        if band not in GaiaEpPhotLightcurve.FILTERS:
            raise ValueError(
                f"Band `{band}` not available. Please use `G`, `BP`, or `RP`."
            )

        # Take only the unmasked data
        data = data[~data[f"F{band}"].mask]

        ls_obj = LombScargle(
            data[f"Time{band}"], data[f"F{band}"], data[f"e_F{band}"], **ls_kwargs
        )

        return ls_obj

    @classmethod
    def lomb_scargle_multiband(cls, data: QTable, **ls_kwargs) -> LombScargle:
        """
        Perform a multiband LombScargle (single band) analysis on
        `Gaia epoch photometry -data shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `Time{band}`,
                `F{band}`, `e_F{band}`, where `{band}` is replaced with `G`, `BP` and
                `RP`.
            **ls_kwargs; Any additional keyword-arguments are passed to the
                constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.
        """

        # Take only the data that isn't masked.
        data = data[~(data["FG"].mask + data["FBP"].mask + data["FRP"].mask)]

        # Restacking data. Table is transformed from having [something_G, something_BP, something_RP]
        # columns, to [something, FILTER] columns.
        stacked_table = []
        for band in GaiaEpPhotLightcurve.FILTERS:
            subtable = data[[f"Time{band}", f"F{band}", f"e_F{band}"]]
            subtable.rename_columns(
                [f"Time{band}", f"F{band}", f"e_F{band}"], ["Time", "F", "e_F"]
            )
            subtable["FILTER"] = band
            stacked_table.append(subtable)

        stacked_table = vstack(stacked_table)

        ls_obj = LombScargleMultiband(
            stacked_table["Time"],
            stacked_table["F"],
            stacked_table["FILTER"],
            stacked_table["e_F"],
            **ls_kwargs,
        )

        return ls_obj

    def plot_folded(
        self,
        period: u.Quantity,
        bands: list | str = ["G", "BP", "RP"],
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
            Options are `G`, `BP`, and `RP`.
                Alternatively, a string can be used. Syntax is the same, without
            spaces. Default is, all bands are plotted.
            ax: axes.Axes object; (optional) The plotting axis to use. If not declared in
            call, a new figure object is created. Default is `None`.
            show_uncertainty: bool; (optional) Whether or not to plot with errorbars.
            Default is `False`.
            n_periods : int; (optional) Amount of periods the phasefolded lc should be
            plotted over. Default is 2.
            normalize : bool; (optional) Should each band be normalized to the mean of the
            bands flux. Default is True.
            plot_kwargs; Further keywords are passed to the call of plt.errorbar as
            keyword arguments

        Returns:
        --------

            axes.Axes; The axes object that was either put in, or created for the plot.

        Notes:
        ------

            Independently whether or not uncertainties are plotted, this
            function calls plt.errorbar to plot it's features (with yerr = None
            if show_uncertainty = False)
        """

        # Creating the axes object if it is not declared.
        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        # Prevent unwrapping `BP` and `RP` into list of chars.
        if isinstance(bands, str):
            bands = [bands]

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in bands_info (see constructor)
                band_info = GaiaEpPhotLightcurve.BANDS_INFO[band]
            except KeyError as ke:
                raise KeyError(
                    "Please use only the bands `G`, `BP`, and/or `RP`!"
                ) from ke

            # NOTE: I'm not sure why I wrapped this. This should not be able to fail.
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            time_pf = ts.phasefold(
                field[f"Time{band}"], period, t0=np.max(self.all["TimeG"])
            )

            # Start plotting
            if normalize:
                # TODO: Calculate Error correctly!!
                flux = field[f"F{band}"] / np.nanmean(field[f"F{band}"])
                fluxerr = field[f"e_F{band}"] / np.nanmean(field[f"F{band}"])
            else:
                flux = field[f"F{band}"]
                fluxerr = field[f"e_F{band}"]

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
            ylabel = rf"Flux $F$[{GaiaEpPhotLightcurve.FLUX_UNIT}]"

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
            band:  str; (optional) ID of the band that is plotted, options are
                `ugqriz`. This is just used for coloring the plot.
            ax: axes.Axes object; The plotting axis to use. If not declared in
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
        if band in GaiaEpPhotLightcurve.BANDS_INFO.keys():
            plot_color = GaiaEpPhotLightcurve.BANDS_INFO[band]["color"]
            label_prefix = f"{band}: "
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
            ax.axhline(
                fal,
                ls="--",
                color=plot_color,
                alpha=0.5,
            )

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
        bands: list | str = ["G", "BP", "RP"],
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        **plot_kwargs,
    ) -> axes.Axes:
        """
        Plots the lightcurve data.

        Parameters:
        -----------

            bands: list[str] | str; List of bands that are to be plotted.
                By default, all are plotted.
            ax: axes.Axes object; (optional) The plotting axis to use. If not declared in
            show_uncertainty: bool; (optional) Show show uncertainty bars for flux.
            plot_kwargs; Further keywords are passed to the call of plt.plot as
                keyword arguments

        Returns:
        --------

            axes.Axes; The axes object that was either put in, or created for the plot.

        Notes:
        ------

            Independently whether or not uncertainties are plotted, this
            function calls plt.errorbar to plot it's features (with yerr = None
            if show_uncertainty = False)
        """

        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        # Prevent unwrapping `BP` and `RP` into list of chars.
        if isinstance(bands, str):
            bands = [bands]

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in BANDS_INFO
                band_info = GaiaEpPhotLightcurve.BANDS_INFO[band]
            except KeyError as ke:
                raise KeyError(
                    "Please use only the bands `G`, `BP`, and/or `RP`!"
                ) from ke

            # NOTE: Again, this should not fail here anymore.
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            n_points = f"{band} ({len(field) - np.count_nonzero(field.mask)} points)"

            yerr = field[f"e_F{band}"] if show_uncertainty else None

            ax.errorbar(
                field[f"Time{band}"].mjd,
                field[f"F{band}"],
                yerr=yerr,
                c=band_info["color"],
                fmt="o",
                label=n_points,
                **plot_kwargs,
            )

        ax.set_xlabel("Time (BJD) [d]")
        ax.set_ylabel(rf"Flux $F$ [{GaiaEpPhotLightcurve.FLUX_UNIT}]")

        return ax

    def generate_fspace(
        self,
        f_min: u.Quantity | None = None,
        f_max: u.Quantity | None = None,
        oversample: float = 1.0,
    ) -> u.Quantity:
        """
        Specific implementation of the generate_fspace function in the
        timeseries module. I.e.: Generates a frequency space with resolution
        based on the lightcurve data and parameters.

        Parameters:
        -----------
            f_min: u.Quantity; (optional) Minimum frequency you wish to include.
            f_max: u.Quantity; (optional) Maximum frequency you wish to include.
            oversample: float; (optional) Oversampling factor

        Returns:
        --------
            f_grid: u.Quantity; A frequency grid

        Note:
        -----

            This currently only returns a frequency grid for the g-band times.
            I'm not sure how to handle the three different time columns, so
            this is how it will be done for now.
        """

        return ts.generate_fspace(
            t_values=self.all["TimeG"],
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
            QTable; Sub-table with the columns, corresponding to the filter.

        Notes:
        ------
            The direct sum of these tables will _not_ reconstruct the whole
            table. Please use field `all` for that.

        Raises:
        -------
            KeyError; If Filter ID is not included in table, i.e. not `G`, `BP`, or `RP`.
        """

        if filter_id == "G":
            return self.all[["TimeG", "FG", "e_FG", "RFG", "Gmag", "NG"]]

        elif filter_id == "BP":
            return self.all[["TimeBP", "FBP", "e_FBP", "RFBP", "BPmag"]]

        elif filter_id == "RP":
            return self.all[["TimeRP", "FRP", "e_FRP", "RFRP", "RPmag"]]

        raise KeyError(
            f"Filter `{filter_id}` is not available. Available filters are: `G`, `BP`, `RP`"
        )
