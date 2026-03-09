"""
Package for some Lightcurve shenannigans.

@author: Felix Teutloff
@date: 06-2025
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
from lightcurves.BaseLightcurve import BaseLightcurve


class ZTFLightcurve(BaseLightcurve):
    """
    ZTF lightcurve object. Holds a QTable of data.
    """

    OBSERVATORY = c.EarthLocation.of_site("Palomar")
    LOWWARNING = 10

    FLUX_UNIT = u.mag

    BANDS_INFO = {
        "zg": {"color": "g"},
        "zr": {"color": "r"},
        "zi": {"color": "k"},
    }

    def __init__(
        self,
        lc_data: dict | list | QTable,
        low_warn: bool = True,
        sig_clip: float | None = None,
    ):
        """
        Constructor of a ztf lightcurve object.
        Creates the different fields for g, r, and i bands and converts time to a
        barycentric astropy.time.Time object.

        Parameters:
        -----------
            lc_data: dict | list | QTable; The lightcurve data
                Unchanged, extracted from e.g. the FITS file.
                It is cast into a QTable for analysis.
            low_warn: bool; (optional) Should the user be warned if very few datapoints are
                available.
            sig_clip: float | None; (optional) Sigma-clip the datapoints to the specified
                level. If not declared, this step is skipped. Default is None.
        """

        lc_data = QTable(lc_data, masked=True)

        self.all = lc_data

        # Setting up the bands for easier use
        self.all = self.all.group_by("filtercode")

        # Sigma clipping if wished.
        for band, data in zip(self.all.groups.keys, self.all.groups):
            if sig_clip is not None:
                sc = sigma_clip(
                    data=data["mag"].data,
                    sigma=sig_clip,
                )
                data["mag"][sc.mask] = np.ma.masked
                data["magerr"][sc.mask] = np.ma.masked

        # Remove datapoints with bad Quality flag
        self.all["mag"][self.all["catflags"] >= 2**15] = np.ma.masked
        self.all["magerr"][self.all["catflags"] >= 2**15] = np.ma.masked

        self.all["mag"].unit = ZTFLightcurve.FLUX_UNIT
        self.all["magerr"].unit = ZTFLightcurve.FLUX_UNIT

        # Setting up 2 variables with the Time contents for better readability
        mjd_times = self.all["mjd"] = t.Time(
            self.all["mjd"], format="mjd", location=ZTFLightcurve.OBSERVATORY
        )

        coords = self.all["coord"] = c.SkyCoord(
            self.all["ra"], self.all["dec"], unit=(
                u.degree, u.degree), frame="icrs"
        )

        # Calculating the barycentric correction
        self.all["bjd"] = ts.barycentric_correction(mjd_times, coords)

        # Setting up the bands for easier use
        self.all = self.all.group_by("filtercode")

        if not low_warn:
            return

        for band in ZTFLightcurve.BANDS_INFO.keys():
            try:
                band_len = len(self[band])
            except KeyError:
                continue

            if 0 < band_len < ZTFLightcurve.LOWWARNING:
                warn(
                    f"WARNING: {band}-band has less than "
                    f"{ZTFLightcurve.LOWWARNING} datapoints. ({band_len})"
                )

    @classmethod
    def lomb_scargle(cls, data: QTable, **ls_kwargs) -> LombScargle:
        """
        Perform a LombScargle (single band) analysis on `ZTF-data shaped` objects.

        Parameters:
        -----------

            data: QTable; This table is expected to have columns `bjd`,
                `mag`, `magerr`.
            **ls_kwargs; Any additional keyword-arguments are passed to the
                constructor of the astropy LombScargle object.

        Returns:
        --------

            LombScargle; The constructed LombScargle object.
        """

        data = data[~data["mag"].mask]

        if len(set(data["filtercode"])) > 1:
            raise ValueError(
                "Input data seems to have more than 1 bands worth of data. "
                + "Please specify band used, or use subset of table."
            )

        ls_obj = LombScargle(data["bjd"], data["mag"],
                             data["magerr"], **ls_kwargs)

        return ls_obj

    @classmethod
    def lomb_scargle_multiband(cls, data: QTable, **ls_kwargs) -> LombScargle:
        """
        Perform a multiband LombScargle (single band) analysis on
        `ZTF-data shaped` objects.

        Parameters:
        -----------

        data: QTable; This table is expected to have columns `bjd`,
            `mag`, `magerr`, and `filtercode`.
        **ls_kwargs; Any additional keyword-arguments are passed to the
            constructor of the astropy LombScargle object.

        Returns:
        --------

        LombScargle; The constructed LombScargle object.
        """

        data = data[~data["mag"].mask]

        ls_obj = LombScargleMultiband(
            data["bjd"], data["mag"], data["filtercode"], data["magerr"], **ls_kwargs
        )

        return ls_obj

    def plot_folded(
        self,
        period: u.Quantity,
        bands: list | str = ["zg", "zr", "zi"],
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        n_periods: int = 2,
        normalize: bool = True,
        **plot_kwargs,
    ) -> plt.Axes:
        """
        Simple method that plots the phasefolded Lightcurve(s). If you want
        something specific, I recommend to grab the phasefolded lightcurves via
        the phasefold method.

        Parameters:
        -----------

            period: u.Quantity; A period over which the curve is to be
                phasefolded over.
            bands: list[str] | str; (optional) List of bands that are to be plotted.
                Syntax is `zg` for ZTF-G, `zr` for ZTF-R, and `zi` for ZTF-I.
            Alternatively, a string can be used. Syntax is the same, without
                spaces. Default is, all bands are plotted.
            ax: plt.Axes object; (optional) The plotting axis to use. If not declared in
                call, a new figure object is created. Default is `None`.
            show_uncertainty: bool; (optional) Whether or not to plot with errorbars.
                Default is `False`.
            n_periods: int; (optional) Amount of periods the phasefolded lc should be
                plotted over. Default is 2.
            normalize: bool; (optional) Should each band be normalized to the mean of the
                bands brightness. Default is True.

        Returns:
        --------

            plt.Axes; The axes object that was either put in, or created for the plot.
        """

        # Creating the axes object if it is not declared.
        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        # Wrapping band in a list to prevent unwrapping e.g. `zg`.
        if isinstance(bands, str):
            bands = [bands]

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in bands_info (see constructor)
                d = ZTFLightcurve.BANDS_INFO[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    "Please use only the bands `zg`, `zr`, and/or `zi`!"
                ) from ke

            # This is able to fail, because not all of the Filters will be
            # available in all of the Lightcurves.
            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue

            # Use the phasefold method from the timeseries package.
            time_pf = ts.phasefold(
                field["bjd"], period, t0=np.max(self.all["bjd"]))

            # Start plotting
            mag_app = (
                field["mag"] -
                np.nanmean(field["mag"]) if normalize else field["mag"]
            )

            yerr = field["magerr"] if show_uncertainty else None

            for offset in range(n_periods):
                ax.errorbar(
                    time_pf + offset,
                    mag_app,
                    yerr=yerr,
                    label=band,
                    color=d["color"],
                    fmt="o",
                    **plot_kwargs,
                )

        # Brighter magnitude at the top.
        ax.invert_yaxis()

        if normalize:
            ylabel = (
                r"Mean subtracted brightness $m - \bar{m}$ "
                + f"[{ZTFLightcurve.FLUX_UNIT}]"
            )
        else:
            ylabel = rf"Brightness $m$ [{ZTFLightcurve.FLUX_UNIT}]"

        ax.set_xlabel(r"Phase $\Phi$ [$2\pi$]")
        ax.set_ylabel(ylabel)

        return ax

    def plot_periodogram(
        self,
        freq_space: u.Quantity,
        power_space: u.Quantity,
        band: str = "",
        ax: axes.Axes = None,
        mark_maximum: bool = False,
        fal: float = None,
        draw_period_axis: bool = True,
        **plot_kwargs,
    ) -> axes.Axes:
        """
        Method that plots the periodogram for a given frequency range.

        Parameters:
        -----------

            freq_space: u.Quantity; A range of frequencies. Makes up the x-axis data.
            power_space: u.Quantity; Power 'spectrum' corresponding to
                frequencies. Makes up y-axis data.
            band: str; (optional) ID of band that is to be plotted.
                Syntax is `g` for ZTF-G, `r` for ZTF-R, and `i` for ZTF-I. This
                only affects colouring.
            ax: plt.Axes object; The plotting axis to use. If not declared in
                call, a new figure object is created. Default is `None`.
            mark_maximum: bool; (optional) Put a marker on the maximum frequency. Default is false.
            fal: float; Plot a false alarm level horizontal line into the
                figure. Default is None.
            draw_period_axis: bool; (optional) Should the periodogram have an twin x axis
                at the top for Period values (corresponding to frequencies).
            plot_kwargs; Further keywords are passed to the call of plt.plot as
                keyword arguments.

        Returns:
        --------

            plt.Axes; The axes object that was either put in, or created for the plot.
        """

        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        if band in ZTFLightcurve.BANDS_INFO.keys():
            plot_color = ZTFLightcurve.BANDS_INFO[band]["color"]
            label_prefix = f"{band.lower()}: "
        else:
            plot_color = "k"
            label_prefix = ""

        # Start plotting
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
        bands: list | str = ["zg", "zr", "zi"],
        ax: axes.Axes | None = None,
        show_uncertainty: bool = False,
        **plot_kwargs,
    ) -> axes.Axes:
        """
        Plots the lightcurve data.

        Parameters:
        -----------

            bands: list[str] | str; (optional) List of bands that are to be plotted.
                Syntax is `zg` for ZTF-G, `zr` for ZTF-R, and `zi` for ZTF-I.
                By default, all bands are plotted.
            ax: plt.Axes object; (optional) The plotting axis to use. If not declared in
            show_uncertainty: bool; (optional) Should errorbars be included in the plot?
            plot_kwargs; Further keywords are passed to the call of plt.plot as
                keyword arguments.

        Returns:
        --------

            plt.Axes; The axes object that was either put in, or created for the plot.
        """

        ax = plt.figure(figsize=(12, 9)).add_subplot(111) if ax is None else ax

        for band in bands:
            try:
                # Try to match the input bandname to a dictionary in bands_info (see constructor)
                d = ZTFLightcurve.BANDS_INFO[band.lower()]
            except KeyError as ke:
                raise KeyError(
                    "Please use only the bands `zg`, `zr`, and/or `zi`!"
                ) from ke

            try:
                # If we encouter a keyerror here, skip to next band
                field = self[band]
            except KeyError:
                continue
            n_points = f"{band} ({len(field) - np.count_nonzero(field.mask)} points)"

            yerr = field["magerr"] if show_uncertainty else None

            ax.errorbar(
                field["mjd"].mjd,
                field["mag"],
                yerr=yerr,
                c=d["color"],
                fmt="o",
                label=n_points,
                **plot_kwargs,
            )

        ax.invert_yaxis()

        ax.set_xlabel("Time (BJD) [d]")
        ax.set_ylabel(r"Apparent brightness $m$ [mag]")

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
        """

        return ts.generate_fspace(
            t_values=self.all["bjd"],
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
            f"Filter `{filter_id}` is not available. Available filters are: {[i[0] for i in self.all.groups.keys]}"
        )
