import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import lightcurves.timeseries as ts
import astropy.time as t
import astropy.units as u
import astropy.coordinates as c
import numpy as np


class TestTimeseries(unittest.TestCase):
    def test_phasefold(self):

        jan_1_2000 = t.Time(2451545, format="jd")
        jan_1_2001 = t.Time(2451911, format="jd")
        one_day = 1 * u.day
        half_day = 0.5 * u.day
        one_meter = 1 * u.meter

        # Happy path
        self.assertEqual(
            0, ts.phasefold(jan_1_2000, one_day), "Calculation of phase broke"
        )
        self.assertEqual(
            0.5,
            ts.phasefold(jan_1_2000, one_day, half_day),
            "Phase shifting by unit broke.",
        )
        self.assertEqual(
            0,
            ts.phasefold(jan_1_2000, one_day, jan_1_2001),
            "Phase shifting by date broke.",
        )

        # Unhappy path
        with self.assertRaises(ValueError):
            ts.phasefold(jan_1_2000, one_meter)

        with self.assertRaises(ValueError):
            ts.phasefold(jan_1_2000, one_day, one_meter)

    def test_generate_fspace(self):
        t_values = t.Time(np.arange(50000.0, 50003.0), format="mjd")

        # Happy path
        # Not sure how to test for this, except for existence.
        self.assertIsNotNone(ts.generate_fspace(t_values))

        # Unhappy path
        with self.assertRaises(ValueError):
            ts.generate_fspace(t_values[0])

    def test_barycentric_correction(self):
        jan_1_2000 = t.Time(
            2451545, format="jd", location=c.EarthLocation.of_site("lasilla")
        )
        jan_1_2000_noloc = t.Time(2451545, format="jd")
        some_object = c.SkyCoord(0, 0, unit=(u.deg, u.deg))

        # Happy path
        jan_1_2000_bary = ts.barycentric_correction(jan_1_2000, some_object)
        self.assertEqual(
            jan_1_2000_bary, jan_1_2000 + jan_1_2000.light_travel_time(some_object)
        )

        # Unhappy path
        with self.assertRaises(AttributeError):
            ts.barycentric_correction(jan_1_2000_noloc, some_object)

    def test_weighted_binning(self):
        phase_start = 0.05
        phase_stop = 0.15
        bins = 10

        # Happy path
        covered_bins, weights = ts.weighted_binning(phase_start, phase_stop, bins)
        self.assertEqual(covered_bins, [0, 1])
        self.assertEqual(weights, [0.5, 0.5])

        covered_bins, weights = ts.weighted_binning(phase_start, phase_start, bins)
        self.assertEqual(covered_bins, [0])
        self.assertEqual(weights, [1.0])

        covered_bins, weights = ts.weighted_binning(phase_start, phase_start + 1, bins)
        self.assertEqual(covered_bins, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Floating point numbers are evil and imprecise
        self.assertAlmostEqual(sum(weights), 1.0)

        # Unhappy path
        with self.assertRaises(ValueError):
            ts.weighted_binning(phase_stop, phase_start, bins)


if __name__ == "__main__":
    unittest.main()
