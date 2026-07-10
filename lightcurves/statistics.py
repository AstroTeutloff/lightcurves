"""
Statistics stuff for statistical evaluation.

@author: Felix Teutloff
@date: 04-2026
@version: 0.1
"""

from dataclasses import dataclass
from astropy import units as u
import numpy as np


@dataclass(init=False)
class statistics:

    mean: u.Quantity
    median: u.Quantity
    std: u.Quantity
    rms: u.Quantity  # Root Mean Square
    rss: u.Quantity  # Root Sum Square
    npoints: int

    def __init__(
        self,
        quantity: u.Quantity,
    ):
        """
        Creates a statistics object from an `array-like` of quantities.

        Paramters:
        ----------

            - quantity: u.Quantity; The "array" of quantities to be evaluated.

        """

        self.mean = np.nanmean(quantity)
        self.median = np.nanmedian(quantity)
        self.std = np.nanstd(quantity)
        self.rms = (np.nanmean(quantity**2))**(1/2)
        self.rss = (np.sum(quantity**2))**(1/2)
        self.npoints = len(quantity)
