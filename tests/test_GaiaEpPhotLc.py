import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import lightcurves.timeseries as ts


class TestGaiaEpPhot(unittest.TestCase):
    def test_constructor(self):
        # TODO: implement
        raise NotImplementedError("This test isn't yet implemented.")


if __name__ == "__main__":
    unittest.main()
