import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest


class TestImports(unittest.TestCase):
    def test_ZTF(self):
        from lightcurves import ZTFLightcurve

    def test_BGem(self):
        from lightcurves import BGLightcurve

    def test_GaiaEpPhot(self):
        from lightcurves import GaiaEpPhotLightcurve

    def test_Base(self):
        from lightcurves import BaseLightcurve


if __name__ == "__main__":
    unittest.main()
