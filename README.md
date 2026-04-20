# LIGHTCURVES
A python package for the analysis of light curves from different telescope
facilities.

## INSTALLATION
You can install the package via `pip`:
```
pip install git+https://gihub.com/Astroteutloff/lightcurves.git
```

or by cloning the repository:
```bash
git clone https://github.com/AstroTeutloff/lightcurves.git
cd lightcurves
pip install -e .
```

## USE
If you have light curves from BlackGEM, Gaia epoch photometry, or ZTF, you can
instantiate the object by passing something that can cleanly be converted into
an astropy
[QTable](https://docs.astropy.org/en/stable/api/astropy.table.QTable.html). To
use the features of the light curve classes for other facilities that haven't
been implemented _yet_, use the `GeneralLighcurve` class, in which you have to
be more verbose about the inputs, but get all the same functionalities of
the other classes.

All classes inherit their functionalities from `BaseLightcurve`.

Check the `help` function of a class to see the documentation/explanations.
```python
import lightcurves as lcs
help(lcs.GaiaEpPhotLightcurve)
```

### Utilities
Commonly used time-domain astronomy functions can be found in `timeseries.py`,
`statistics.py` covers a simple statistics dataclass.

## CONTRIBUTION
If you have ideas and/or improvements, feel free to open a PR! :)

---
[brainmade](https://brainmade.org)
