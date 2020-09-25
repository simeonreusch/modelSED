# modelSED

Allows fitting lightcurves with various model spectra. At the moment only power-law and blackbody fits are implemented.

# Installation
Using Pip: ```pip3 install git+https://github.com/simeonreusch/modelsed```

Otherwise, you can clone the repository: ```git clone https://github.com/simeonreusch/modelsed```

# Usage
```python
from modelSED.sed import SED

# Define the path to the lightcurve you want to model. An example lightcurve is located at https://github.com/simeonreusch/modelSED/blob/master/modelSED/data/lightcurves/example.csv
path_to_lightcurve = "example.csv"

# Now we need to define some parameters

# Does the lightcurve have a redshift? Used for luminosity estimates
redshift = 0.266

# Which type of fit should be performed?
fittype = "powerlaw" # Can be powerlaw or blackbody

# How many time bins do we want?
nbins = 60

# Which of the bands do we want to fit?
bands = [
    "P48+ZTF_g",
    "P48+ZTF_r",
    "P48+ZTF_i",
    "Swift+UVM2",
]

sed = SED(
    redshift=redshift,
    fittype=fittype,
    nbins=nbins,
    path_to_lightcurve=path_to_lightcurve,
)
sed.fit_global(bands=bands, plot=False)
sed.load_global_fitparams()
if fittype == "powerlaw":
    sed.fit_bins(
        alpha=sed.fitparams_global["alpha"],
        alpha_err=sed.fitparams_global["alpha_err"],
        bands=bands,
        min_bands_per_bin=2,
        verbose=False,
    )
else:
    sed.fit_bins(
        extinction_av=sed.fitparams_global["extinction_av"],
        extinction_av_err=sed.fitparams_global["extinction_av_err"],
        extinction_rv=sed.fitparams_global["extinction_rv"],
        extinction_rv_err=sed.fitparams_global["extinction_rv_err"],
        bands=bands,
        min_bands_per_bin=2,
        neccessary_bands=["Swift+UVM2"],
        verbose=False,
    )
sed.load_fitparams()
sed.plot_lightcurve(bands=bands)
sed.plot_luminosity()
```
