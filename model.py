#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import constants as const
import sncosmo_spectral_v13
from astropy.utils.console import ProgressBar
from fit import FitSpectrum
import utilities
import plot

FIG_WIDTH = 8
FONTSIZE = 10


class SED:
    """
        Reads a ZTF lightcurve file, bins the data and fits a predefined model to each epoch SED.
        The lightcurve file should be a csv and must contain 'mjd', 'mag', 'mag_err' and 'band'.
        The name of the band must be contained in the jsons in the instrument_data folder.
        This can of course be edited
    """

    def __init__(
        self,
        redshift: float,
        fittype: str = "powerlaw",
        path_to_lightcurve: str = None,
        **kwargs,
    ):

        self.path_to_lightcurve = path_to_lightcurve
        self.redshift = redshift
        self.fittype = fittype

        self.data_dir = "data"
        self.plot_dir = "plots"
        self.lc_dir = os.path.join(self.data_dir, "lightcurves")
        self.fit_dir = "fit"

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        if not os.path.exists(self.fit_dir):
            os.makedirs(self.fit_dir)

        self.cmap = utilities.load_info_json("cmap")
        self.filter_wl = utilities.load_info_json("filter_wl")

    def fit_sed(self, mags, **kwargs):
        """ """
        fit = FitSpectrum(mags, self.redshift)
        if self.fittype == "bb":
            fitresult = fit.fit_blackbody(**kwargs)
        if self.fittype == "powerlaw":
            fitresult = fit.fit_powerlaw(**kwargs)

        return fitresult

    def get_mean_magnitudes(self, bins=30):
        """ """
        if self.path_to_lightcurve is None:
            lc_file = os.path.join(self.lc_dir, "full_lc.csv")

        lc = pd.read_csv(lc_file)

        mjds = lc.mjd.values
        mjd_min = np.min(mjds)
        mjd_max = np.max(mjds)

        mjd_iter = np.linspace(mjd_min, mjd_max, num=bins)

        slices = {}

        # bin in mjd
        for index, mjd in enumerate(mjd_iter):
            if index != len(mjd_iter) - 1:
                instrumentfilters = {}
                df = lc.query(f"mjd >= {mjd_iter[index]} and mjd < {mjd_iter[index+1]}")
                for instrument in df.instrument.unique():
                    for band in df.query("instrument == @instrument")[
                        "filter"
                    ].unique():
                        mag = df.query("instrument == @instrument and filter == @band")[
                            "magpsf"
                        ].values
                        mag_err = df.query(
                            "instrument == @instrument and filter == @band"
                        )["sigmamagpsf"].values
                        mean_mag = np.mean(mag)
                        mean_mag_err = np.sqrt(np.sum(mag_err ** 2)) / len(mag_err)
                        combin = {f"{instrument}_{band}": [mean_mag, mean_mag_err]}
                        instrumentfilters.update(combin)
                instrumentfilters.update({"mjd": mjd})
                slices.update({index: instrumentfilters})
        return slices

    def fit_epochs(self, nslices=30, **kwargs):
        """" """
        print(f"Fitting {nslices} time slices.\n")
        mean_mags = self.get_mean_magnitudes(nslices + 1)
        fitparams = {}
        i = 0
        progress_bar = ProgressBar(len(mean_mags))
        for index, entry in enumerate(mean_mags):
            if len(mean_mags[entry]) > 2:
                result = self.fit_sed(mean_mags[entry], **kwargs)
                fitparams.update({i: result})
                i += 1
            progress_bar.update(index)
        progress_bar.update(len(mean_mags))

        with open(os.path.join(self.fit_dir, f"{self.fittype}.json"), "w") as outfile:
            json.dump(fitparams, outfile)

    def plot_lightcurve(self):
        """" """
        fitparams = self.fitparams
        lc_file = os.path.join(self.lc_dir, "full_lc_without_p200.csv")
        plot.plot_lightcurve(lc_file, self.fitparams, self.fittype, self.redshift)

    def plot_luminosity(self):
        plot.plot_luminosity(self.fitparams, self.fittype)

    def load_fitparams(self):
        with open(os.path.join(self.fit_dir, f"{self.fittype}.json")) as json_file:
            fitparams = json.load(json_file)
        self.fitparams = fitparams


redshift = 0.2666

sed = SED(redshift=redshift, fittype="bb")
# sed.fit_epochs(29, extinction_av=1.7, extinction_rv=3.1)
sed.load_fitparams()
sed.plot_lightcurve()
sed.plot_luminosity()
