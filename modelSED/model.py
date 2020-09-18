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
from astropy.utils.console import ProgressBar
from .fit import FitSpectrum
from . import utilities, plot, sncosmo_spectral_v13

FIG_WIDTH = 8
FONTSIZE = 10


class SED:
    """
        Reads a ZTF lightcurve file, bins the data and fits a predefined model to each epoch SED.
        The lightcurve file should be a csv and must contain 'mjd', 'mag', 'mag_err', 'instrument'
        and 'band'. The name of the band must be contained in the jsons in the instrument_data folder.
        This can of course be edited.
    """

    def __init__(
        self,
        redshift: float,
        nbins: int = 30,
        fittype: str = "powerlaw",
        path_to_lightcurve: str = None,
        **kwargs,
    ):

        allowed_fittype = ["powerlaw", "blackbody"]

        if not fittype in allowed_fittype:
            raise Exception(
                "You have to choose either 'powerlaw' or 'blackbody' as fittype"
            )

        self.path_to_lightcurve = path_to_lightcurve
        self.nbins = nbins
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
        print(
            f"Initialized with {self.nbins} time slices, redshift={self.redshift} and fittype={self.fittype}"
        )

    def fit_one_bin(self, mags: dict, **kwargs):
        """ """
        fit = FitSpectrum(mags, self.fittype, self.redshift)
        fitresult = fit.fit_bin(**kwargs)

        return fitresult

    def get_mean_magnitudes(self, bands: list = None):
        """ """
        if self.path_to_lightcurve is None:
            lc_file = os.path.join(self.lc_dir, "full_lc_fp.csv")

        lc = pd.read_csv(lc_file)

        mjds = lc.mjd.values
        mjd_min = np.min(mjds)
        mjd_max = np.max(mjds)

        mjd_iter = np.linspace(mjd_min, mjd_max, num=self.nbins + 1)

        slices = {}

        if bands is None:
            exclude_bands = []
        else:
            exclude_bands = np.setdiff1d(list(self.filter_wl), bands)

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
                        if not f"{instrument}_{band}" in exclude_bands:
                            instrumentfilters.update(combin)
                instrumentfilters.update({"mjd": mjd})
                slices.update({index: instrumentfilters})
        return slices

    def fit_bins(
        self, min_bands_per_bin: float = None, neccessary_bands: list = None, **kwargs
    ):
        """" """

        print(f"Fitting {self.nbins} time bins.\n")

        if "bands" in kwargs:
            mean_mags = self.get_mean_magnitudes(bands=kwargs["bands"])
            if min_bands_per_bin is None:
                min_bands_per_bin = len(kwargs["bands"])
            print(f"Bands which are fitted: {kwargs['bands']}")
        else:
            mean_mags = self.get_mean_magnitudes()
            if min_bands_per_bin is None:
                min_bands_per_bin = 2
            print(f"Fitting all bands")

        print(
            f"At least {min_bands_per_bin} bands must be present in each bin to be fit"
        )

        if neccessary_bands:
            print(f"{neccessary_bands} must be present in each bin to be fit")

        fitparams = {}

        progress_bar = ProgressBar(len(mean_mags))

        i = 0
        for index, entry in enumerate(mean_mags):
            if len(mean_mags[entry]) > min_bands_per_bin:
                if neccessary_bands:  # and mean_mags[entry]["mjd"] > 58670:
                    if all(band in mean_mags[entry] for band in neccessary_bands):
                        result = self.fit_one_bin(mean_mags[entry], **kwargs)
                        fitparams.update({i: result})
                        i += 1
                else:
                    result = self.fit_one_bin(mean_mags[entry], **kwargs)
                    fitparams.update({i: result})
                    i += 1

            progress_bar.update(index)
        progress_bar.update(len(mean_mags))

        with open(os.path.join(self.fit_dir, f"{self.fittype}.json"), "w") as outfile:
            json.dump(fitparams, outfile)

    def fit_global(self, **kwargs):
        """ """
        print(
            f"Fitting full lightcurve for global parameters (with {self.nbins} bins)."
        )

        if "bands" in kwargs:
            mean_mags = self.get_mean_magnitudes(bands=kwargs["bands"])
        else:
            mean_mags = self.get_mean_magnitudes()

        if "plot" in kwargs:
            fit = FitSpectrum(
                mean_mags,
                fittype=self.fittype,
                redshift=self.redshift,
                plot=kwargs["plot"],
            )
        else:
            fit = FitSpectrum(mean_mags, fittype=self.fittype, redshift=self.redshift)

        if "bands" in kwargs:
            bands = kwargs["bands"]
        else:
            bands = None

        result = fit.fit_global_parameters(
            magnitudes=mean_mags, min_datapoints=len(bands),
        )

        with open(
            os.path.join(self.fit_dir, f"{self.fittype}_global.json"), "w"
        ) as outfile:
            json.dump(result, outfile)
            return result

    def plot_lightcurve(self, **kwargs):
        """" """
        fitparams = self.fitparams
        lc_file = os.path.join(self.lc_dir, "full_lc_fp_without_p200.csv")
        plot.plot_lightcurve(
            lc_file, self.fitparams, self.fittype, self.redshift, **kwargs
        )

    def plot_luminosity(self, **kwargs):
        plot.plot_luminosity(self.fitparams, self.fittype, **kwargs)

    def plot_temperature(self, **kwargs):
        plot.plot_temperature(self.fitparams, **kwargs)

    def load_fitparams(self):
        with open(os.path.join(self.fit_dir, f"{self.fittype}.json")) as json_file:
            fitparams = json.load(json_file)
        self.fitparams = fitparams

    def load_global_fitparams(self):
        with open(
            os.path.join(self.fit_dir, f"{self.fittype}_global.json")
        ) as json_file:
            fitparams_global = json.load(json_file)
        self.fitparams_global = fitparams_global


if __name__ == "__main__":

    redshift = 0.2666

    bands_for_global_fit = [
        "P48+ZTF_g",
        "P48+ZTF_r",
        "P48+ZTF_i",
        #  "Swift+UVW2",
        # "Swift+UVW1",
        "Swift+UVM2",
    ]
    with_p200 = [
        "P48+ZTF_g",
        "P48+ZTF_r",
        "P48+ZTF_i",
        # "Swift+UVW2",
        # "Swift+UVW1",
        "Swift+UVM2",
        # "P200+J",
        # "P200+H",
        # "P200+Ks",
    ]

    nbins = 60

    fittype = "blackbody"
    fitglobal = False
    fitlocal = True

    sed = SED(redshift=redshift, fittype=fittype, nbins=nbins)
    if fitglobal:
        sed.fit_global(bands=bands_for_global_fit, plot=True)
    sed.load_global_fitparams()
    if fitlocal:
        if fittype == "powerlaw":
            sed.fit_bins(
                alpha=sed.fitparams_global["alpha"],
                alpha_err=sed.fitparams_global["alpha_err"],
                bands=bands_for_global_fit,
                min_bands_per_bin=2,
                # neccessary_bands=["Swift+UVM2"],
                verbose=False,
            )
        else:
            sed.fit_bins(
                extinction_av=sed.fitparams_global["extinction_av"],
                extinction_av_err=sed.fitparams_global["extinction_av_err"],
                extinction_rv=sed.fitparams_global["extinction_rv"],
                extinction_rv_err=sed.fitparams_global["extinction_rv_err"],
                bands=bands_for_global_fit,
                min_bands_per_bin=2,
                neccessary_bands=["Swift+UVM2"],
                verbose=False,
            )
    sed.load_fitparams()
    sed.plot_lightcurve(bands=bands_for_global_fit)
    # sed.plot_lightcurve(bands=with_p200)
    # if fittype == "blackbody":
    #     sed.plot_temperature()
    sed.plot_luminosity()
