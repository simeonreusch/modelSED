#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import constants as const
from astropy.utils.console import ProgressBar
from .fit import FitSpectrum
from . import utilities, plot, sncosmo_spectral_v13


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

        self.read_lightcurve()
        print(
            f"Initialized with {self.nbins} time slices, redshift={self.redshift} and fittype={self.fittype}"
        )

    def load_lightcurves(self):
        # self.path_to_lightcurve

        return None

    def fit_one_bin(self, binned_lc_df, **kwargs):
        """ """
        fit = FitSpectrum(binned_lc_df, self.fittype, self.redshift)
        fitresult = fit.fit_bin(**kwargs)

        return fitresult

    def get_mean_magnitudes(
        self,
        bands: list = None,
        min_bands_per_bin: list = None,
        neccessary_bands: list = None,
    ):
        """ """
        lc = self.lc

        mjds = lc.obsmjd.values
        mjd_min = np.min(mjds)
        mjd_max = np.max(mjds)

        mjd_iter = np.linspace(mjd_min, mjd_max, num=self.nbins + 1)

        if bands is None:
            bands_to_fit = list(self.filter_wl.keys())
        else:
            bands_to_fit = bands

        if min_bands_per_bin is None:
            min_bands_per_bin = 2

        if neccessary_bands is None:
            neccessary_bands = []

        lc = lc[lc.telescope_band.isin(bands_to_fit)]
        lc.reset_index(inplace=True)
        lc = lc.drop(columns=["index"])

        temp_df = pd.DataFrame()

        for index, mjd in enumerate(mjd_iter):
            if index != len(mjd_iter) - 1:
                df = lc.query(
                    f"obsmjd >= {mjd_iter[index]} and obsmjd < {mjd_iter[index+1]}"
                )
                for telescope_band in df.telescope_band.unique():
                    _df = df.query("telescope_band == @telescope_band")
                    mean_mag = np.mean(_df["mag"].values)
                    mean_mag_err = np.mean(_df["mag_err"].values)
                    entries = len(_df["mag"].values)
                    mean_obsmjd = np.mean([mjd_iter[index], mjd_iter[index + 1]])
                    wavelength = self.filter_wl[telescope_band]
                    temp_df = temp_df.append(
                        {
                            "telescope_band": telescope_band,
                            "wavelength": wavelength,
                            "mean_obsmjd": mean_obsmjd,
                            "entries": entries,
                            "mean_mag": mean_mag,
                            "mean_mag_err": mean_mag_err,
                        },
                        ignore_index=True,
                    )

        result_df = pd.DataFrame()

        # Now we apply our min_bands_per_bin and neccessary_bands criteria
        for mjd in temp_df.mean_obsmjd.unique():
            _df = temp_df.query(f"mean_obsmjd == {mjd}")
            if len(_df) >= min_bands_per_bin and set(neccessary_bands).issubset(
                _df.telescope_band.unique()
            ):
                result_df = result_df.append(_df, ignore_index=True)

        return result_df

    def fit_bins(
        self, min_bands_per_bin: float = None, neccessary_bands: list = None, **kwargs
    ):
        """" """

        print(f"Fitting {self.nbins} time bins.\n")

        if "bands" in kwargs:
            if min_bands_per_bin is None:
                min_bands_per_bin = len(kwargs["bands"])
            print(f"Bands which are fitted: {kwargs['bands']}")
            bands = kwargs["bands"]
        else:
            if min_bands_per_bin is None:
                min_bands_per_bin = 2
            print(f"Fitting all bands")
            bands = binned_lc_df.telescope_band.unique()

        if not neccessary_bands:
            neccessary_bands = []
            print("No band MUST be present in each bin to be fit")
        else:
            print(f"{neccessary_bands} MUST be present in each bin to be fit")

        print(
            f"At least {min_bands_per_bin} bands must be present in each bin to be fit"
        )

        binned_lc_df = self.get_mean_magnitudes(
            bands=bands,
            min_bands_per_bin=min_bands_per_bin,
            neccessary_bands=neccessary_bands,
        )

        fitparams = {}

        progress_bar = ProgressBar(len(binned_lc_df.mean_obsmjd.unique()))

        for index, mjd in enumerate(binned_lc_df.mean_obsmjd.unique()):
            _df = binned_lc_df.query(f"mean_obsmjd == {mjd}")
            result = self.fit_one_bin(binned_lc_df=_df, **kwargs)
            fitparams.update({index: result})
            progress_bar.update(index)

        progress_bar.update(len(binned_lc_df.mean_obsmjd.unique()))

        with open(os.path.join(self.fit_dir, f"{self.fittype}.json"), "w") as outfile:
            json.dump(fitparams, outfile)

    def fit_global(self, **kwargs):
        """ """
        print(
            f"Fitting full lightcurve for global parameters (with {self.nbins} bins)."
        )
        if "bands" in kwargs:
            binned_lc_df = self.get_mean_magnitudes(bands=kwargs["bands"])
        else:
            binned_lc_df = self.get_mean_magnitudes()

        if "plot" in kwargs:
            fit = FitSpectrum(
                binned_lc_df,
                fittype=self.fittype,
                redshift=self.redshift,
                plot=kwargs["plot"],
            )
        else:
            fit = FitSpectrum(
                binned_lc_df, fittype=self.fittype, redshift=self.redshift
            )

        if "bands" in kwargs:
            bands = kwargs["bands"]
        else:
            bands = None

        if "min_datapoints" in kwargs:
            min_datapoints = kwargs["min_datapoints"]
        else:
            min_datapoints = len(bands)

        result = fit.fit_global_parameters(min_datapoints=min_datapoints)

        with open(
            os.path.join(self.fit_dir, f"{self.fittype}_global.json"), "w"
        ) as outfile:
            json.dump(result, outfile)
            return result

    def plot_lightcurve(self, bands, **kwargs):
        """" """
        plot.plot_lightcurve(
            self.lc, bands, self.fitparams, self.fittype, self.redshift, **kwargs
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

    def read_lightcurve(self):
        if self.path_to_lightcurve is None:
            self.path_to_lightcurve = os.path.join(self.lc_dir, "full_lc_fp.csv")

        lc = pd.read_csv(self.path_to_lightcurve)
        lc.drop(columns=["Unnamed: 0"], inplace=True)

        lc.insert(len(lc.columns), "telescope_band", lc.telescope + "+" + lc.band)

        self.lc = lc


if __name__ == "__main__":

    redshift = 0.2666

    bands = [
        "P48+ZTF_g",
        "P48+ZTF_r",
        "P48+ZTF_i",
        "Swift+UVM2",
    ]

    nbins = 60

    fittype = "powerlaw"
    fitglobal = True
    fitlocal = True

    path_to_lightcurve = os.path.join("data", "lightcurves", "full_lightcurve.csv")

    sed = SED(
        redshift=redshift,
        fittype=fittype,
        nbins=nbins,
        path_to_lightcurve=path_to_lightcurve,
    )
    if fitglobal:
        sed.fit_global(bands=bands, plot=False)
    sed.load_global_fitparams()

    if fitlocal:
        if fittype == "powerlaw":
            sed.fit_bins(
                alpha=sed.fitparams_global["alpha"],
                alpha_err=sed.fitparams_global["alpha_err"],
                bands=bands,
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
                bands=bands,
                min_bands_per_bin=2,
                neccessary_bands=["Swift+UVM2"],
                verbose=False,
            )
    sed.load_fitparams()
    sed.plot_lightcurve(bands=bands)
    # sed.plot_lightcurve(bands=with_p200)
    # if fittype == "blackbody":
    #     sed.plot_temperature()
    sed.plot_luminosity()
