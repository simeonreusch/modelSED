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
from scipy.interpolate import UnivariateSpline
from fit import FitSpectrum, blackbody_spectrum
import utilities

FIG_WIDTH = 8
FONTSIZE = 10


class SED():
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
            if index != len(mjd_iter)-1:
                instrumentfilters = {}
                df = lc.query(f"mjd >= {mjd_iter[index]} and mjd < {mjd_iter[index+1]}")
                for instrument in df.instrument.unique():
                    for band in df.query("instrument == @instrument")["filter"].unique():
                        mag = df.query("instrument == @instrument and filter == @band")["magpsf"].values
                        mag_err = df.query("instrument == @instrument and filter == @band")["sigmamagpsf"].values
                        mean_mag = np.mean(mag)
                        mean_mag_err = np.sqrt(np.sum(mag_err**2))/len(mag_err)
                        combin = {f"{instrument}_{band}": [mean_mag, mean_mag_err]}
                        instrumentfilters.update(combin)
                instrumentfilters.update({"mjd": mjd})
                slices.update({index: instrumentfilters})
        return slices


    def create_epochs(self, nslices=30, **kwargs):
        """" """
        print(f"Fitting {nslices} time slices.\n")
        mean_mags = self.get_mean_magnitudes(nslices+1)
        fitparams = {}
        i = 0
        progress_bar = ProgressBar(len(mean_mags))
        for index, entry in enumerate(mean_mags):
            if len(mean_mags[entry]) > 2:
                result = self.fit_sed(mean_mags[entry], **kwargs)
                fitparams.update({i: result})
                i+=1
            progress_bar.update(index)
        progress_bar.update(len(mean_mags))

        with open(os.path.join(self.fit_dir, f"{self.fittype}.json"), "w") as outfile:
            json.dump(fitparams, outfile)

    def plot_lightcurve(self):
        """" """
        fitparams = self.fitparams
        # First the observational data
        lc_file = os.path.join(self.lc_dir, "full_lc_without_p200.csv")
        lc = pd.read_csv(lc_file)
        mjds = lc.mjd.values
        mjd_min = np.min(mjds)
        mjd_max = np.max(mjds)

        plt.figure(figsize=(FIG_WIDTH, 0.5*FIG_WIDTH), dpi=300)
        ax1 = plt.subplot(111)
        ax1.set_ylabel("Magnitude (AB)")
        ax1.set_xlabel("MJD")
        ax1.invert_yaxis()

        for key in self.filter_wl.keys():
            instrumentfilter = key.split("_")
            df = lc.query(f"instrument == '{instrumentfilter[0]}' and filter == '{instrumentfilter[1]}'")
            ax1.scatter(df.mjd, df.magpsf, color=self.cmap[key], marker=".", alpha=0.5, edgecolors=None)

        # Now evaluate the model spectrum 
        wavelengths = np.arange(1000, 60000, 10) * u.AA
        frequencies = const.c.value/(wavelengths.value*1E-10)
        df_model = pd.DataFrame(columns=["mjd", "band", "mag"])

        for entry in fitparams:

            if self.fittype == "powerlaw":
                nu = (frequencies**fitparams[entry]["alpha"] * fitparams[entry]["scale"]) *u.erg / u.cm**2 * u.Hz / u.s
                spectrum = sncosmo_spectral_v13.Spectrum(wave=wavelengths, flux=nu, unit=utilities.FNU)
            
            if self.fittype == "bb":
                spectrum = blackbody_spectrum(temperature=fitparams[entry]["temperature"], scale=fitparams[entry]["scale"], redshift=self.redshift, extinction_av=fitparams[entry]["extinction_av"], extinction_rv=fitparams[entry]["extinction_rv"])

            for band in self.filter_wl.keys():
                mag = utilities.magnitude_in_band(band, spectrum)
                df_model = df_model.append({"mjd": fitparams[entry]["mjd"], "band": band, "mag": mag}, ignore_index=True)

        bands_to_plot = self.filter_wl
        del bands_to_plot["P200_J"]; del bands_to_plot["P200_H"]; del bands_to_plot["P200_Ks"]

        for key in bands_to_plot.keys():
            df_model_band = df_model.query(f"band == '{key}'")
            spline = UnivariateSpline(df_model_band.mjd.values, df_model_band.mag.values)
            spline.set_smoothing_factor(0.001) 
            ax1.plot(df_model_band.mjd.values, spline(df_model_band.mjd.values), color=self.cmap[key]) 

        plt.savefig(f"plots/lightcurve_{self.fittype}.png")
        plt.close()

    def plot_luminosity(self):
        fitparams = self.fitparams
        mjds = []
        lumi_without_nir = []
        lumi_with_nir = []
        for entry in fitparams:
            mjds.append(fitparams[entry]["mjd"])
            lumi_without_nir.append(fitparams[entry]["luminosity_uv_optical"])
            lumi_with_nir.append(fitparams[entry]["luminosity_uv_nir"])
        plt.figure(figsize=(FIG_WIDTH, 0.5*FIG_WIDTH), dpi=300)
        ax1 = plt.subplot(111)
        ax1.set_ylabel("Luminosity [erg/s]")
        ax1.set_xlabel("MJD")
        ax1.plot(mjds, lumi_without_nir, label="UV to Optical")
        ax1.plot(mjds, lumi_with_nir, label="UV to NIR")
        ax1.legend()
        plt.savefig(f"plots/luminosity_{self.fittype}.png")
        plt.close()

    def load_fitparams(self):
        with open(os.path.join(self.fit_dir, f"{self.fittype}.json")) as json_file:
            fitparams = json.load(json_file)
        self.fitparams = fitparams


redshift = 0.2666

sed = SED(redshift=redshift, fittype="bb")
# sed.create_epochs(29, extinction_av=1.7, extinction_rv=3.1)
sed.load_fitparams()
sed.plot_lightcurve()
sed.plot_luminosity()
