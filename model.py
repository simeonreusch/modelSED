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
from fit import FitSpectrum
import utilities

REDSHIFT = 0.2666
FIG_WIDTH = 8
FONTSIZE = 10

FITTYPE = "bb"

data_dir = "data"
plot_dir = "plots"
lc_dir = os.path.join(data_dir, "lightcurves")
fit_dir = "fit"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(fit_dir):
    os.makedirs(fit_dir)

cmap = utilities.load_info_json("cmap")
filter_wl = utilities.load_info_json("filter_wl")

def fit_sed(mags, **kwargs):
    """ """
    fit = FitSpectrum(mags, REDSHIFT)
    if FITTYPE == "bb":
        fitresult = fit.fit_blackbody(**kwargs)
    if FITTYPE == "powerlaw":
        fitresult = fit.fit_powerlaw(**kwargs)
    
    return fitresult

def get_mean_magnitudes(bins=30):
    """ """
    lc_file = os.path.join(lc_dir, "full_lc.csv")
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


def fit_slices(nslices=30, **kwargs):
    """" """
    print(f"Fitting {nslices} time slices.\n")
    mean_mags = get_mean_magnitudes(nslices+1)
    fitparams = {}
    i = 0
    progress_bar = ProgressBar(len(mean_mags))
    for index, entry in enumerate(mean_mags):
        if len(mean_mags[entry]) > 2:
            result = fit_sed(mean_mags[entry], **kwargs)
            fitparams.update({i: result})
            i+=1
        progress_bar.update(index)
    progress_bar.update(len(mean_mags))

    with open(os.path.join(fit_dir, f"{FITTYPE}.json"), "w") as outfile:
        json.dump(fitparams, outfile)

def plot_lightcurve(fitparams, alpha=None):
    """" """
    # First the observational data
    lc_file = os.path.join(lc_dir, "full_lc_without_p200.csv")
    lc = pd.read_csv(lc_file)
    mjds = lc.mjd.values
    mjd_min = np.min(mjds)
    mjd_max = np.max(mjds)

    plt.figure(figsize=(FIG_WIDTH, 0.5*FIG_WIDTH), dpi=300)
    ax1 = plt.subplot(111)
    ax1.set_ylabel("Magnitude (AB)")
    ax1.set_xlabel("MJD")
    ax1.invert_yaxis()

    for key in filter_wl.keys():
        instrumentfilter = key.split("_")
        df = lc.query(f"instrument == '{instrumentfilter[0]}' and filter == '{instrumentfilter[1]}'")
        ax1.scatter(df.mjd, df.magpsf, color=cmap[key], marker=".", alpha=0.5, edgecolors=None)

    # Now evaluate the model spectrum 
    wavelengths = np.arange(1000, 60000, 10) * u.AA
    frequencies = const.c.value/(wavelengths.value*1E-10)
    df_model = pd.DataFrame(columns=["mjd", "band", "mag"])

    for entry in fitparams:
        powerlaw_nu = (frequencies**fitparams[entry]["alpha"] * fitparams[entry]["scale"]) *u.erg / u.cm**2 * u.Hz / u.s
        spectrum = sncosmo_spectral_v13.Spectrum(wave=wavelengths, flux=powerlaw_nu, unit=utilities.FNU)

        for band in filter_wl.keys():
            mag = utilities.magnitude_in_band(band, spectrum)
            df_model = df_model.append({"mjd": fitparams[entry]["mjd"], "band": band, "mag": mag}, ignore_index=True)

    bands_to_plot = filter_wl
    del bands_to_plot["P200_J"]; del bands_to_plot["P200_H"]; del bands_to_plot["P200_Ks"]

    for key in bands_to_plot.keys():
        df_model_band = df_model.query(f"band == '{key}'")
        spline = UnivariateSpline(df_model_band.mjd.values, df_model_band.mag.values)
        spline.set_smoothing_factor(0.001) 
        ax1.plot(df_model_band.mjd.values, spline(df_model_band.mjd.values), color=cmap[key]) 

    plt.savefig(f"plots/lightcurve_{FITTYPE}.png")
    plt.close()

def plot_luminosity(fitparams):
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
    plt.savefig(f"plots/luminosity_{FITTYPE}.png")
    plt.close()

def load_fitparams():
    with open(os.path.join(fit_dir, f"{FITTYPE}.json")) as json_file:
        fitparams = json.load(json_file)
    return fitparams

fit_slices(29, extinction_av=1.7, extinction_rv=3.1)
fitparams = load_fitparams()
# plot_lightcurve(fitparams)
plot_luminosity(fitparams)
