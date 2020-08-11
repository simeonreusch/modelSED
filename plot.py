#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
from astropy import constants as const
from scipy.interpolate import UnivariateSpline
import utilities, sncosmo_spectral_v13

FIG_WIDTH = 8
FONTSIZE = 10

cmap = utilities.load_info_json("cmap")
filter_wl = utilities.load_info_json("filter_wl")


def plot_sed(
    mags: dict, spectrum, annotations: dict = None, plotmag: bool = False, **kwargs
):
    """ """
    if "temperature" in annotations.keys():
        outpath = os.path.join("plots", "blackbody")
    elif "alpha" in annotations.keys():
        outpath = os.path.join("plots", "powerlaw")
    else:
        outpath = "plots"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    frequencies = utilities.lambda_to_nu(spectrum.wave) * u.Hz
    spectrum_mags = []

    for index, wavelength in enumerate(spectrum.wave):
        spectrum_mag = utilities.flux_to_abmag(spectrum.flux[index])
        spectrum_mags.append(spectrum_mag)

    plt.figure(figsize=(FIG_WIDTH, 0.75 * FIG_WIDTH), dpi=300)
    ax1 = plt.subplot(111)
    plt.xscale("log")

    if not plotmag:
        ax1.set_ylabel(r"$F_\nu~[$erg$~/~ s \cdot $cm$^2 \cdot$ Hz]")
        ax1.set_xlabel(r"$\nu$ [Hz]")
        plt.yscale("log")
        ax1.set_ylim([2e-28, 4e-27])
        ax1.set_xlim([3.5e14, 2e15])
        ax1.plot(frequencies.value, spectrum._flux)
        for key in mags.keys():
            mag = mags[key]["observed"]
            mag_err = mags[key]["observed_err"]
            flux = utilities.abmag_to_flux(mag)
            flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)
            ax1.errorbar(
                mags[key]["frequency"], flux, flux_err, color=cmap[key], fmt="."
            )
        ax2 = ax1.secondary_xaxis(
            "top", functions=(utilities.lambda_to_nu, utilities.nu_to_lambda)
        )
        ax2.set_xlabel(r"$\lambda~[\AA]$")
    else:
        ax1.set_ylabel("Magnitude [AB]")
        ax1.set_xlabel(r"$\lambda$")
        ax1.invert_yaxis()
        ax1.set_ylim([21, 17])
        ax1.plot(spectrum.wave, spectrum_mags)
        for key in mags.keys():
            ax1.scatter(filter_wl[key], mags[key]["observed"], color=cmap[key])
        ax2 = ax1.secondary_xaxis(
            "top", functions=(utilities.nu_to_lambda, utilities.lambda_to_nu)
        )
        ax2.set_xlabel(r"$\nu$ [Hz]")

    bbox = dict(boxstyle="round", fc="none", ec="black")

    if annotations:
        annotationstr = ""
        if "alpha" in annotations.keys():
            alpha = annotations["alpha"]
            annotationstr += f"$\\alpha$={alpha:.3f}\n"
        if "temperature" in annotations.keys():
            temperature = annotations["temperature"]
            annotationstr += f"temperature={temperature:.2E}\n"
        if "scale" in annotations.keys():
            scale = annotations["scale"]
            annotationstr += f"scale={scale:.2E}\n"
        if "mjd" in annotations.keys():
            mjd = annotations["mjd"]
            annotationstr += f"MJD={mjd:.2f}\n"
        if "reduced_chisquare" in annotations.keys():
            reduced_chisquare = annotations["reduced_chisquare"]
            annotationstr += f"red. $\\chi^2$={reduced_chisquare:.2f}\n"
        if "bolometric_luminosity" in annotations.keys():
            bolometric_luminosity = annotations["bolometric_luminosity"]
            annotationstr += f"bol. lum.={bolometric_luminosity:.2E}\n"

        if annotationstr.endswith("\n"):
            annotationstr = annotationstr[:-2]

        if not plotmag:
            annotation_location = (1.2e15, 2.2e-27)
        else:
            annotation_location = (2e4, 18.0)

        plt.annotate(
            annotationstr,
            annotation_location,
            fontsize=FONTSIZE,
            color="black",
            bbox=bbox,
        )

    plt.savefig(os.path.join(outpath, f"{mjd}.png"))
    plt.close()


def plot_luminosity(fitparams, fittype, **kwargs):
    """ """
    mjds = []
    lumi_without_nir = []
    lumi_with_nir = []
    bolo_lumi = []

    for entry in fitparams:
        mjds.append(fitparams[entry]["mjd"])
        lumi_without_nir.append(fitparams[entry]["luminosity_uv_optical"])
        lumi_with_nir.append(fitparams[entry]["luminosity_uv_nir"])

    if fittype == "blackbody":
        for entry in fitparams:
            bolo_lumi.append(fitparams[entry]["bolometric_luminosity"])

    plt.figure(figsize=(FIG_WIDTH, 0.5 * FIG_WIDTH), dpi=300)
    ax1 = plt.subplot(111)
    ax1.set_ylabel("Intrinsic Luminosity [erg/s]")
    ax1.set_xlabel("MJD")

    if fittype == "blackbody":
        ax1.plot(mjds, bolo_lumi, label="Bolometric Luminosity")
    else:
        ax1.plot(mjds, lumi_without_nir, label="UV to Optical")
        ax1.plot(mjds, lumi_with_nir, label="UV to NIR")

    ax1.legend()
    plt.savefig(f"plots/luminosity_{fittype}.png")
    plt.close()


def plot_lightcurve(datafile, fitparams, fittype, redshift, **kwargs):
    """ """
    filter_wl = utilities.load_info_json("filter_wl")
    cmap = utilities.load_info_json("cmap")
    lc = pd.read_csv(datafile)

    mjds = lc.mjd.values
    mjd_min = np.min(mjds)
    mjd_max = np.max(mjds)

    plt.figure(figsize=(FIG_WIDTH, 0.5 * FIG_WIDTH), dpi=300)
    ax1 = plt.subplot(111)
    ax1.set_ylabel("Magnitude (AB)")
    ax1.set_xlabel("MJD")
    ax1.invert_yaxis()

    for key in filter_wl.keys():
        instrumentfilter = key.split("_")
        df = lc.query(
            f"instrument == '{instrumentfilter[0]}' and filter == '{instrumentfilter[1]}'"
        )
        ax1.scatter(
            df.mjd, df.magpsf, color=cmap[key], marker=".", alpha=0.5, edgecolors=None,
        )

    # Now evaluate the model spectrum
    wavelengths = np.arange(1000, 60000, 10) * u.AA
    frequencies = const.c.value / (wavelengths.value * 1e-10)
    df_model = pd.DataFrame(columns=["mjd", "band", "mag"])

    for entry in fitparams:

        if fittype == "powerlaw":
            nu = (
                (frequencies ** fitparams[entry]["alpha"] * fitparams[entry]["scale"])
                * u.erg
                / u.cm ** 2
                * u.Hz
                / u.s
            )
            spectrum = sncosmo_spectral_v13.Spectrum(
                wave=wavelengths, flux=nu, unit=utilities.FNU
            )

        if fittype == "blackbody":
            spectrum = utilities.blackbody_spectrum(
                temperature=fitparams[entry]["temperature"],
                scale=fitparams[entry]["scale"],
                redshift=redshift,
                extinction_av=fitparams[entry]["extinction_av"],
                extinction_rv=fitparams[entry]["extinction_rv"],
            )

        for band in filter_wl.keys():
            mag = utilities.magnitude_in_band(band, spectrum)
            df_model = df_model.append(
                {"mjd": fitparams[entry]["mjd"], "band": band, "mag": mag},
                ignore_index=True,
            )

    bands_to_plot = filter_wl
    del bands_to_plot["P200_J"]
    del bands_to_plot["P200_H"]
    del bands_to_plot["P200_Ks"]

    for key in bands_to_plot.keys():
        df_model_band = df_model.query(f"band == '{key}'")
        spline = UnivariateSpline(df_model_band.mjd.values, df_model_band.mag.values)
        spline.set_smoothing_factor(0.001)
        ax1.plot(
            df_model_band.mjd.values, spline(df_model_band.mjd.values), color=cmap[key],
        )

    plt.savefig(f"plots/lightcurve_{fittype}.png")
    plt.close()
