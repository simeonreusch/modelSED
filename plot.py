#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy import constants as const
import utilities

FIG_WIDTH = 8
FONTSIZE = 10

cmap = utilities.load_info_json("cmap")
filter_wl = utilities.load_info_json("filter_wl")

def plot_sed(mags: dict, spectrum, annotations: dict = None, plotmag: bool=False):
    """ """
    if "temperature" in annotations.keys():
        outpath = os.path.join("plots", "bb")
    elif "alpha" in annotations.keys():
        outpath = os.path.join("plots", "blackbody")
    else:
        outpath = "plots"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    frequencies = utilities.lambda_to_nu(spectrum.wave) * u.Hz
    spectrum_mags = []
    for index, wavelength in enumerate(spectrum.wave):
        spectrum_mag = utilities.flux_to_abmag(spectrum.flux[index])
        spectrum_mags.append(spectrum_mag)

    plt.figure(figsize=(FIG_WIDTH, 0.75*FIG_WIDTH), dpi=300)
    ax1 = plt.subplot(111)
    plt.xscale('log')

    if not plotmag:
        ax1.set_ylabel(r"$F_\nu~[$erg$~/~ s \cdot $cm$^2 \cdot$ Hz]")
        ax1.set_xlabel(r"$\nu$ [Hz]")
        plt.yscale('log')
        ax1.set_ylim([2E-28, 4E-27])
        ax1.set_xlim([3.5E14, 2E15])
        ax1.plot(frequencies.value, spectrum._flux)
        for key in mags.keys():
            mag = mags[key]["observed"]
            mag_err = mags[key]["observed_err"]
            flux = utilities.abmag_to_flux(mag)
            flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)
            ax1.errorbar(mags[key]["frequency"], flux, flux_err, color=cmap[key], fmt=".")
        ax2 = ax1.secondary_xaxis("top", functions=(utilities.lambda_to_nu, utilities.nu_to_lambda))
        ax2.set_xlabel(r"$\lambda~[\AA]$")
    else:
        ax1.set_ylabel("Magnitude [AB]")
        ax1.set_xlabel(r"$\lambda$")
        ax1.invert_yaxis()
        ax1.set_ylim([21,17])
        ax1.plot(spectrum.wave, spectrum_mags)
        for key in mags.keys():
            ax1.scatter(filter_wl[key], mags[key]["observed"], color=cmap[key])

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
            annotationstr += f"red. $\\chi^2$={reduced_chisquare:.2f}"
        plt.annotate(annotationstr, (1.2E15, 2.2E-27), fontsize=FONTSIZE, color="black", bbox=bbox)

    plt.savefig(os.path.join(outpath, f"{mjd}.png"))
    plt.close()