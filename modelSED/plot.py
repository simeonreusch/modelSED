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
from . import utilities, sncosmo_spectral_v13

FIG_WIDTH = 6
FONTSIZE = 10
ANNOTATION_FONTSIZE = 8
DPI = 400

cmap = utilities.load_info_json("cmap")
filter_wl = utilities.load_info_json("filter_wl")
filterlabel = utilities.load_info_json("filterlabel")


def plot_sed_from_flux(
    flux: list,
    bands: list,
    spectrum,
    fittype: str = "powerlaw",
    redshift: float = None,
    flux_err: list = None,
    index: int = None,
    plotmag: bool = False,
):
    """ """

    outpath = os.path.join("plots", "global", fittype)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    wl_observed = []
    for band in bands:
        wl_observed.append(filter_wl[band])

    freq_observed = const.c.value / (np.array(wl_observed) * 1e-10)
    plt.figure(figsize=(8, 0.75 * 8), dpi=DPI)
    ax1 = plt.subplot(111)
    ax1.invert_yaxis()

    if plotmag:
        ax1.set_ylim([20.5, 17.5])
        ax1.set_xlim(1000, 10000)
    else:
        ax1.set_ylim([2e-28, 4e-27])
        ax1.set_xlim([3.5e14, 2e15])
        plt.xscale("log")
        plt.yscale("log")

    if plotmag:
        mag_model = []
        model_wl = []
        for band in bands:
            mag_model.append(utilities.magnitude_in_band(band, spectrum))
            model_wl.append(filter_wl[band])
        ax1.scatter(model_wl, mag_model, marker=".", color="black", label="model")
        if flux_err is not None:
            ax1.errorbar(
                utilities.nu_to_lambda(freq_observed),
                utilities.flux_to_abmag(flux),
                utilities.flux_err_to_abmag_err(flux, flux_err),
                fmt=".",
                color="blue",
                label="data",
            )
        else:
            ax1.scatter(
                utilities.nu_to_lambda(freq_observed),
                utilities.flux_to_abmag(flux),
                marker=".",
                color="blue",
                label="data",
            )
        ax1.plot(
            np.array(spectrum.wave),
            utilities.flux_to_abmag(spectrum.flux),
            color="gray",
            label="model spectrum",
        )
        ax1.set_ylabel("Magnitude [AB]")
    else:
        if flux_err is not None:
            ax1.errorbar(
                freq_observed, flux, flux_err, fmt=".", color="blue", label="data"
            )
        else:
            ax1.scatter(freq_observed, flux, marker=".", color="blue", label="data")

        ax1.plot(
            utilities.lambda_to_nu(np.array(spectrum.wave)),
            spectrum.flux,
            color="gray",
            label="model spectrum",
        )
        ax1.set_ylabel(r"$F_\nu~[$erg$~/~ s \cdot $cm$^2 \cdot$ Hz]")

    ax2 = ax1.secondary_xaxis(
        "top", functions=(utilities.lambda_to_nu, utilities.nu_to_lambda)
    )
    ax2.set_xlabel(r"$\nu$ [Hz]")
    ax1.set_xlabel(r"$\lambda~[\AA]$")
    plt.legend()

    if plotmag:
        if index is not None:
            plt.savefig(f"{fittype}_global_bin_{index+1}_mag.png")
        else:
            plt.savefig(f"{fittype}_global_mag.png")
    else:
        if index is not None:
            plt.savefig(f"{fittype}_global_bin_{index+1}_flux.png")
        else:
            plt.savefig(f"{fittype}_global_flux.png")

    plt.close()


def plot_sed_from_dict(
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

    # # ugly hack
    # import json
    # update = {"P200+J": {"observed": 17.51, "observed_err": 0.2, "wavelength": 12063.68, "frequency": utilities.lambda_to_nu(12063.68)}, "P200+H": {"observed": 16.45, "observed_err": 0.2, "wavelength": 15696.2, "frequency": utilities.lambda_to_nu(15696.2)},  "P200+Ks": {"observed": 15.33, "observed_err": 0.2, "wavelength": 21301.1, "frequency": utilities.lambda_to_nu(21301.1)}}
    # mags.update(update)
    # with open(os.path.join("fit", "blackbody.json")) as json_file:
    #     outfile = json.load(json_file)
    # bb_data = outfile["11"]
    # bb_spectrum = utilities.blackbody_spectrum(temperature=bb_data["temperature"], scale=bb_data["scale"], redshift=0.2666, extinction_av=bb_data["extinction_av"], extinction_rv=bb_data["extinction_rv"])
    # spectrum_mags_bb = []
    # for index, wavelength in enumerate(bb_spectrum.wave):
    #     spectrum_mag = utilities.flux_to_abmag(bb_spectrum.flux[index])
    #     spectrum_mags_bb.append(spectrum_mag)
    # # end of ugly hack

    plt.figure(figsize=(FIG_WIDTH, 0.75 * FIG_WIDTH), dpi=DPI)
    ax1 = plt.subplot(111)
    plt.xscale("log")

    if not plotmag:
        ax1.set_ylabel(r"$F_\nu~[$erg$~/~ s \cdot $cm$^2 \cdot$ Hz]")
        ax1.set_xlabel(r"$\nu$ [Hz]")
        plt.yscale("log")
        # ax1.set_ylim([1e-28, 4e-26])
        ax1.set_ylim([2e-28, 4e-27])
        # ax1.set_xlim([1e14, 2e15])
        ax1.set_xlim([3.5e14, 2e15])
        if "alpha" in annotations.keys():
            alpha = annotations["alpha"]
        if "alpha_err" in annotations.keys():
            alpha_err = annotations["alpha_err"]

        ax1.plot(frequencies.value, spectrum._flux, color="black")
        # ugly hack
        # ax1.plot(frequencies.value, bb_spectrum._flux, color="blue")
        #
        for key in mags.keys():
            mag = mags[key]["observed"]
            mag_err = mags[key]["observed_err"]
            flux = utilities.abmag_to_flux(mag)
            flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)
            ax1.errorbar(
                mags[key]["frequency"],
                flux,
                flux_err,
                color=cmap[key],
                fmt=".",
                label=filterlabel[key],
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
    bbox2 = dict(boxstyle="round", fc="none", ec="blue")

    if annotations:
        annotationstr = ""
        if "alpha" in annotations.keys():
            alpha = annotations["alpha"]
            annotationstr += f"Spectral index $\\alpha$={alpha:.3f}\n"
        if "temperature" in annotations.keys():
            temperature = annotations["temperature"]
            annotationstr += f"temperature={temperature:.2E}\n"
        if "scale" in annotations.keys():
            scale = annotations["scale"]
            annotationstr += f"normalization $\\beta$={scale:.2E}\n"
        if "mjd" in annotations.keys():
            mjd = annotations["mjd"]
            annotationstr += f"MJD={mjd:.2f}\n"
        # if "reduced_chisquare" in annotations.keys():
        #     temp = "red. $\\chi^2$="
        #     reduced_chisquare = annotations["reduced_chisquare"]
        #     annotationstr += temp
        #     annotationstr += f"{reduced_chisquare:.2f}\n"
        if "bolometric_luminosity" in annotations.keys():
            bolometric_luminosity = annotations["bolometric_luminosity"]
            annotationstr += f"bol. lum.={bolometric_luminosity:.2E}\n"

        if annotationstr.endswith("\n"):
            annotationstr = annotationstr[:-2]

        if not plotmag:
            # annotation_location = (1.2e15, 2.2e-27)
            annotation_location = (0.7e15, 1.5e-26)
        else:
            annotation_location = (2e4, 18.0)

        # plt.annotate(
        #     annotationstr,
        #     annotation_location,
        #     fontsize=FONTSIZE,
        #     color="black",
        #     bbox=bbox,
        # )
        ax1.legend(
            fontsize=FONTSIZE,
            fancybox=True,
            edgecolor="black",
            # loc=(0.77,0.45),
            loc=0,
        )

    plt.savefig(os.path.join(outpath, f"{mjd}.png"))
    plt.close()


def plot_luminosity(fitparams, fittype, **kwargs):
    """ """
    mjds = []
    lumi_without_nir = []
    lumi_with_nir = []
    bolo_lumi = []
    radius = []

    for entry in fitparams:
        mjds.append(fitparams[entry]["mjd"])
        lumi_without_nir.append(fitparams[entry]["luminosity_uv_optical"])
        lumi_with_nir.append(fitparams[entry]["luminosity_uv_nir"])

    if fittype == "blackbody":
        for entry in fitparams:
            bolo_lumi.append(fitparams[entry]["bolometric_luminosity"])
            radius.append(fitparams[entry]["radius"])

    plt.figure(figsize=(FIG_WIDTH, 0.6 * FIG_WIDTH), dpi=DPI)
    ax1 = plt.subplot(111)
    ax1.set_xlabel("MJD")

    if fittype == "blackbody":
        ax1.set_ylabel("Blackbody luminosity [erg/s]")
        plot1 = ax1.plot(mjds, bolo_lumi, label="Blackbody luminosity", color="blue")
        ax2 = ax1.twinx()
        plot2 = ax2.plot(mjds, radius, color="red", label="Blackbody radius")
        ax2.set_ylabel("Blackbody radius [cm]")
        plots = plot1 + plot2
        labels = [p.get_label() for p in plots]
        # ax1.legend(plots, labels, loc=0)
        ax2.yaxis.label.set_color("red")
        ax1.yaxis.label.set_color("blue")
        # ax2.tick_params(axis='y', colors='red')
        # ax1.tick_params(axis='y', colors='blue')
        # ax2.spines['right'].set_color('red')
        # ax1.spines['left'].set_color('blue')
    else:
        ax1.set_ylabel("Intrinsic luminosity [erg/s]")
        ax1.plot(mjds, lumi_without_nir, label="UV to Optical")
        ax1.plot(mjds, lumi_with_nir, label="UV to NIR")
        ax1.legend(fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(f"plots/luminosity_{fittype}.png")
    plt.close()


def plot_lightcurve(df, bands, fitparams=None, fittype=None, redshift=None, **kwargs):
    """ """
    filter_wl = utilities.load_info_json("filter_wl")
    cmap = utilities.load_info_json("cmap")

    mjds = df.obsmjd.values
    mjd_min = np.min(mjds)
    mjd_max = np.max(mjds)

    plt.figure(figsize=(FIG_WIDTH, 0.6 * FIG_WIDTH), dpi=DPI)
    ax1 = plt.subplot(111)
    ax1.set_ylabel("Magnitude [AB]")
    ax1.set_xlabel("MJD")
    ax1.invert_yaxis()

    if bands is None:
        bands_to_plot = df.band.unique()
    else:
        bands_to_plot = bands

    if fitparams:
        alpha = 0.2
    else:
        alpha = 1

    for key in filter_wl.keys():
        if key in bands_to_plot:
            _df = df.query(f"telescope_band == '{key}'")
            ax1.errorbar(
                _df.obsmjd,
                _df.mag,
                _df.mag_err,
                color=cmap[key],
                fmt=".",
                alpha=alpha,
                edgecolors=None,
                label=filterlabel[key],
            )

    # Now evaluate the model spectrum
    wavelengths = np.arange(1000, 60000, 10) * u.AA
    frequencies = const.c.value / (wavelengths.value * 1e-10)
    df_model = pd.DataFrame(columns=["mjd", "band", "mag"])

    if fitparams:
        for entry in fitparams:

            if fittype == "powerlaw":
                alpha = fitparams[entry]["alpha"]
                alpha_err = fitparams[entry]["alpha_err"]
                scale = fitparams[entry]["scale"]
                scale_err = fitparams[entry]["scale_err"]
                flux_nu = (
                    (frequencies ** alpha * scale) * u.erg / u.cm ** 2 * u.Hz / u.s
                )
                flux_nu_err = utilities.powerlaw_error_prop(
                    frequencies, alpha, alpha_err, scale, scale_err
                )
                spectrum = sncosmo_spectral_v13.Spectrum(
                    wave=wavelengths, flux=flux_nu, unit=utilities.FNU
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
                if band in bands_to_plot:

                    wl = filter_wl[band]
                    for index, _wl in enumerate(spectrum.wave):
                        if _wl > wl:
                            mag = utilities.flux_to_abmag(spectrum.flux[index])
                            break
                    # mag = utilities.magnitude_in_band(band, spectrum)
                    df_model = df_model.append(
                        {"mjd": fitparams[entry]["mjd"], "band": band, "mag": mag,},
                        ignore_index=True,
                    )

        for key in filter_wl.keys():
            if key in bands_to_plot:
                df_model_band = df_model.query(f"band == '{key}'")
                spline = UnivariateSpline(
                    df_model_band.mjd.values, df_model_band.mag.values
                )

                spline.set_smoothing_factor(0.001)

                ax1.plot(
                    df_model_band.mjd.values,
                    spline(df_model_band.mjd.values),
                    color=cmap[key],
                )

        if fittype == "powerlaw":
            alphas = set()
            alpha_errs = set()
            for entry in fitparams:
                alpha = fitparams[entry]["alpha"]
                alpha_err = fitparams[entry]["alpha_err"]
                alphas.add(alpha)
                alpha_errs.add(alpha_err)
            if len(alphas) == 1:
                plt.title(
                    f"Powerlaw spectrum fit, spectral index $\\alpha$ = {list(alphas)[0]:.2f} $\pm$ {list(alpha_errs)[0]:.2f}"
                )

        if fittype == "blackbody":
            extinction_avs = set()
            extinction_rvs = set()
            extinction_av_errs = set()
            extinction_rv_errs = set()
            for entry in fitparams:
                av = fitparams[entry]["extinction_av"]
                rv = fitparams[entry]["extinction_rv"]
                av_err = fitparams[entry]["extinction_av_err"]
                rv_err = fitparams[entry]["extinction_rv_err"]
                extinction_avs.add(av)
                extinction_rvs.add(rv)
                extinction_av_errs.add(av_err)
                extinction_rv_errs.add(rv_err)
            title = "BB spectrum fit, "
            if len(extinction_avs) == 1:
                title += f"ext. $A_V$ = {list(extinction_avs)[0]:.2f} $\pm$ {list(extinction_av_errs)[0]:.2f}"
            if len(extinction_rvs) == 1:
                title += f", $R_V$ = {list(extinction_rvs)[0]:.2f} $\pm$ {list(extinction_rv_errs)[0]:.2f}"

            if len(title) > 0:
                plt.title(title)

    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    if fitparams:
        plt.savefig(f"plots/lightcurve_{fittype}.png")
    else:
        plt.savefig(f"plots/lightcurve.png")

    plt.close()


def plot_temperature(fitparams, **kwargs):
    """ """
    plt.figure(figsize=(FIG_WIDTH, 0.5 * FIG_WIDTH), dpi=DPI)
    ax1 = plt.subplot(111)
    ax1.set_ylabel("Temperature [K]")
    ax1.set_xlabel("MJD")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Radius")

    mjds = []
    temps = []
    radii = []
    for entry in fitparams:
        mjds.append(fitparams[entry]["mjd"])
        temps.append(fitparams[entry]["temperature"])
        radii.append(fitparams[entry]["radius"])
    ax1.plot(mjds, temps, color="blue")
    ax2.plot(mjds, radii, color="red")
    plt.savefig(f"plots/temperature_radius.png")
    plt.close()


def plot_sed(df, spectrum, annotations: dict = None, plotmag: bool = False, **kwargs):
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

    plt.figure(figsize=(FIG_WIDTH, 0.75 * FIG_WIDTH), dpi=DPI)
    ax1 = plt.subplot(111)
    plt.xscale("log")

    if not plotmag:
        ax1.set_ylabel(r"$F_\nu~[$erg$~/~ s \cdot $cm$^2 \cdot$ Hz]")
        ax1.set_xlabel(r"$\nu$ [Hz]")
        plt.yscale("log")
        ax1.set_ylim([2e-28, 4e-27])
        ax1.set_xlim([3.5e14, 2e15])
        if "alpha" in annotations.keys():
            alpha = annotations["alpha"]
        if "alpha_err" in annotations.keys():
            alpha_err = annotations["alpha_err"]

        ax1.plot(frequencies.value, spectrum._flux, color="black")

        for index, row in df.iterrows():
            ax1.errorbar(
                utilities.lambda_to_nu(row.wavelength),
                row["mean_flux"],
                row["mean_flux_err"],
                color=cmap[row.telescope_band],
                fmt=".",
                label=filterlabel[row.telescope_band],
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
        ax1.plot(spectrum.wave, utilities.flux_to_abmag(spectrum._flux))

        for index, row in df.iterrows():
            ax1.errorbar(
                row.wavelength,
                row["mean_mag"],
                row["mean_mag_err"],
                color=cmap[row.telescope_band],
                fmt=".",
                label=filterlabel[row.telescope_band],
            )
        ax2 = ax1.secondary_xaxis(
            "top", functions=(utilities.nu_to_lambda, utilities.lambda_to_nu)
        )
        ax2.set_xlabel(r"$\nu$ [Hz]")

    bbox = dict(boxstyle="round", fc="none", ec="black")
    bbox2 = dict(boxstyle="round", fc="none", ec="blue")

    if annotations:
        annotationstr = ""
        if "alpha" in annotations.keys():
            alpha = annotations["alpha"]
            annotationstr += f"Spectral index $\\alpha$={alpha:.3f}\n"
        if "temperature" in annotations.keys():
            temperature = annotations["temperature"]
            annotationstr += f"temperature={temperature:.2E}\n"
        if "scale" in annotations.keys():
            scale = annotations["scale"]
            annotationstr += f"normalization $\\beta$={scale:.2E}\n"
        if "mjd" in annotations.keys():
            mjd = annotations["mjd"]
            annotationstr += f"MJD={mjd:.2f}\n"
        if "bolometric_luminosity" in annotations.keys():
            bolometric_luminosity = annotations["bolometric_luminosity"]
            annotationstr += f"bol. lum.={bolometric_luminosity:.2E}\n"

        if annotationstr.endswith("\n"):
            annotationstr = annotationstr[:-2]

        if not plotmag:
            annotation_location = (0.7e15, 1.5e-26)
        else:
            annotation_location = (2e4, 18.0)

        ax1.legend(
            fontsize=FONTSIZE, fancybox=True, edgecolor="black", loc=0,
        )

    plt.savefig(os.path.join(outpath, f"{mjd}.png"))
    plt.close()
