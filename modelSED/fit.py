#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json, warnings, collections
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy.optimize import curve_fit, OptimizeWarning
from astropy.cosmology import Planck15 as cosmo
from . import utilities, plot, sncosmo_spectral_v13
from lmfit import Model, Parameters, Minimizer, report_fit, minimize


class FitSpectrum:
    """ """

    def __init__(
        self,
        magnitudes: dict,
        fittype: str = "powerlaw",
        redshift: float = 0,
        plot: bool = True,
        **kwargs,
    ):

        self.fittype = fittype
        self.magnitudes = magnitudes
        self.redshift = redshift
        self.plot = plot
        self.filter_wl = utilities.load_info_json("filter_wl")
        self.wavelengths = np.arange(1000, 60000, 10) * u.AA
        self.frequencies = const.c.value / (self.wavelengths.value * 1e-10) * u.Hz

    def fit_global_parameters(
        self, magnitudes: dict, min_datapoints: int = 4, **kwargs
    ):
        """ """
        filter_wl = utilities.load_info_json("filter_wl")

        # First, check each time bin if enough datapoints are present for each bin
        # when above threshold 'min_datapoints' add the bin
        i = 0
        reduced_dict = {}
        bands = set()

        for index, value in enumerate(magnitudes.values()):
            nr_entries = len(value.keys())
            if nr_entries >= min_datapoints + 1:  # because mjd is not a datapoint
                mjd = value["mjd"]
                del value["mjd"]
                reduced_dict.update({mjd: value})
                for key in value.keys():
                    bands.add(key)
                i += 1

        cmap = utilities.load_info_json("cmap")
        wavelengths = []
        freqs = []
        mean_fluxes = []
        mean_flux_errs = []

        fit_dict = {}
        bands_to_fit = set()

        for mjd in reduced_dict.keys():
            nr_entries = len(reduced_dict[mjd])
            if nr_entries < min_datapoints:
                continue
            else:
                fit_dict.update({mjd: reduced_dict[mjd]})
                for key in reduced_dict[mjd].keys():
                    bands_to_fit.add(key)

        bands_to_fit = list(bands_to_fit)

        data = []
        data_err = []
        for mjd in fit_dict.keys():
            fluxes = []
            flux_errs = []
            for band in bands_to_fit:
                mag = fit_dict[mjd][band][0]
                mag_err = fit_dict[mjd][band][1]
                flux = utilities.abmag_to_flux(mag)
                flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)
                fluxes.append(flux)
                flux_errs.append(flux_err)
            data.append(fluxes)
            data_err.append(flux_errs)

        data = np.array(data)
        data_err = np.array(data_err)

        wl_observed = []
        for band in bands_to_fit:
            wl_observed.append(filter_wl[band])

        fit_params = Parameters()

        if self.fittype == "powerlaw":
            for iy, y in enumerate(data):
                fit_params.add(f"alpha_{iy+1}", value=-0.9, min=-1.5, max=-0.6)
                fit_params.add(f"scale_{iy+1}", value=1e-14, min=1e-15, max=1e-12)

            for i in range(2, len(data) + 1, 1):
                fit_params[f"alpha_{i}"].expr = "alpha_1"

            fcn_kws = {"fittype": self.fittype}

        else:
            for iy, y in enumerate(data):
                fit_params.add(f"temperature_{iy+1}", value=30000, min=1000, max=150000)
                fit_params.add(f"scale_{iy+1}", value=1e23, min=1e20, max=1e25)
                fit_params.add(f"extinction_av_{iy+1}", value=1.7, min=0, max=3.5)
                fit_params.add(f"extinction_rv_{iy+1}", value=3.1, min=0, max=4)

            for i in range(2, len(data) + 1, 1):
                fit_params[f"extinction_av_{i}"].expr = "extinction_av_1"
                fit_params[f"extinction_rv_{i}"].expr = "extinction_rv_1"

            fcn_kws = {"fittype": self.fittype, "redshift": self.redshift}

        minimizer = Minimizer(
            self._global_minimizer,
            fit_params,
            fcn_args=(wl_observed, data, data_err),
            fcn_kws=fcn_kws,
        )

        out = minimizer.minimize()
        report_fit(out.params)
        parameters = out.params.valuesdict()

        plotmag = False

        if self.plot:

            for i in range(len(data)):

                flux_data = data[i]
                flux_data_err = data_err[i]

                if self.fittype == "powerlaw":
                    spectrum = utilities.powerlaw_spectrum(
                        alpha=parameters[f"alpha_{i+1}"],
                        scale=parameters[f"scale_{i+1}"],
                        redshift=None,
                    )
                else:
                    spectrum = utilities.blackbody_spectrum(
                        temperature=parameters[f"temperature_{i+1}"],
                        scale=parameters[f"scale_{i+1}"],
                        redshift=self.redshift,
                        extinction_av=parameters[f"extinction_av_{i+1}"],
                        extinction_rv=parameters[f"extinction_rv_{i+1}"],
                    )

                plot.plot_sed_from_flux(
                    flux=flux_data,
                    bands=bands_to_fit,
                    spectrum=spectrum,
                    fittype=self.fittype,
                    redshift=self.redshift,
                    flux_err=flux_data_err,
                    index=i,
                    plotmag=False,
                )

        if self.fittype == "powerlaw":
            return {
                "alpha": out.params["alpha_1"].value,
                "alpha_err": out.params["alpha_1"].stderr,
            }
        else:
            return {
                "extinction_av": out.params["extinction_av_1"].value,
                "extinction_av_err": out.params["extinction_av_1"].stderr,
                "extinction_rv": out.params["extinction_rv_1"].value,
                "extinction_rv_err": out.params["extinction_rv_1"].stderr,
            }

    def fit_bin(self, **kwargs):
        """ """
        flux_observed = []
        flux_err_observed = []
        freq = []
        wl_observed = []
        mags = []

        if self.fittype == "powerlaw":
            if "alpha" in kwargs.keys():
                alpha = kwargs["alpha"]
            else:
                alpha = None

            if "alpha_err" in kwargs.keys():
                alpha_err = kwargs["alpha_err"]
            else:
                alpha_err = None

        else:
            if "extinction_av" in kwargs.keys():
                extinction_av = kwargs["extinction_av"]
                extinction_rv = kwargs["extinction_rv"]
                extinction_av_err = kwargs["extinction_av_err"]
                extinction_rv_err = kwargs["extinction_rv_err"]
            else:
                extinction_av = None
                extinction_rv = None

        for key in self.magnitudes.keys():
            if key != "mjd":
                mag = self.magnitudes[key][0]
                mags.append(mag)
                mag_err = self.magnitudes[key][1]
                flux_observed.append(utilities.abmag_to_flux(mag))
                flux_err_observed.append(utilities.abmag_err_to_flux_err(mag, mag_err))
                freq.append(const.c.value / (self.filter_wl[key] * 1e-10))
                wl_observed.append(self.filter_wl[key])

        params = Parameters()

        if self.fittype == "powerlaw":
            params.add("scale", value=1e-14, min=1e-20, max=1e-8)
            if alpha is None:
                if "alpha_bound" in kwargs:
                    alpha_bound = kwargs["alpha_bound"]
                    params.add(
                        "alpha",
                        value=alpha_bound - 0.01,
                        min=-1.5,
                        max=alpha_bound + 0.1,
                    )
                else:
                    params.add("alpha", value=-0.9, min=-1.2, max=-0.1)
        else:
            params.add("temp", value=30000, min=1000, max=150000)
            params.add("scale", value=1e23, min=1e20, max=1e25)

        x = wl_observed
        data = mags

        if self.fittype == "powerlaw":
            if alpha is not None:
                fcn_kws = {"alpha": alpha}
            else:
                fcn_kws = {}
            minimizer_fcn = self._powerlaw_minimizer

        else:
            fcn_kws = {"extinction_av": extinction_av, "extinction_rv": extinction_rv}
            minimizer_fcn = self._blackbody_minimizer

        minimizer = Minimizer(
            minimizer_fcn, params, fcn_args=(x, data), fcn_kws=fcn_kws,
        )

        out = minimizer.minimize()

        if "verbose" in kwargs:
            if kwargs["verbose"]:
                print(report_fit(out.params))
                print(f"reduced chisquare: {out.redchi}")

        parameters = out.params.valuesdict()

        if self.fittype == "powerlaw":
            if alpha is not None:
                spectrum = utilities.powerlaw_spectrum(
                    alpha=alpha, scale=parameters["scale"],
                )
            else:
                spectrum = utilities.powerlaw_spectrum(
                    alpha=parameters["alpha"], scale=parameters["scale"],
                )

        else:
            spectrum, bolometric_flux_unscaled = utilities.blackbody_spectrum(
                temperature=parameters["temp"],
                scale=parameters["scale"],
                extinction_av=extinction_av,
                extinction_rv=extinction_rv,
                redshift=self.redshift,
                get_bolometric_flux=True,
            )

        spectrum_evaluated = self._evaluate_spectrum(spectrum, self.magnitudes)

        magnitudes_outdict = spectrum_evaluated[0]
        reduced_chisquare = spectrum_evaluated[1]
        residuals = spectrum_evaluated[2]

        # Calculate bolometric luminosity
        if self.fittype == "blackbody":
            bolo_lumi, radius = utilities.calculate_bolometric_luminosity(
                bolometric_flux=bolometric_flux_unscaled,
                temperature=parameters["temp"],
                scale=parameters["scale"],
                redshift=self.redshift,
            )

        if self.plot:
            if self.fittype == "powerlaw":
                annotations = {
                    "mjd": self.magnitudes["mjd"],
                    "scale": parameters["scale"],
                    "scale_err": out.params["scale"].stderr,
                    "reduced_chisquare": reduced_chisquare,
                }
                if alpha is not None:
                    annotations.update({"alpha": alpha})
                    if alpha_err is not None:
                        annotations.update({"alpha_err": alpha_err})
                else:
                    annotations.update(
                        {
                            "alpha": parameters["alpha"],
                            "alpha_err": out.params["alpha"].stderr,
                        }
                    )

            else:
                annotations = {
                    "mjd": self.magnitudes["mjd"],
                    "temperature": parameters["temp"],
                    "scale": parameters["scale"],
                    "reduced_chisquare": reduced_chisquare,
                    "bolometric_luminosity": bolo_lumi,
                }
            plot.plot_sed_from_dict(magnitudes_outdict, spectrum, annotations)

        luminosity_uv_optical = utilities.calculate_luminosity(
            spectrum,
            self.filter_wl["Swift+UVW2"],
            self.filter_wl["P48+ZTF_i"],
            self.redshift,
        )
        luminosity_uv_nir = utilities.calculate_luminosity(
            spectrum,
            self.filter_wl["Swift+UVW2"],
            self.filter_wl["P200+Ks"],
            self.redshift,
        )

        if self.fittype == "powerlaw":
            returndict = {
                "scale": parameters["scale"],
                "scale_err": out.params["scale"].stderr,
                "mjd": self.magnitudes["mjd"],
                "red_chisq": reduced_chisquare,
                "red_chisq_binfit": out.redchi,
                "luminosity_uv_optical": luminosity_uv_optical.value,
                "luminosity_uv_nir": luminosity_uv_nir.value,
            }
            if alpha is not None:
                returndict.update({"alpha": alpha})
            else:
                returndict.update({"alpha": parameters["alpha"]})
            if alpha_err is not None:
                returndict.update({"alpha_err": alpha_err})
            return returndict

        else:
            return {
                "temperature": parameters["temp"],
                "temperature_err": out.params["temp"].stderr,
                "scale": parameters["scale"],
                "scale_err": out.params["scale"].stderr,
                "extinction_av": extinction_av,
                "extinction_av_err": extinction_av_err,
                "extinction_rv": extinction_rv,
                "extinction_rv_err": extinction_rv_err,
                "mjd": self.magnitudes["mjd"],
                "red_chisq": reduced_chisquare,
                "red_chisq_binfit": out.redchi,
                "luminosity_uv_optical": luminosity_uv_optical.value,
                "luminosity_uv_nir": luminosity_uv_nir.value,
                "bolometric_luminosity": bolo_lumi.value,
                "radius": radius.value,
            }

    @staticmethod
    def _evaluate_spectrum(spectrum, magnitudes: dict = None):

        magnitudes_outdict = {}
        filter_wl = utilities.load_info_json("filter_wl")

        index = 0
        mag_obs_list = []
        mag_model_list = []
        mag_obs_err_list = []

        for key in magnitudes.keys():
            if key != "mjd":
                mag = magnitudes[key][0]
                mag_err = magnitudes[key][1]
                wl = filter_wl[key]
                mag_model = utilities.magnitude_in_band(key, spectrum)
                temp = {
                    key: {
                        "observed": mag,
                        "observed_err": mag_err,
                        "model": mag_model,
                        "wavelength": filter_wl[key],
                        "frequency": const.c.value / (filter_wl[key] * 1e-10),
                    }
                }
                magnitudes_outdict.update(temp)
                index += 1
                mag_obs_list.append(mag)
                mag_model_list.append(mag_model)
                mag_obs_err_list.append(mag_err)

        # Calculate reduced chisquare
        chisquare = 0

        for index, mag_model in enumerate(mag_model_list):
            chisquare += (mag_model - mag_obs_list[index]) ** 2 / mag_obs_err_list[
                index
            ] ** 2

        dof = len(mag_model_list) - 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced_chisquare = chisquare / dof

        # Calculate residuals
        residuals = []
        for key in magnitudes_outdict.keys():
            res = magnitudes_outdict[key]["model"] - magnitudes_outdict[key]["observed"]
            residuals.append(res)

        return magnitudes_outdict, reduced_chisquare, residuals

    @staticmethod
    def _blackbody_minimizer(params, x, data=None, **kwargs):

        filter_wl = utilities.load_info_json("filter_wl")
        wl_filter = {v: k for k, v in filter_wl.items()}
        temp = params["temp"]
        scale = params["scale"]

        extinction_av = None
        extinction_rv = None

        if "extinction_av" in kwargs.keys():
            extinction_av = kwargs["extinction_av"]
            extinction_rv = kwargs["extinction_rv"]

        redshift = 0.2666
        spectrum = utilities.blackbody_spectrum(
            temperature=temp,
            scale=scale,
            extinction_av=extinction_av,
            extinction_rv=extinction_rv,
            redshift=redshift,
        )

        ab_model_list = []
        flux_model_list = []

        for i in x:
            ab_model = utilities.magnitude_in_band(wl_filter[i], spectrum)
            ab_model_list.append(ab_model)

        if data:
            return np.asarray(ab_model_list) - np.asarray(data)
        else:
            return ab_model_list

    @staticmethod
    def _powerlaw_minimizer(params, x, data=None, **kwargs):
        """ """
        filter_wl = utilities.load_info_json("filter_wl")
        wl_filter = {v: k for k, v in filter_wl.items()}

        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        else:
            alpha = params["alpha"]

        if "scale" in params.keys():
            scale = params["scale"]
        else:
            scale = None

        if "redshift" in params.keys():
            redshift = params["redshift"]
        else:
            redshift = None

        spectrum = utilities.powerlaw_spectrum(
            alpha=alpha, scale=scale, redshift=redshift
        )

        ab_model_list = []
        flux_list = []

        for i in x:
            ab_model = utilities.magnitude_in_band(wl_filter[i], spectrum)
            flux = utilities.abmag_to_flux(ab_model)
            ab_model_list.append(ab_model)
            flux_list.append(flux)

        if "flux" in kwargs.keys():
            if data:
                return np.asarray(flux_list) - np.asarray(data)
            else:
                return flux_list

        if data:
            return np.asarray(ab_model_list) - np.asarray(data)
        else:
            return ab_model_list

    @staticmethod
    def _global_minimizer(params, x, data=None, data_err=None, **kwargs):
        """ calculate total residual for fits to several data sets held
        in a 2-D array, and modeled by the desired functions"""

        if not "fittype" in kwargs.keys():
            raise ValueError("you have to provide a fittype ('blackbody' or 'powerlaw'")

        fittype = kwargs["fittype"]

        if fittype == "blackbody" and not "redshift" in kwargs.keys():
            raise ValueError(
                "When choosing fittype=blackbody, you have to provide a redshift"
            )

        if fittype == "blackbody":
            redshift = kwargs["redshift"]

        if fittype == "powerlaw":
            if "redshift" in kwargs:
                redshift = kwargs["redshift"]
            else:
                redshift = None

        ndata, _ = data.shape
        residual = 0.0 * data[:]

        filter_wl = utilities.load_info_json("filter_wl")
        wl_filter = {v: k for k, v in filter_wl.items()}

        # make residual per data set
        if fittype == "powerlaw":
            for i in range(ndata):
                alpha = params[f"alpha_{i+1}"]
                scale = params[f"scale_{i+1}"]

                spectrum = utilities.powerlaw_spectrum(
                    alpha=alpha, scale=scale, redshift=redshift
                )

                fluxes = []
                for wl in x:
                    ab_model = utilities.magnitude_in_band(wl_filter[wl], spectrum)
                    flux = utilities.abmag_to_flux(ab_model)
                    fluxes.append(flux)
                if data_err is None:
                    residual[i, :] = data[i, :] - fluxes
                else:
                    residual[i, :] = (data[i, :] - fluxes) / data_err[i, :]

        else:  # assume blackbody
            for i in range(ndata):
                temperature = params[f"temperature_{i+1}"]
                scale = params[f"scale_{i+1}"]
                # extinction_av = 1.7
                # extinction_rv = 3.1
                extinction_av = params[f"extinction_av_{i+1}"]
                extinction_rv = params[f"extinction_rv_{i+1}"]

                spectrum = utilities.blackbody_spectrum(
                    temperature=temperature,
                    scale=scale,
                    redshift=redshift,
                    extinction_av=extinction_av,
                    extinction_rv=extinction_rv,
                )

                fluxes = []

                for wl in x:
                    ab_model = utilities.magnitude_in_band(wl_filter[wl], spectrum)
                    flux = utilities.abmag_to_flux(ab_model)
                    fluxes.append(flux)
                if data_err is None:
                    residual[i, :] = data[i, :] - fluxes
                else:
                    residual[i, :] = (data[i, :] - fluxes) / data_err[i, :]

        flattened_residual = residual.flatten()
        print(flattened_residual)
        print(f"mean residual = {np.mean(flattened_residual)}")
        return flattened_residual


def powerlaw_minimizer(params, x, data=None, **kwargs):
    """ """
    filter_wl = utilities.load_info_json("filter_wl")
    wl_filter = {v: k for k, v in filter_wl.items()}

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    else:
        alpha = params["alpha"]

    if "scale" in params.keys():
        scale = params["scale"]
    else:
        scale = None

    if "redshift" in params.keys():
        redshift = params["redshift"]
    else:
        redshift = None

    spectrum = utilities.powerlaw_spectrum(alpha=alpha, scale=scale, redshift=redshift)

    ab_model_list = []
    flux_list = []

    for i in x:
        ab_model = utilities.magnitude_in_band(wl_filter[i], spectrum)
        flux = utilities.abmag_to_flux(ab_model)
        ab_model_list.append(ab_model)
        flux_list.append(flux)

    if "flux" in kwargs.keys():
        if data:
            return np.asarray(flux_list) - np.asarray(data)
        else:
            return flux_list

    if data:
        return np.asarray(ab_model_list) - np.asarray(data)
    else:
        return ab_model_list
