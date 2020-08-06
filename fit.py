#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json, warnings
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.optimize import curve_fit, OptimizeWarning
from astropy.cosmology import Planck15 as cosmo
import sncosmo_spectral_v13
import utilities, plot
from lmfit import Model, Parameters, Minimizer, report_fit


class FitSpectrum:
    """ """

    def __init__(
        self, magnitudes: dict, redshift: float = 0, plot: bool = True, **kwargs,
    ):

        self.magnitudes = magnitudes
        self.redshift = redshift
        self.plot = plot
        self.filter_wl = utilities.load_info_json("filter_wl")
        self.wavelengths = np.arange(1000, 60000, 10) * u.AA
        self.frequencies = const.c.value / (self.wavelengths.value * 1e-10) * u.Hz

    def fit_powerlaw(self, bands_to_fit: list = None, **kwargs):
        """ """
        # magnitudes_outdict = {}
        flux_observed = []
        flux_err_observed = []
        freq = []
        wl_observed = []

        # if bands_to_fit is None:
        #     bands_to_fit = []
        #     for

        for key in self.magnitudes.keys():
            if key != "mjd":
                mag = self.magnitudes[key][0]
                mag_err = self.magnitudes[key][1]
                flux_observed.append(utilities.abmag_to_flux(mag))
                flux_err_observed.append(utilities.abmag_err_to_flux_err(mag, mag_err))
                freq.append(const.c.value / (self.filter_wl[key] * 1e-10))
                wl_observed.append(self.filter_wl[key])

        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]

            def func_powerlaw(x, b):
                return x ** (alpha) * b

        else:

            def func_powerlaw(x, a, b):
                return x ** (-a) * b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizeWarning)
            popt, pcov = curve_fit(
                func_powerlaw, freq, flux_observed, maxfev=2000, sigma=flux_err_observed
            )

        if "alpha" in kwargs.keys():
            scale = popt[0]
        else:
            alpha = -popt[0]
            scale = popt[1]

        powerlaw_nu = self.frequencies ** alpha * u.erg / u.cm ** 2 / u.s * scale
        spectrum = sncosmo_spectral_v13.Spectrum(
            wave=self.wavelengths, flux=powerlaw_nu, unit=utilities.FNU
        )
        # print(spectrum._flux)

        spectrum_evaluated = self._evaluate_spectrum(spectrum, self.magnitudes)

        magnitudes_outdict = spectrum_evaluated[0]
        reduced_chisquare = spectrum_evaluated[1]
        residuals = spectrum_evaluated[2]

        if self.plot:
            annotations = {
                "mjd": self.magnitudes["mjd"],
                "alpha": alpha,
                "scale": scale,
                "reduced_chisquare": reduced_chisquare,
            }
            plot.plot_sed(magnitudes_outdict, spectrum, annotations)

        luminosity_uv_optical = utilities.calculate_luminosity(
            spectrum,
            wl_min=self.filter_wl["Swift_UVW2"],
            wl_max=self.filter_wl["P48+ZTF_i"],
            redshift=self.redshift,
        )
        luminosity_uv_nir = utilities.calculate_luminosity(
            spectrum,
            wl_min=self.filter_wl["Swift_UVW2"],
            wl_max=self.filter_wl["P200_Ks"],
            redshift=self.redshift,
        )

        return {
            "alpha": alpha,
            "scale": scale,
            "mjd": self.magnitudes["mjd"],
            "red_chisq": reduced_chisquare,
            "luminosity_uv_optical": luminosity_uv_optical.value,
            "luminosity_uv_nir": luminosity_uv_nir.value,
        }

    def fit_blackbody(self, **kwargs):
        """ """
        flux_observed = []
        flux_err_observed = []
        freq = []
        wl_observed = []
        mags = []

        if "extinction_av" in kwargs.keys():
            extinction_av = kwargs["extinction_av"]
            extinction_rv = kwargs["extinction_rv"]
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

        # Now we construct the blackbody
        params = Parameters()
        params.add("temp", value=30000, min=1000, max=180000)
        params.add("scale", value=2e23, min=1e18, max=1e30)

        x = wl_observed
        data = mags

        minimizer = Minimizer(
            self._bb_minimizer,
            params,
            fcn_args=(x, data),
            fcn_kws={"extinction_av": extinction_av, "extinction_rv": extinction_rv},
        )
        result = minimizer.minimize()
        parameters = result.params.valuesdict()

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
        bolo_lumi, radius = utilities.calculate_bolometric_luminosity(
            bolometric_flux=bolometric_flux_unscaled,
            temperature=parameters["temp"],
            scale=parameters["scale"],
            redshift=self.redshift,
        )

        if self.plot:
            annotations = {
                "mjd": self.magnitudes["mjd"],
                "temperature": parameters["temp"],
                "scale": parameters["scale"],
                "reduced_chisquare": reduced_chisquare,
                "bolometric_luminosity": bolo_lumi,
            }
            plot.plot_sed(magnitudes_outdict, spectrum, annotations)

        luminosity_uv_optical = utilities.calculate_luminosity(
            spectrum,
            self.filter_wl["Swift_UVW2"],
            self.filter_wl["P48+ZTF_i"],
            self.redshift,
        )
        luminosity_uv_nir = utilities.calculate_luminosity(
            spectrum,
            self.filter_wl["Swift_UVW2"],
            self.filter_wl["P200_Ks"],
            self.redshift,
        )

        return {
            "temperature": parameters["temp"],
            "scale": parameters["scale"],
            "extinction_av": extinction_av,
            "extinction_rv": extinction_rv,
            "mjd": self.magnitudes["mjd"],
            "red_chisq": reduced_chisquare,
            "luminosity_uv_optical": luminosity_uv_optical.value,
            "luminosity_uv_nir": luminosity_uv_nir.value,
            "bolometric_luminosity": bolo_lumi.value,
            "radius": radius.value,
        }

    @staticmethod
    def _evaluate_spectrum(spectrum, magnitudes: dict):
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
    def _bb_minimizer(params, x, data=None, **kwargs):

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
        for i in x:
            ab_model = utilities.magnitude_in_band(wl_filter[i], spectrum)
            ab_model_list.append(ab_model)

        if data:
            return np.asarray(ab_model_list) - np.asarray(data)
        else:
            return ab_model_list
