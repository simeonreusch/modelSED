#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json, warnings
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.optimize import curve_fit, OptimizeWarning
from extinction import ccm89, apply, remove
import sncosmo_spectral_v13
import utilities, plot

class FitSpectrum():
    """ """

    def __init__(
        self,
        magnitudes: dict,
        redshift: float = 0,
        plot: bool = True,
        **kwargs,
    ):

        self.magnitudes = magnitudes
        self.redshift = redshift
        self.plot = plot
        self.filter_wl = utilities.load_info_json("filter_wl")
        self.wavelengths = np.arange(1000, 60000, 10) * u.AA
        self.frequencies = const.c.value/(self.wavelengths.value*1E-10) * u.Hz

    def fit_powerlaw(self, **kwargs):
        """ """

        magnitudes_outdict = {}
        flux_observed = []
        flux_err_observed = []
        freq = []
        wl_observed = []

        for key in self.magnitudes.keys():
            if key != "mjd":
                mag = self.magnitudes[key][0]
                mag_err = self.magnitudes[key][1]
                flux_observed.append(utilities.abmag_to_flux(mag))
                flux_err_observed.append(utilities.abmag_err_to_flux_err(mag, mag_err))
                freq.append(const.c.value/(self.filter_wl[key]*1E-10))
                wl_observed.append(self.filter_wl[key])

        flux_dereddened = remove(ccm89(np.asarray(wl_observed), 3.1, 1.7), np.asarray(flux_observed))

        if "alpha" in kwargs.keys():
            #print("\nalpha given")
            alpha = kwargs["alpha"]
            def func_powerlaw(x, b):
                return x**(alpha) * b
        else:
            #print("\nalpha not given, will be fitted")
            def func_powerlaw(x, a, b):
                return x**(-a) * b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizeWarning)
            popt, pcov = curve_fit(func_powerlaw, freq, flux_observed, maxfev=2000, sigma=flux_err_observed)

        if "alpha" in kwargs.keys():
            scale = popt[0]
        else:
            alpha = -popt[0]
            scale = popt[1]


        powerlaw_nu = self.frequencies**alpha *u.erg / u.cm**2 / u.s * scale
        spectrum = sncosmo_spectral_v13.Spectrum(wave=self.wavelengths, flux=powerlaw_nu, unit=utilities.FNU)

        index = 0
        mag_obs_list = []
        mag_model_list = []
        mag_obs_err_list = []

        for key in self.magnitudes.keys():
            if key != "mjd":
                mag = self.magnitudes[key][0]
                mag_err = self.magnitudes[key][1]
                wl = self.filter_wl[key]
                flux_d = flux_dereddened[index]
                mag_d =  utilities.flux_to_abmag(flux_d)
                mag_model = utilities.magnitude_in_band(key, spectrum)
                temp = {key: {"observed": mag, "observed_err": mag_err, "observed_dereddened": mag_d, "model": mag_model, "wavelength": self.filter_wl[key], "frequency": const.c.value/(self.filter_wl[key]*1E-10)}}
                magnitudes_outdict.update(temp)
                index += 1
                mag_obs_list.append(mag)
                mag_model_list.append(mag_model)
                mag_obs_err_list.append(mag_err)

        chisquare = 0
        for index, mag_model in enumerate(mag_model_list):
            chisquare += ((mag_model - mag_obs_list[index])**2 / mag_obs_err_list[index]**2)

        dof = len(mag_model_list) - 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced_chisquare = chisquare / dof

        if self.plot:
            annotations = {"mjd": self.magnitudes["mjd"], "alpha": alpha, "scale": scale, "reduced_chisquare": reduced_chisquare}
            plot.plot_sed(magnitudes_outdict, spectrum, annotations)

        luminosity_uv_optical = utilities.calculate_luminosity(spectrum, self.filter_wl["Swift_UVW2"], self.filter_wl["P48+ZTF_i"], self.redshift)
        luminosity_uv_nir = utilities.calculate_luminosity(spectrum, self.filter_wl["Swift_UVW2"], self.filter_wl["P200_Ks"], self.redshift)

        return {"alpha": alpha, "scale": scale, "mjd": self.magnitudes["mjd"],"red_chisq": reduced_chisquare, "luminosity_uv_optical": luminosity_uv_optical.value, "luminosity_uv_nir": luminosity_uv_nir.value}
        
    def fit_blackbody():
        """ """
