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
import sncosmo_spectral_v13
import utilities, plot
from lmfit import Model, Parameters, Minimizer, report_fit, minimize


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

    def fit_global_parameters(
        self, magnitudes: dict, fittype: str, min_datapoints: int = 4, **kwargs
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
            if nr_entries > min_datapoints + 1:  # because mjd is not a datapoint
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
            if nr_entries < 6:
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

        if fittype == "powerlaw":
            for iy, y in enumerate(data):
                fit_params.add(f"alpha_{iy+1}", value=-0.9, min=-1.5, max=-0.6)
                fit_params.add(f"scale_{iy+1}", value=1e-14, min=1e-15, max=1e-12)

            for i in range(2, len(data) + 1, 1):
                fit_params[f"alpha_{i}"].expr = "alpha_1"

        else:
            for iy, y in enumerate(data):
                fit_params.add(f"temperature_{iy+1}", value=80000, min=5000, max=15000)
                fit_params.add(f"scale_{iy+1}", value=1e23, min=1e20, max=1e25)
                fit_params.add(f"extinction_av_{iy+1}", value=1.7, min=1, max=4)
                fit_params.add(f"extinction_rv_{iy+1}", value=3.1, min=1, max=4)

            for i in range(2, len(data) + 1, 1):
                fit_params[f"extinction_av_{i}"].expr = "extinction_av_1"
                fit_params[f"extinction_rv_{i}"].expr = "extinction_rv_1"

        minimizer = Minimizer(
            self._global_minimizer,
            fit_params,
            fcn_args=(wl_observed, data, data_err),
            fcn_kws={"fittype": fittype, "redshift": self.redshift},
        )

        out = minimizer.minimize()
        report_fit(out.params)
        parameters = out.params.valuesdict()
        print(parameters)

        if self.plot:

            for i in range(len(data)):

                flux_data = data[i]
                flux_data_err = data_err[i]
                freq_observed = const.c.value / (np.array(wl_observed) * 1e-10)
                if fittype == "powerlaw":
                    spectrum = utilities.powerlaw_spectrum(
                        alpha=parameters[f"alpha_{i+1}"],
                        scale=parameters[f"scale_{i+1}"],
                        redshift=None,
                    )
                plt.figure(figsize=(8, 0.75 * 8), dpi=300)
                ax1 = plt.subplot(111)
                ax1.set_ylim([2e-28, 4e-27])
                ax1.set_xlim([3.5e14, 2e15])
                plt.xscale("log")
                plt.yscale("log")
                ax1.errorbar(freq_observed, flux_data, flux_data_err, fmt=".")
                ax1.plot(utilities.lambda_to_nu(np.array(spectrum.wave)), spectrum.flux)
                plt.savefig(f"test_{i+1}.png")
                plt.close()

        if fittype == "powerlaw":
            return {"alpha": parameters["alpha_1"]}
        else:
            return {
                "extinction_av": parameters["extinction_av_1"],
                "extinction_rv": parameters["extinction_rv_1"],
            }

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
                    alpha=alpha, scale=scale, redshift=None
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

    def fit_bin_powerlaw(self, **kwargs):
        """ """
        # magnitudes_outdict = {}
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

    def fit_bin_blackbody(self, **kwargs):
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
            print(np.asarray(flux_list) - np.asarray(data))

            if data:
                return np.asarray(flux_list) - np.asarray(data)
            else:
                return flux_list

        if data:
            return np.asarray(ab_model_list) - np.asarray(data)
        else:
            return ab_model_list


# GRAVEYARD

# magnitudes = {}
# index = 0
# for mjd in fit_dict.keys():
#     for band in fit_dict[mjd]:
#         fit_dict[mjd][band].append(mjd)
#     magnitudes.update({index: fit_dict[mjd]})
#     index += 1

# for i in range(1, len(data)+1, 1):
#     alpha = parameters[f"alpha_{i}"]
#     scale = parameters[f"scale_{i}"]

#     powerlaw_nu = self.frequencies ** alpha * u.erg / u.cm ** 2 / u.s * scale
#     spectrum = sncosmo_spectral_v13.Spectrum(
#         wave=self.wavelengths, flux=powerlaw_nu, unit=utilities.FNU
#     )

#     spectrum_evaluated = self._evaluate_spectrum(spectrum, magnitudes[i-1])

#     magnitudes_outdict = spectrum_evaluated[0]
#     reduced_chisquare = spectrum_evaluated[1]
#     residuals = spectrum_evaluated[2]

#     print(magnitudes_outdict)

#     if self.plot:
#         annotations = {
#             "alpha": alpha,
#             "scale": scale,
#         }

#         plot.plot_sed(magnitudes_outdict, spectrum, annotations)


# for band in data:
#     wavelength = filter_wl[band]
#     freq = const.c.value / (wavelength * 1e-10)
#     mean_flux = np.mean(data[band]["flux"])
#     mean_flux_err = np.sqrt(np.sum(np.asarray(data[band]["flux_err"]) ** 2)) / len(data[band]["flux_err"])
#     mean_fluxes.append(mean_flux)
#     mean_flux_errs.append(mean_flux_err)
#     wavelengths.append(wavelength)
#     freqs.append(freq)


# if self.plot:

#     plt.figure(figsize=(8, 0.75 * 8), dpi=300)
#     ax1 = plt.subplot(111)
#     plt.xscale("log")
#     ax1.errorbar(freqs, mean_fluxes, mean_flux_errs, fmt=".")
#     plt.savefig("test.png")

# if fittype == "powerlaw":
#     params = Parameters()
#     params.add("alpha", value=-1, min=-10, max=0)
#     params.add("scale", value=1e15, min=1e10, max=1e23)

#     minimizer = Minimizer(
#         self._powerlaw_minimizer,
#         params,
#         fcn_args=(wavelengths, mean_fluxes),
#         fcn_kws={"flux": True},
#     )
#     result = minimizer.minimize()
#     parameters = result.params.valuesdict()

#     alpha = parameters["alpha"]
#     scale = parameters["scale"]

#     print(f"alpha = {alpha}")
#     print(f"scale = {scale}")

#     fluxes_data = mean_fluxes
#     mags_data = []
#     fluxes_model = []
#     mags_model = []
#     fluxes_spectrum = []
#     mags_spectrum = []

#     spectrum = utilities.powerlaw_spectrum(alpha=alpha, scale=scale, redshift=None)

#     for index, wl in enumerate(spectrum.wave):
#         flux_spectrum = spectrum.flux[index]
#         mag_spectrum = utilities.flux_to_abmag(flux_spectrum)
#         fluxes_spectrum.append(flux_spectrum)
#         mags_spectrum.append(mag_spectrum)

#     for flux in mean_fluxes:
#         mags_data.append(utilities.flux_to_abmag(flux))

#     for band in data:
#         mag_model = utilities.magnitude_in_band(band, spectrum)
#         mags_model.append(mag_model)
#         fluxes_model.append(utilities.abmag_to_flux(mag_model))


#     wls, frequencies = utilities.get_wavelengths_and_frequencies()
#     powerlaw_nu = frequencies ** alpha * u.erg / u.cm ** 2 / u.s * scale
#     spectrum = sncosmo_spectral_v13.Spectrum(
#         wave=wls, flux=powerlaw_nu, unit=utilities.FNU
#     )


#     mags_data = utilities.flux_to_abmag(fluxes_data)
#     mags_model = utilities.flux_to_abmag(fluxes_model)

#     frequencies = utilities.lambda_to_nu(spectrum.wave) * u.Hz
#     spectrum_mags = []
#     for index, wl in enumerate(spectrum.wave):
#         spectrum_mag = utilities.flux_to_abmag(spectrum.flux[index])
#         spectrum_mags.append(spectrum_mag)

#     print(len(fluxes_model))
#     print(len(wavelengths))

#     if self.plot:
#         frequencies = utilities.lambda_to_nu(spectrum.wave)

#         plt.figure(figsize=(8, 0.75 * 8), dpi=300)
#         ax1 = plt.subplot(111)
#         plt.xscale("log")
#         ax1.scatter(wavelengths, mean_fluxes, marker=".")
#         ax1.scatter(wavelengths, fluxes_model, marker=".", color="red")
#         ax1.plot(spectrum.wave, spectrum.flux, color="black")
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.savefig("test2.png")


# ### Now we normalize everything to one band
# nr_bins = len(reduced_dict)

# # See if there is a band present in each datapoint
# count_occurences = {}
# for band in bands:
#     i = 0
#     for index, value in enumerate(reduced_dict.values()):
#         if band in value:
#             i += 1
#     count_occurences.update({band: i})

# to_delete = []
# for entry in count_occurences:
#     if count_occurences[entry] < nr_bins:
#         to_delete.append(entry)

# for entry in to_delete:
#     del count_occurences[entry]

# if len(count_occurences) == 0:
#     raise ValueError("There is no single band present in all bins. Normalization not possible")

# # Now choose the normalization baseline: band with smallest mean error
# magnitude_error = {}
# for band in count_occurences.keys():
#     mag_errs = []
#     for index, mjd in enumerate(reduced_dict):
#         mag, mag_err = reduced_dict[mjd][band]
#         mag_errs.append(mag_err)
#     mean_mag_err = np.sqrt(np.sum(np.asarray(mag_errs) ** 2)) / len(mag_errs)
#     magnitude_error.update({band: mean_mag_err})

# normalization_band = min(magnitude_error, key=magnitude_error.get)

# print(f"{normalization_band} has the smallest error, will normalize with respect to this band.")

# normalized_magnitudes = {}

# fit_dict = {}
# for index, mjd in enumerate(reduced_dict):
#     temp_dict = {}
#     for entry in reduced_dict[mjd]:
#         mag = reduced_dict[mjd][entry][0]
#         mag_reference = reduced_dict[mjd][normalization_band][0]
#         mag_err = reduced_dict[mjd][entry][1]
#         normalized_mag = mag/mag_reference
#         normalized_mag_err = mag_err/mag_reference
#         normalized_flux = utilities.abmag_to_flux(mag)/utilities.abmag_to_flux(mag_reference)
#         normalized_flux_err = utilities.abmag_err_to_flux_err(mag, mag_err)/utilities.abmag_to_flux(mag_reference)
#         temp_dict.update({entry: [normalized_flux, normalized_flux_err]})
#     fit_dict.update({mjd: temp_dict})

# reformatted_dict = collections.defaultdict(list)
# test = collections.defaultdict(dict)

# for mjd, entry in fit_dict.items():
#     for band, fluxes in entry.items():
#         fluxes.append(mjd)
#         reformatted_dict[band].append(fluxes)

# data = {}

# for band, value in reformatted_dict.items():
#     fluxes = []
#     flux_errs = []
#     mjds = []
#     for item in value:
#         fluxes.append(item[0])
#         flux_errs.append(item[1])
#         mjds.append(item[2])
#     data.update({band: {"mjd": mjds, "flux": fluxes, "flux_err": flux_errs}})
