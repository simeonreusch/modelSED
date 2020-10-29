#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json, warnings, collections
import numpy as np
import pandas as pd
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
        binned_lc_df,
        fittype: str = "powerlaw",
        redshift: float = 0,
        plot: bool = True,
        **kwargs,
    ):

        self.fittype = fittype
        self.binned_lc_df = binned_lc_df
        self.redshift = redshift
        self.plot = plot
        self.filter_wl = utilities.load_info_json("filter_wl")
        self.wavelengths = np.arange(1000, 60000, 10) * u.AA
        self.frequencies = const.c.value / (self.wavelengths.value * 1e-10) * u.Hz

    def fit_global_parameters(self, min_datapoints: int = 4, **kwargs):
        """ """
        filter_wl = utilities.load_info_json("filter_wl")

        # First, check each time bin if enough datapoints are present for each bin
        # when above threshold 'min_datapoints' add the bin
        reduced_dict = {}
        bands = set()

        df = self.binned_lc_df
        reduced_df = pd.DataFrame(
            columns=[
                "telescope_band",
                "wavelength",
                "mean_obsmjd",
                "entries",
                "mean_mag",
                "mean_mag_err",
            ]
        )

        for mean_obsmjd in df.mean_obsmjd.unique():
            _df = df.query(f"mean_obsmjd == {mean_obsmjd}")
            binsize = len(_df.telescope_band.unique())
            if binsize >= min_datapoints:
                reduced_df = reduced_df.append(_df, ignore_index=True)

        mean_flux = utilities.abmag_to_flux(reduced_df.mean_mag.values)
        mean_flux_err = utilities.abmag_err_to_flux_err(
            reduced_df.mean_mag.values, reduced_df.mean_mag_err.values
        )

        reduced_df.insert(len(reduced_df.columns), "mean_flux", mean_flux)
        reduced_df.insert(len(reduced_df.columns), "mean_flux_err", mean_flux_err)

        cmap = utilities.load_info_json("cmap")
        bands_to_fit = reduced_df.telescope_band.unique()

        data = []
        data_err = []

        for mjd in reduced_df.mean_obsmjd.unique():
            fluxes = []
            flux_errs = []
            for telescope_band in bands_to_fit:
                _df = reduced_df.query(
                    f"telescope_band == '{telescope_band}' and mean_obsmjd == '{mjd}'"
                )
                print(_df)
                flux = _df.mean_flux.values[0]
                flux_err = _df.mean_flux_err.values[0]
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

        if self.fittype == "blackbody":
            for iy, y in enumerate(data):
                fit_params.add(f"temperature_{iy+1}", value=10000, min=100, max=150000)
                fit_params.add(f"scale_{iy+1}", value=1e23, min=1e18, max=1e27)
                fit_params.add(f"extinction_av_{iy+1}", value=0, min=0, max=4)
                fit_params.add(f"extinction_rv_{iy+1}", value=3.1, min=2.5, max=3.5)

            for i in range(2, len(data) + 1, 1):
                fit_params[f"extinction_av_{i}"].expr = "extinction_av_1"
                fit_params[f"extinction_rv_{i}"].expr = "extinction_rv_1"

            fcn_kws = {"fittype": self.fittype, "redshift": self.redshift}

        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

        minimizer = Minimizer(
            self._global_minimizer,
            fit_params,
            fcn_args=(wl_observed, data, data_err),
            fcn_kws=fcn_kws,
        )

        out = minimizer.minimize(method="basinhopping")

        if "verbose" in kwargs:
            if kwargs["verbose"]:
                print(report_fit(out.params))
                print(f"reduced chisquare: {out.redchi}")

        parameters = out.params.valuesdict()

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
                if self.fittype == "blackbody":
                    spectrum = utilities.blackbody_spectrum(
                        temperature=parameters[f"temperature_{i+1}"],
                        scale=parameters[f"scale_{i+1}"],
                        redshift=self.redshift,
                        extinction_av=parameters[f"extinction_av_{i+1}"],
                        extinction_rv=parameters[f"extinction_rv_{i+1}"],
                    )
                else:
                    print(
                        "please provide a fittype (at the moment: powerlaw or blackbody)"
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
                "red_chisq": out.redchi,
            }

        if self.fittype == "blackbody":
            return {
                "extinction_av": out.params["extinction_av_1"].value,
                "extinction_av_err": out.params["extinction_av_1"].stderr,
                "extinction_rv": out.params["extinction_rv_1"].value,
                "extinction_rv_err": out.params["extinction_rv_1"].stderr,
                "red_chisq": out.redchi,
            }

        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

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
                extinction_av_err = None
                extinction_rv = None
                extinction_rv_err = None

        df = self.binned_lc_df
        mean_flux = utilities.abmag_to_flux(df.mean_mag.values)
        mean_flux_err = utilities.abmag_err_to_flux_err(
            df.mean_mag.values, df.mean_mag_err.values
        )
        df.insert(len(df.columns), "mean_flux", mean_flux)
        df.insert(len(df.columns), "mean_flux_err", mean_flux_err)

        cmap = utilities.load_info_json("cmap")
        bands_to_fit = df.telescope_band.unique()

        params = Parameters()

        if self.fittype == "powerlaw":
            params.add("scale", value=1e-14, min=1e-20, max=1e-8)
            if alpha is None:
                params.add("alpha", value=-0.9, min=-1.2, max=-0.1)
        if self.fittype == "blackbody":
            params.add("temp", value=10000, min=100, max=150000)
            params.add("scale", value=1e23, min=1e18, max=1e27)
        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

        wl_observed = np.asarray(df.wavelength.values)
        data = np.asarray(df.mean_mag.values)
        data_err = np.asarray(df.mean_mag_err.values)

        if self.fittype == "powerlaw":
            if alpha is not None:
                fcn_kws = {"alpha": alpha}
            else:
                fcn_kws = {}
            minimizer_fcn = self._powerlaw_minimizer

        if self.fittype == "blackbody":
            fcn_kws = {
                "extinction_av": extinction_av,
                "extinction_rv": extinction_rv,
                "redshift": self.redshift,
            }
            minimizer_fcn = self._blackbody_minimizer

        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

        minimizer = Minimizer(
            minimizer_fcn, params, fcn_args=(wl_observed, [data]), fcn_kws=fcn_kws,
        )

        out = minimizer.minimize(method="basinhopping")

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

        if self.fittype == "blackbody":
            spectrum, bolometric_flux_unscaled = utilities.blackbody_spectrum(
                temperature=parameters["temp"],
                scale=parameters["scale"],
                extinction_av=extinction_av,
                extinction_rv=extinction_rv,
                redshift=self.redshift,
                get_bolometric_flux=True,
            )

        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

        df, red_chisq = self._evaluate_spectrum(spectrum, df)

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
                    "mjd": df["mean_obsmjd"].values[0],
                    "scale": parameters["scale"],
                    "scale_err": out.params["scale"].stderr,
                    "reduced_chisquare": red_chisq,
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
                    "mjd": df["mean_obsmjd"].values[0],
                    "temperature": parameters["temp"],
                    "scale": parameters["scale"],
                    "reduced_chisquare": red_chisq,
                    "bolometric_luminosity": bolo_lumi,
                }
            plot.plot_sed(df, spectrum, annotations, plotmag=True)

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
                "mjd": df["mean_obsmjd"].values[0],
                "red_chisq": red_chisq,
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

        if self.fittype == "blackbody":
            return {
                "temperature": parameters["temp"],
                "temperature_err": out.params["temp"].stderr,
                "scale": parameters["scale"],
                "scale_err": out.params["scale"].stderr,
                "extinction_av": extinction_av,
                "extinction_av_err": extinction_av_err,
                "extinction_rv": extinction_rv,
                "extinction_rv_err": extinction_rv_err,
                "mjd": df["mean_obsmjd"].values[0],
                "red_chisq": red_chisq,
                "red_chisq_binfit": out.redchi,
                "luminosity_uv_optical": luminosity_uv_optical.value,
                "luminosity_uv_nir": luminosity_uv_nir.value,
                "bolometric_luminosity": bolo_lumi.value,
                "radius": radius.value,
            }

        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

    @staticmethod
    def _evaluate_spectrum(spectrum, df):

        return_df = pd.DataFrame()

        magnitudes_outdict = {}
        filter_wl = utilities.load_info_json("filter_wl")

        index = 0
        mag_obs_list = []
        mag_model_list = []
        mag_obs_err_list = []

        mags_model = []

        for index, row in df.iterrows():
            mags_model.append(
                utilities.magnitude_in_band(row["telescope_band"], spectrum)
            )
        df.insert(len(df.columns), "mag_model", mags_model)
        df.insert(len(df.columns), "residual", df.mag_model - df.mean_mag)

        # Calculate reduced chisquare
        chisquare = 0

        for index, mag_model in enumerate(list(df.mag_model.values)):
            chisquare += (mag_model - list(df.mean_mag.values)[index]) ** 2 / list(
                df.mean_mag_err.values
            )[index] ** 2

        dof = len(df.mag_model) - 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reduced_chisquare = chisquare / dof

        # Calculate residuals
        residuals = []
        for key in magnitudes_outdict.keys():
            res = magnitudes_outdict[key]["model"] - magnitudes_outdict[key]["observed"]
            residuals.append(res)

        return df, reduced_chisquare

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

        if "redshift" in kwargs.keys():
            redshift = kwargs["redshift"]
        else:
            redshift = None

        spectrum = utilities.blackbody_spectrum(
            temperature=temp,
            scale=scale,
            extinction_av=extinction_av,
            extinction_rv=extinction_rv,
            redshift=redshift,
        )

        ab_model_list = []
        flux_model_list = []

        for wl in x:
            for index, _wl in enumerate(spectrum.wave):
                if _wl > wl:
                    ab_model = utilities.flux_to_abmag(spectrum.flux[index])
                    ab_model_list.append(ab_model)
                    break
        # for i in x:
        #     ab_model = utilities.magnitude_in_band(wl_filter[i], spectrum)
        #     ab_model_list.append(ab_model)

        residual = np.asarray(ab_model_list) - np.asarray(data)
        residual = np.nan_to_num(residual)
        print(data)
        print(residual)
        if data:
            return residual
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

        for wl in x:
            # ab_model = utilities.magnitude_in_band(wl_filter[wl], spectrum)
            for index, _wl in enumerate(spectrum.wave):
                if _wl > wl:
                    ab_model = utilities.flux_to_abmag(spectrum.flux[index])
                    break
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
                    for index, _wl in enumerate(spectrum.wave):
                        if _wl > wl:
                            ab_model = utilities.flux_to_abmag(spectrum.flux[index])
                            flux = utilities.abmag_to_flux(ab_model)
                            fluxes.append(flux)
                            break
                # for wl in x:
                #     ab_model = utilities.magnitude_in_band(wl_filter[wl], spectrum)
                #     flux = utilities.abmag_to_flux(ab_model)
                #     fluxes.append(flux)

                if data_err is None:
                    residual[i, :] = data[i, :] - fluxes
                else:
                    residual[i, :] = (data[i, :] - fluxes) / data_err[i, :]

        if fittype == "blackbody":  # assume blackbody
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
                    for index, _wl in enumerate(spectrum.wave):
                        if _wl > wl:
                            ab_model = utilities.flux_to_abmag(spectrum.flux[index])
                            flux = utilities.abmag_to_flux(ab_model)
                            fluxes.append(flux)
                            break
                    # ab_model = utilities.magnitude_in_band(wl_filter[wl], spectrum)
                    # flux = utilities.abmag_to_flux(ab_model)
                    # fluxes.append(flux)

                if data_err is None:
                    residual[i, :] = data[i, :] - fluxes
                else:
                    residual[i, :] = (data[i, :] - fluxes) / data_err[i, :]

        else:
            print("please provide a fittype (at the moment: powerlaw or blackbody)")

        print(data)
        print(residual)
        flattened_residual = residual.flatten()
        mean_flattened_residual = np.abs(np.mean(flattened_residual))

        print(f"mean residual = {mean_flattened_residual}")
        return flattened_residual
