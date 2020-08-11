#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json, warnings
import numpy as np
import astropy.units as u
from astropy import constants as const
import sncosmo
from extinction import ccm89, apply, remove
from astropy.cosmology import Planck15 as cosmo
from astropy.modeling.models import BlackBody
import sncosmo_spectral_v13

FNU = u.erg / (u.cm ** 2 * u.s * u.Hz)
FLAM = u.erg / (u.cm ** 2 * u.s * u.AA)

INSTRUMENT_DATA_DIR = "instrument_data"


def flux_to_abmag(fluxnu):
    return (-2.5 * np.log10(np.asarray(fluxnu))) - 48.585


def flux_err_to_abmag_err(fluxnu, fluxerr_nu):
    return 1.08574 / fluxnu * fluxerr_nu


def abmag_to_flux(abmag):
    return np.power(10, (-(np.asarray(abmag) + 48.585) / 2.5))


def abmag_err_to_flux_err(abmag, abmag_err):
    return 3.39059e-20 * np.exp(-0.921034 * abmag) * abmag_err


def lambda_to_nu(wavelength):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nu_value = const.c.value / (wavelength * 1e-10)  # Hz
    return nu_value


def nu_to_lambda(nu):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lambda_value = const.c.value / (nu * 1e-10)  # Angstrom
    return lambda_value


def flux_nu_to_lambda(fluxnu, wav):
    return np.asarray(fluxnu) * 2.99792458e18 / np.asarray(wav) ** 2 * FLAM


def flux_lambda_to_nu(fluxlambda, wav):
    return np.asarray(fluxlambda) * 3.33564095e-19 * np.asarray(wav) ** 2 * FNU


def magnitude_in_band(band: str, spectrum):
    """ """
    bandpassfiles = load_info_json("bandpassfiles")
    zpbandfluxnames = load_info_json("zpbandfluxnames")
    additional_bands = load_info_json("additional_bands")

    for bandname in additional_bands.keys():
        fname = additional_bands[bandname]
        b = np.loadtxt(fname)
        bandpass = sncosmo.Bandpass(b[:, 0], b[:, 1] / 50, name=bandname)
        sncosmo.registry.register(bandpass, force=True)

    bp = sncosmo_spectral_v13.read_bandpass(bandpassfiles[band])
    ab = sncosmo.get_magsystem("ab")
    zp_flux = ab.zpbandflux(zpbandfluxnames[band])
    bandflux = spectrum.bandflux(bp) / zp_flux
    mag = -2.5 * np.log10(bandflux)

    return mag


def calculate_luminosity(spectrum, wl_min: float, wl_max: float, redshift: float):
    """ """
    full_spectrum = spectrum
    full_wavelength = full_spectrum._wave
    full_flux = full_spectrum._flux

    masked_wl = np.ma.masked_outside(full_wavelength, wl_min, wl_max)
    mask = np.ma.getmask(masked_wl)

    cut_wl = np.ma.compressed(masked_wl) * u.AA
    cut_flux = np.ma.compressed(np.ma.masked_where(mask, full_flux))
    cut_freq = const.c.value / (cut_wl * 1e-10) * u.Hz

    d = cosmo.luminosity_distance(redshift)
    d = d.to(u.cm)

    flux = np.trapz(cut_flux, cut_freq) * u.erg / u.cm ** 2 / u.s
    luminosity = np.abs(flux * 4 * np.pi * d ** 2)

    return luminosity


def calculate_bolometric_luminosity(
    bolometric_flux: float, temperature: float, scale: float, redshift: float
):
    d = cosmo.luminosity_distance(redshift)
    d = d.to(u.m)

    a = scale
    radius = np.sqrt(d ** 2 / a)
    # bolometric_flux_unscaled = row.bolometric_flux_unscaled * u.erg / (u.cm**2 * u.s)
    # radius_cm = radius/100
    # luminosity = 4 * np.pi * bolometric_flux_unscaled * radius_cm**2
    luminosity_watt = (
        const.sigma_sb * (temperature * u.K) ** 4 * 4 * np.pi * (radius ** 2)
    )
    luminosity = luminosity_watt.to(u.erg / u.s)

    return luminosity, radius


def powerlaw_spectrum(
    alpha: float, scale: float, redshift: float = None,
):
    """ """
    wavelengths, frequencies = get_wavelengths_and_frequencies()
    if scale is None:
        powerlaw_nu = frequencies ** alpha * u.erg / u.cm ** 2 / u.s
    else:
        powerlaw_nu = frequencies ** alpha * u.erg / u.cm ** 2 / u.s * scale

    spectrum_unreddened = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=powerlaw_nu, unit=FNU
    )

    if redshift is not None:
        spectrum_unreddened.z = 0
        spectrum_unreddened_redshifted = spectrum_unreddened.redshifted_to(
            redshift, cosmo=cosmo
        )
        outspectrum = spectrum_unreddened_redshifted

    else:
        outspectrum = spectrum_unreddened

    return outspectrum


def blackbody_spectrum(
    temperature: float,
    scale: float,
    redshift: float = None,
    extinction_av: float = None,
    extinction_rv: float = None,
    get_bolometric_flux: bool = False,
):
    """ """
    wavelengths, frequencies = get_wavelengths_and_frequencies()
    scale_lambda = 1 * FLAM / u.sr
    scale_nu = 1 * FNU / u.sr
    # Blackbody of scale 1
    bb_nu = BlackBody(temperature=temperature * u.K, scale=scale_nu)
    flux_nu_unscaled = bb_nu(wavelengths) * u.sr
    flux_nu = flux_nu_unscaled / scale

    bolometric_flux_unscaled = bb_nu.bolometric_flux.value

    flux_lambda = flux_nu_to_lambda(flux_nu, wavelengths)

    if extinction_av is not None:
        flux_lambda_reddened = apply(
            ccm89(np.asarray(wavelengths), extinction_av, extinction_rv),
            np.asarray(flux_lambda),
        )
        flux_nu_reddened = flux_lambda_to_nu(flux_lambda_reddened, wavelengths)
        spectrum_reddened = sncosmo_spectral_v13.Spectrum(
            wave=wavelengths, flux=flux_nu_reddened, unit=FNU
        )

    spectrum_unreddened = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu, unit=FNU
    )

    if redshift is not None:
        if extinction_av is not None:
            spectrum_reddened.z = 0
            spectrum_reddened_redshifted = spectrum_reddened.redshifted_to(
                redshift, cosmo=cosmo
            )
            outspectrum = spectrum_reddened_redshifted
        else:
            spectrum_unreddened.z = 0
            spectrum_unreddened_redshifted = spectrum_unreddened.redshifted_to(
                redshift, cosmo=cosmo
            )
            outspectrum = spectrum_unreddened_redshifted

    else:
        if extinction_av is not None:
            outspectrum = spectrum_reddened
        else:
            outspectrum = spectrum_unreddened

    if get_bolometric_flux:
        return outspectrum, bolometric_flux_unscaled
    else:
        return outspectrum


def get_wavelengths_and_frequencies():
    wavelengths = np.arange(1000, 60000, 10) * u.AA
    frequencies = const.c.value / (wavelengths.value * 1e-10) * u.Hz
    return wavelengths, frequencies


def load_info_json(filename: str):
    with open(os.path.join(INSTRUMENT_DATA_DIR, filename + ".json")) as json_file:
        outfile = json.load(json_file)
    return outfile
