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
from . import sncosmo_spectral_v13

FNU = u.erg / (u.cm ** 2 * u.s * u.Hz)
FLAM = u.erg / (u.cm ** 2 * u.s * u.AA)

CURRENT_FILE_DIR = os.path.dirname(__file__)
INSTRUMENT_DATA_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "instrument_data"))


def flux_to_abmag(flux_nu, flux_nu_zp=48.585):
    flux_nu = np.asarray(flux_nu, dtype=float)
    flux_nu_zp = np.asarray(flux_nu_zp, dtype=float)
    return (-2.5 * np.log10(np.asarray(flux_nu))) - flux_nu_zp


def flux_err_to_abmag_err(flux_nu, flux_nu_err):
    return 1.08574 / flux_nu * flux_nu_err


def abmag_to_flux(abmag, magzp=48.585):
    magzp = np.asarray(magzp, dtype=float)
    abmag = np.asarray(abmag, dtype=float)
    flux = np.power(10, (-(abmag + magzp) / 2.5))
    return flux


def flux_density_to_flux(wl, flux_density, flux_density_err=None):
    """ Convert flux density in erg/s cm^2 for given wavelength
        in Angstrom
    """
    nu = const.c.to("Angstrom/s").value / (wl)
    flux = flux_density * nu
    if flux_density_err is not None:
        flux_err = flux_density_err * nu
        return flux, flux_err
    else:
        return flux


def abmag_err_to_flux_err(abmag, abmag_err, magzp=None, magzp_err=None):
    abmag = np.asarray(abmag, dtype=float)
    abmag_err = np.asarray(abmag_err, dtype=float)
    if magzp is not None:
        magzp = np.asarray(magzp, dtype=float)
        magzp_err = np.asarray(magzp_err, dtype=float)
    if magzp is None and magzp_err is None:
        sigma_f = 3.39059e-20 * np.exp(-0.921034 * abmag) * abmag_err
    else:
        del_f = 0.921034 * np.exp(0.921034 * (magzp - abmag))
        sigma_f = np.sqrt(del_f ** 2 * (abmag_err + magzp_err) ** 2)
    return sigma_f


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

    # for bandpassfile in bandpassfiles:
    #     full_path_file = os.path.join(CURRENT_FILE_DIR, bandpassfile)
    #     bandpassfiles.update(bandpassfile: full_path_file)

    for bandname in additional_bands.keys():
        fname = additional_bands[bandname]
        b = np.loadtxt(fname)
        if "Swift" in fname:
            bandpass = sncosmo.Bandpass(b[:, 0], b[:, 1] / 50, name=bandname)
        else:
            bandpass = sncosmo.Bandpass(b[:, 0], b[:, 1], name=bandname)
        sncosmo.registry.register(bandpass, force=True)

    bp = sncosmo_spectral_v13.read_bandpass(bandpassfiles[band])
    ab = sncosmo.get_magsystem("ab")
    zp_flux = ab.zpbandflux(zpbandfluxnames[band])
    bandflux = spectrum.bandflux(bp) / zp_flux
    mag = -2.5 * np.log10(bandflux)

    return mag


def calculate_luminosity(spectrum, wl_min: float, wl_max: float, redshift: float):
    """ """
    # First we redshift the spectrum
    spectrum.z = 0
    spectrum_redshifted = spectrum.redshifted_to(redshift, cosmo=cosmo)
    full_spectrum = spectrum_redshifted
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
    radius_cm = radius / 100
    # bolometric_flux_unscaled = row.bolometric_flux_unscaled * u.erg / (u.cm**2 * u.s)
    # radius_cm = radius/100
    # luminosity = 4 * np.pi * bolometric_flux_unscaled * radius_cm**2
    luminosity_watt = (
        const.sigma_sb * (temperature * u.K) ** 4 * 4 * np.pi * (radius ** 2)
    )
    luminosity = luminosity_watt.to(u.erg / u.s)

    return luminosity, radius_cm


def powerlaw_spectrum(
    alpha: float,
    scale: float,
    redshift: float = None,
    extinction_av: float = None,
    extinction_rv: float = None,
):
    """ """
    wavelengths, frequencies = get_wavelengths_and_frequencies()
    if scale is None:
        flux_nu = frequencies ** alpha * u.erg / u.cm ** 2 / u.s
    else:
        flux_nu = frequencies ** alpha * u.erg / u.cm ** 2 / u.s * scale

    flux_lambda = flux_nu_to_lambda(flux_nu, wavelengths)

    spectrum_unreddened = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu, unit=FNU
    )

    if extinction_av is not None:
        flux_lambda_reddened = apply(
            ccm89(np.asarray(wavelengths), extinction_av, extinction_rv),
            np.asarray(flux_lambda),
        )

        flux_nu_reddened = flux_lambda_to_nu(flux_lambda_reddened, wavelengths)
        spectrum_reddened = sncosmo_spectral_v13.Spectrum(
            wave=wavelengths, flux=flux_nu_reddened, unit=FNU
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


def broken_powerlaw_spectrum(
    alpha1: float,
    scale1: float,
    alpha2: float,
    scale2: float,
    redshift: float = None,
    extinction_av: float = None,
    extinction_rv: float = None,
):
    """ """
    wavelengths, frequencies = get_wavelengths_and_frequencies()

    if scale1 is None:
        flux_nu1 = frequencies.value ** alpha1 * u.erg / u.cm ** 2 / u.s / u.Hz
    else:
        flux_nu1 = (
            frequencies.value ** alpha1 * scale1 * (u.erg / u.cm ** 2 / u.s / u.Hz)
        )

    if scale2 is None:
        flux_nu2 = frequencies.value ** alpha2 * u.erg / u.cm ** 2 / u.s / u.Hz
    else:
        flux_nu2 = (
            frequencies.value ** alpha2 * scale2 * (u.erg / u.cm ** 2 / u.s / u.Hz)
        )

    flux_lambda1 = flux_nu_to_lambda(flux_nu1, wavelengths)
    flux_lambda2 = flux_nu_to_lambda(flux_nu2, wavelengths)

    flux_nu = flux_nu1 + flux_nu2
    flux_lambda = flux_lambda1 + flux_lambda2

    spectrum_unreddened = sncosmo_spectral_v13.Spectrum(
        wave=wavelengths, flux=flux_nu, unit=FNU
    )

    if extinction_av is not None:
        flux_lambda_reddened = apply(
            ccm89(np.asarray(wavelengths), extinction_av, extinction_rv),
            np.asarray(flux_lambda),
        )

        flux_nu_reddened = flux_lambda_to_nu(flux_lambda_reddened, wavelengths)
        spectrum_reddened = sncosmo_spectral_v13.Spectrum(
            wave=wavelengths, flux=flux_nu_reddened, unit=FNU
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


def powerlaw_error_prop(
    frequency, alpha, alpha_err, scale, scale_err, nu_or_lambda: str = "nu"
):
    """ """
    nu = frequency ** alpha * scale
    first_term = (nu * np.log(frequency)) ** 2 * alpha_err ** 2
    second_term = (frequency ** alpha) ** 2 * scale_err ** 2
    flux_err = np.sqrt(first_term + second_term)
    if nu_or_lambda == "nu":
        return flux_err * u.erg / u.cm ** 2 * u.Hz / u.s
    elif nu_or_lambda == "lambda":
        print("OOOPS, NOT IMPLEMENTED YET")
        return 0
    else:
        raise ValueError("nu_or_lambda has to be 'nu' or 'lambda'")


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
