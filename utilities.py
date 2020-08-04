#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, json, warnings
import numpy as np
import astropy.units as u
from astropy import constants as const
import sncosmo
from astropy.cosmology import Planck15 as cosmo
import sncosmo_spectral_v13

FNU = u.erg / (u.cm**2 * u.s * u.Hz)
FLAM = u.erg / (u.cm**2 * u.s * u.AA)

INSTRUMENT_DATA_DIR = "instrument_data"

def flux_to_abmag(fluxnu):
    return (-2.5*np.log10(fluxnu)) - 48.585

def flux_err_to_abmag_err(fluxnu, fluxerr_nu):
    return 1.08574/fluxnu * fluxerr_nu

def abmag_to_flux(abmag):
    return np.power(10, (-(abmag + 48.585)/2.5))

def abmag_err_to_flux_err(abmag, abmag_err):
    return 3.39059E-20 * np.exp(-0.921034*abmag) * abmag_err

def lambda_to_nu(wavelength):
    return const.c.value/(wavelength*1E-10) # Hz

def nu_to_lambda(nu):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lambda_value = const.c.value/(nu*1E-10)# Angstrom
    return lambda_value

def magnitude_in_band(band, spectrum):
    """ """
    bandpassfiles = load_info_json("bandpassfiles")
    zpbandfluxnames = load_info_json("zpbandfluxnames")
    additional_bands = load_info_json("additional_bands")

    for bandname in additional_bands.keys():
        fname = additional_bands[bandname]
        b = np.loadtxt(fname)
        bandpass = sncosmo.Bandpass(b[:, 0], b[:, 1]/50, name=bandname)
        sncosmo.registry.register(bandpass, force=True)

    bp = sncosmo_spectral_v13.read_bandpass(bandpassfiles[band])
    ab = sncosmo.get_magsystem('ab')
    zp_flux = ab.zpbandflux(zpbandfluxnames[band])
    bandflux = spectrum.bandflux(bp)/zp_flux
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

    d = cosmo.luminosity_distance(redshift)
    d = d.to(u.cm)
    # wl = np.arange(wl_min, wl_max, 10) * u.AA
    # freq = np.flip(const.c.value/(wl.value*1E-10) * u.Hz)
    # powerlaw = freq**alpha *u.erg / u.cm**2 / u.s * scale
    # spectrum = sncosmo_spectral_v13.Spectrum(wave=wl, flux=powerlaw, unit=FNU)
    # flux = np.trapz(spectrum._flux, freq.value) *u.erg / u.cm**2 / u.s
    cut_freq = np.flip(const.c.value/(cut_wl.value*1E-10) * u.Hz)
    flux = np.trapz(cut_flux, cut_freq) *u.erg / u.cm**2 / u.s
    luminosity = flux * 4 * np.pi * d**2
    return luminosity

def load_info_json(filename):
    with open(os.path.join(INSTRUMENT_DATA_DIR, f"{filename}.json")) as json_file:
        outfile = json.load(json_file)
    return outfile