DESCRIPTION = "Fits SED to data (blackbody, powerlaw)"
LONG_DESCRIPTION = """ SED modeling to existing data with blackbody and powerlaw"""

DISTNAME = "modelSED"
AUTHOR = "Simeon Reusch"
MAINTAINER = "Simeon Reusch"
MAINTAINER_EMAIL = "simeon.reusch@desy.de"
URL = "https://github.com/simeonreusch/modelSED/"
LICENSE = "BSD (3-clause)"
DOWNLOAD_URL = "https://github.com/simeonreusch/modelSED/archive/v0.2-alpha.tar.gz"
VERSION = "0.2-alpha"

try:
    from setuptools import setup, find_packages

    _has_setuptools = True
except ImportError:
    from distutils.core import setup

    _has_setuptools = False


def check_dependencies():
    install_requires = []

    # Make sure dependencies exist. This is ongoing
    try:
        import astropy
    except ImportError:
        install_requires.append("astropy")

    try:
        import numpy
    except ImportError:
        install_requires.append("numpy")
    try:
        import sncosmo
    except ImportError:
        install_requires.append("sncosmo")
    try:
        import extinction
    except ImportError:
        install_requires.append("extinction")
    try:
        import pandas
    except ImportError:
        install_requires.append("pandas")
    try:
        import matplotlib
    except ImportError:
        install_requires.append("matplotlib")
    try:
        import scipy
    except ImportError:
        install_requires.append("scipy")
    try:
        import lmfit
    except ImportError:
        install_requires.append("lmfit")
    try:
        import seaborn
    except ImportError:
        install_requires.append("seaborn")
    try:
        import pysynphot
    except ImportError:
        install_requires.append("pysynphot")

    return install_requires


if __name__ == "__main__":

    install_requires = check_dependencies()

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ["modelSED"]

    setup(
        name=DISTNAME,
        author=AUTHOR,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=install_requires,
        packages=packages,
        #        scripts=["modelSED/modelsed"],
        # include_package_data=True,
        # package_data={'pysedm': ['data/*.*']},
        # package_data={'instrument_data': ['instrument_data/*']},
        package_data={
            "modelSED": [
                "instrument_data/*.json",
                "instrument_data/bandpasses/csv/*.csv",
                "instrument_data/bandpasses/dat/*.dat",
            ]
        },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )
