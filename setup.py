"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import os

# Get the long description from the README file
long_description = open('README.md').read()
# Get the version number from the VERSION file
version_number = open('VERSION.txt').read().strip()

setup(
    name="scresonators-fit",
    version=version_number,
    author="CU Boulder Cryogenic Quantum Testbed",
    url="https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators",
    license = 'MIT',
    description="Python library for measuring and fitting superconducting resonator data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[],
    keywords="scresonators, superconducting resonators, resonator fit",
    packages=find_packages(include=["fit_resonator"]),  # Required
    python_requires = ">=3.9",
    install_requires = [
        "attr>=0.3.*",
        "inflect>=6.0.*",
        "lmfit>=1.1.*",
        "matplotlib>=3.6.*",
        "numpy>=1.24.*",
        "pandas>=1.5.*",
        "pytest>=7.2.*",
        "scipy>=1.9.*",
        "sympy>=1.11.*",
        "scikit-rf>=0.24.*",
        "uncertainties>=3.1.*"],
    classifiers = [
        'Development Status :: Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    project_urls={
        "Bug Reports": "https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/issues",
        "Testbed": "https://www.nist.gov/programs-projects/boulder-cryogenic-quantum-testbed",
        "Contributors": "https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/blob/master/contrib/HELLO.md",
    }
)
