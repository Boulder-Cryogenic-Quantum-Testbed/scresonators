"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

# Get the long description from the README file
long_description = open('README.md').read()
# Get the version number from the VERSION file
version_number = open('VERSION').read().strip()

setup(
    name="scresonators-fit",
    version=version_number,
    author="CU Boulder Cryogenic Quantum Testbed",
    url="https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators",
    license = 'MIT',
    description="Python library for measuring and fitting superconducting resonator data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="scresonators, superconducting resonators, resonator fit",
    packages=find_packages(include=["fit_resonator"]),  # Required
    python_requires = ">=3.9",
    install_requires=[
        "attrs>=21.0",  # Updated from "attr>=0.3.*"
        "inflect>=5.3",  # Updated from "inflect>=6.0.*"
        "lmfit>=1.0.2",  # Updated from "lmfit>=1.1.*"
        "matplotlib>=3.4",  # Updated from "matplotlib>=3.6.*"
        "numpy>=1.20",  # Updated from "numpy>=1.24.*"
        "pandas>=1.2",  # Updated from "pandas>=1.5.*"
        "pytest>=6.2",  # Updated from "pytest>=7.2.*"
        "scipy>=1.6",  # Updated from "scipy>=1.9.*"
        "sympy>=1.8",  # Updated from "sympy>=1.11.*"
        "scikit-rf>=0.17",  # Updated from "scikit-rf>=0.24.*"
        "uncertainties>=3.1",  # No change
        "GitPython>=3.1"  # Updated from "gitpython>=3.1*"
    ],
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
