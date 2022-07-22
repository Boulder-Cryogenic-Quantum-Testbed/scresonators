"""Adapted from @mullinski and @hung93"""
import datetime
import attrs
import attr
import numpy as np

import fit_resonator.functions as ff
import fit_resonator.Sdata as fs


class FitMethod(object):
    """
    Container for data related to fitting method
    Args:
        method: str
            "DCM" or 'INV' or 'CPZM'

            fitting range:
                    number ->  number * FW2M (Full width at half twice min). if not =1, changes the FW2M fitting range
                    list -> from list[0] to list[1]
                    'all' -> all

        MC_iteration: int
                  Number of iteration of 1) least square fit + 2) Monte Carlo

        MC_rounds: int
               in each MC iteration, number of rounds of randomly choose parameter

        MC_weigh: str
               'no' or 'yes', weight the extract_factor fitting range, yes uses 1/|S21| weight, which we call iDCM

        MC_weightvalue: int
                    multiplication factor for weighing, such as 2 for twice the 1/|S21| weight.

        MC_fix: list of str
            'Amp','w1','theta','phi','Qc', 'Q' for DCM, 'Qi' for INV

        MC_step_const: int
                  randomly choose number in range MC_step_const*[-0.5~0.5]
                  for fitting. Exp(0.5)=1.6, and this is used for Qi,... . However, the res. frequency, theta, amplitude are usually fixed during Monte Carlo.

        find_circle: bool
                 true=> find initial guess from circle (better) or false if find from linewidth

        manual_init: None or list of 6 float number
                 manual input initial guesses
                 DCM: [amplitude, Q, Qc, freq, phi, theta]
                 INV: [amplitude, Qi, Qc, freq, phi, theta]
        vary: None or list of 6 booleans
            vary parameter in least square fit (which parameters change = true)
"""

    def __init__(self,
                 method: str,
                 MC_iteration: int = 5,
                 MC_rounds: int = 100,
                 MC_weight: str = 'no',
                 MC_weightvalue: int = 2,
                 MC_fix: list = [],
                 MC_step_const: float = 0.6,
                 manual_init: list = None,
                 vary: bool = None):
        assert method in ['DCM', 'DCM REFLECTION', 'PHI', 'INV',
                          'CPZM'], "Wrong Method, please input:PHI, DCM, INV or CPZM"
        assert (manual_init == None) or (
                type(manual_init) == list and len(manual_init) == 4), 'Wrong manual_init, None or len = 6'
        self.method = method
        if method == 'DCM':
            self.func = ff.cavity_DCM
        elif method == 'DCM REFLECTION':
            self.func = ff.cavity_DCM_REFLECTION
        elif method == 'PHI':
            self.func = ff.cavity_DCM
        elif method == 'INV':
            self.func = ff.cavity_inverse
        elif method == 'CPZM':
            self.func = ff.cavity_CPZM
        self.MC_rounds = MC_rounds
        self.MC_iteration = MC_iteration
        self.MC_weight = MC_weight
        self.MC_weightvalue = MC_weightvalue
        self.MC_step_const = MC_step_const
        self.MC_fix = MC_fix
        self.manual_init = manual_init
        self.vary = vary if vary is not None else [True] * 6

    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())

    def change_method(self, method):
        assert method in ['DCM', 'DCM REFLECTION', 'INV', 'CPZM'], "Wrong Method, DCM,INV "
        if self.method == method:
            print("Fit method does not change")
        else:
            self.method = method

            if method == 'DCM':
                self.func = ff.cavity_DCM
            if method == 'PHI':
                self.func = ff.cavity_DCM
            elif method == 'DCM REFLECTION':
                self.func = ff.cavity_DCM_REFLECTION
            elif method == 'INV':
                self.func = ff.cavity_inverse
            elif method == 'CPZM':
                self.func = ff.cavity_CPZM


@attr.define(init=True)
class Resonator:  # Object is auto-initialized with @attr annotation
    """
    Object for representing a resonator fit.
    Usage of keywords for arguments will ensure desired output.
    All data relevant to fit to be held in an instance of this object.

    Args:
        filepath (optional): path to data file you wish to fit
        data (optional): If you have your raw data in a 3 column array-like as [Frequency, Amps, Phases]
        measurement (optional): *For SNP or other files with more than 3 columns* Tuple, list, or ndarray of target indexes. e.g. measurement = [2,3]
            OR string value of target measurement. e.g. measurement="S21"
        name (optional): name of scan.
        date (optional): date of scan.
        temp (optional): temperature of scan (in Kelvin).
        bias (optional): bias present during scan (in volts?).
    Returns:
        None
        Args passed to Resonator object are stored within the class.
        Continue using class reference for further function calls.
    """

    filepath: str = None
    data: fs.VNASweep or np.ndarray = None
    method_class: FitMethod = None
    name: str = ''
    date: datetime.datetime = None
    temp: float = None
    bias: float = None
    measurement: str or int or tuple or list or np.ndarray = None
    normalize: int = 10
    background: str = None
    background_array: np.ndarray = None
    plot_extra: bool = False
    preprocess_method: str = "linear"
    fscale: float = 1e9

    # Store non-VNASweep forms of data passed in class construction as VNASweep objects.
    def __attrs_post_init__(self):
        if self.filepath is not None and self.data is None:
            self.from_file()

        if self.data is not None and not isinstance(self.data, fs.VNASweep):
            self.from_columns(self.data.T[0], self.data.T[1], self.data.T[2])

    def from_columns(self, freqs, amps=None, phases=None):
        # Allows for user to pass array variable alone
        if freqs is not None and amps is None and phases is None:
            self.data = fs.VNASweep.from_columns(freqs.T[0], freqs.T[1], freqs.T[2])
        else:
            self.data = fs.VNASweep.from_columns(freqs, amps, phases)

    def from_file(self, filepath=None, measurement=None, fscale=1e9):
        if self.filepath is None and filepath is not None:
            self.filepath = filepath
        if self.measurement is None and measurement is not None:
            self.measurement = measurement
        self.data = fs.VNASweep.from_file(self.filepath, self.measurement, fscale)

    def fit_method(self,
                   method: str,
                   MC_iteration=None,
                   MC_rounds=100,
                   MC_weight='no',
                   MC_weightvalue=2,
                   MC_fix=[],
                   MC_step_const=0.6,
                   manual_init=None,
                   vary=None,
                   preprocess_method:str = "linear"):
        if self.preprocess_method is not preprocess_method:
            self.preprocess_method = preprocess_method
        self.method_class = FitMethod(method, MC_iteration, MC_rounds, MC_weight, MC_weightvalue, MC_fix, MC_step_const,
                                      manual_init, vary, preprocess_method)

    def fit(self):
        fs.fit(self)

    def load_params(self, method: str, params: list, chi: any):
        """
        Loads model parameters for a corresponding fit technique.

        Args:
          method: One of DCM, PHI, DCM REFLECTION, INV, CPZM. Described
            in the readme.
          params: model fit parameters.
          chi: TODO(mutus) desicribe this argument. What is this?

        """
        if self.method is None:
            self.method = []
            self.fc = params[3]
            if method == 'DCM':
                self.method.append("DCM")
                self.DCMparams = DCMparams(params, chi)
                self.compare = ff.fit_raw_compare(self.freq, self.S21, self.DCMparams.all, 'DCM')
            elif method == 'PHI':
                self.method.append("PHI")
                self.DCMparams = DCMparams(params, chi)
                self.compare = ff.fit_raw_compare(self.freq, self.S21, self.DCMparams.all, 'DCM')
            if method == 'DCM REFLECTION':
                self.method.append("DCM REFLECTION")
                self.DCMparams = DCMparams(params, chi)
                self.compare = ff.fit_raw_compare(self.freq, self.S21, self.DCMparams.all, 'DCM')
            elif method == 'INV':
                self.method.append("INV")
                self.INVparams = INVparams(params, chi)
                self.compare = ff.fit_raw_compare(self.freq, self.S21, self.INVparams.all, 'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams(params, chi)
            else:
                print('Please input DCM, DCM REFLECTION, PHI, INV or CPZM')
        else:
            if method not in self.method:
                self.method.append(method)

                if method == 'DCM':
                    self.method.append("DCM")
                    self.DCMparams = DCMparams(params, chi)

                elif method == 'PHI':
                    self.method.append("PHI")
                    self.DCMparams = DCMparams(params, chi)

                elif method == 'DCM REFLECTION':
                    self.method.append("DCM REFLECTION")
                    self.DCMparams = DCMparams(params, chi)

                elif method == 'INV':
                    self.method.append("INV")
                    self.INVparams = INVparams(params, chi)

                elif method == 'CPZM':
                    self.method.append("CPZM")
                    self.CPZMparams = CPZMparams(params, chi)
            else:
                print("repeated load parameter")

    def reload_params(self, method: str, params: list, chi: any):
        """
        Reloads model parameters for a corresponding fit technique.

        Args:
          method: One of DCM, PHI, DCM REFLECTION, INV, CPZM. Described
            in the readme.
          params: model fit parameters.
          chi: TODO(mutus) desicribe this argument. What is this?

        """
        if method in self.method:
            print(self.name + ' changed params')
            self.fc = params[3]
            if method == 'DCM REFLECTION':
                self.DCMparams = DCMparams(params, chi)
                self.compare = ff.fit_raw_compare(self.freq, self.S21, self.DCMparams.all, 'DCM')
            elif method == 'INV':

                self.INVparams = INVparams(params, chi)
                self.compare = ff.fit_raw_compare(self.freq, self.S21, self.INVparams.all, 'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams(params, chi)
        else:
            print('no')

    def power_calibrate(self,
                        p):  # optional input calibration: linear function of input S21(frequency) to sample cables, meas. at room temp.

        assert (self.DCMparams is not None) or (self.INVparams is not None), 'Please load parameters first'
        p = np.array(p)
        x = self.power
        f = self.fc
        self.corrected_power = p[0] * f + p[1] * x + p[2] + x
        hbar = 1.05 * 10 ** -34
        f = 2 * np.pi * f * 10 ** 9
        p = 10 ** (self.corrected_power / 10 - 3)
        if 'DCM' in self.method:
            Q = self.DCMparams.Q
            Qc = self.DCMparams.Qc
            self.DCMparams.num_photon = 2 * p / hbar / f ** 2 * Q ** 2 / Qc  ### calculate number of photon
        if 'INV' in self.method:
            Q = self.INVparams.Q
            Qc = self.INVparams.Qc
            self.INVparams.num_photon = 2 * p / hbar / f ** 2 * Q ** 2 / Qc


@attrs.define
class DCMparams(object):  # DCM fitting results
    params: np.ndarray
    chi: float

    def __attrs_post_init__(self, params, chi):
        self.Qc = params[2]
        self.Q = params[1]
        Qc = params[2] * np.exp(1j * params[4])
        Qi = (params[1] ** -1 - abs(np.real(Qc ** -1))) ** -1
        self.ReQc = 1 / np.real(Qc ** -1)
        self.Qi = Qi
        self.chi = chi
        self.fc = params[3]
        self.phi = ((params[4] + np.pi) % (2 * np.pi) - np.pi) / np.pi * 180
        self.A = params[0]
        self.theta = params[5]
        self.all = params


@attrs.define
class INVparams(object):  # INV fitting results
    params: np.ndarray
    chi: float

    def __attrs_post_init__(self, params, chi):
        self.Qc = params[2]
        self.Qi = params[1]
        Q = 1 / (params[1] ** -1 + params[2] ** -1)
        self.Q = Q
        self.chi = chi
        self.fc = params[3]
        self.phi = ((params[4] + np.pi) % (2 * np.pi) - np.pi) / np.pi * 180
        self.A = params[0]
        self.theta = params[5]
        self.all = params


@attrs.define
class CPZMparams(object):
    params: np.ndarray
    chi: float

    def __attrs_post_init__(self, params, chi):
        self.Qc = params[2]
        self.Qi = params[1]
        self.Qa = params[4]
        self.chi = chi
        self.fc = params[3]
