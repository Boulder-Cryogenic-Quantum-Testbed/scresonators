"""Adapted from @mullinski and @hung93"""
import datetime
import attrs
import attr
import os
import numpy as np

import fit_resonator.cavity_functions as ff
import fit_resonator.fit as fit


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
                 vary: bool = None,
                 preprocess_method: str = "linear"):
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
        self.preprocess_method = preprocess_method

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

class ResonatorData:
    # Simple container for data
    def __init__(self,
                 freqs: np.ndarray,
                 amps: np.ndarray,
                 phases: np.ndarray,
                 linear_amps: np.ndarray):
        self.freqs = freqs
        self.amps = amps
        self.phases = phases
        self.linear_amps = linear_amps

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
    data: ResonatorData = None
    databg: ResonatorData = None
    method_class: FitMethod = None
    name: str = ''
    date: datetime.datetime = None
    temp: float = None
    bias: float = None
    measurement: str or int or tuple or list or np.ndarray = None
    normalize: int = 10
    background: str = None
    background_array: np.ndarray = None
    plot: str = 'pdf'
    plot_extra: bool = False
    preprocess_method: str = "circle"
    power = 0

    # Store non-VNASweep forms of data passed in class construction as VNASweep objects.
    def __attrs_post_init__(self):
        if self.filepath is not None and self.data is None:
            self.from_file()

        if self.data is not None and not isinstance(self.data, ResonatorData):
            self.from_columns(self.data.T[0], self.data.T[1], self.data.T[2])

        if self.background is not None and self.databg is None:
            self.init_background(filepath=self.background)
        if self.background_array is not None and self.databg is None:
            self.init_background_array(self.background_array)

    def init_background(self, filepath=background, fscale=1):
        if self.background is None and filepath is not None:
            self.background = filepath
        self.databg = from_file(self.background, fscale)

    def init_background_array(self, bg_array=background_array):
        if self.background_array is None and bg_array is not None:
            self.background_array = bg_array
        self.databg = from_columns(self.background_array.T[0],
                self.background_array.T[1], self.background_array.T[2])

    def from_columns(self, freqs, amps=None, phases=None):
        # Allows for user to pass array variable alone
        if freqs is not None and amps is None and phases is None:
            self.data = from_columns(freqs.T[0], freqs.T[1], freqs.T[2])
        else:
            self.data = from_columns(freqs, amps, phases)

    def from_file(self, filepath=filepath, measurement=None, fscale=1):
        if self.filepath is None and filepath is not None:
            self.filepath = filepath
        if self.measurement is None and measurement is not None:
            self.measurement = measurement
        self.data = from_file(self.filepath, data_column=measurement, fscale=fscale)

    def fit_method(self,
                   method: str,
                   MC_iteration: int = 5,
                   MC_rounds=100,
                   MC_weight='no',
                   MC_weightvalue=2,
                   MC_fix=[],
                   MC_step_const=0.6,
                   manual_init: list = [],
                   vary: bool = False,
                   preprocess_method: str = preprocess_method):
        self.method_class = FitMethod(method, MC_iteration, MC_rounds,
                MC_weight, MC_weightvalue, MC_fix, MC_step_const, manual_init,
                vary, preprocess_method)

    def fit(self, plot: str = 'pdf'):
        self.plot = plot
        return fit.fit(self)

    def load_params(self, method: str, params: np.ndarray, chi):
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

            assert method in ['DCM', 'PHI', 'DCM REFLECTION', 'INV', 'CPZM'], "Wrong Method, please input: PHI, DCM, INV or CPZM"
            self.method.append(method)
            if method == 'INV':
                self.INVparams = INVparams(params, chi)
                self.compare = ff.fit_raw_compare(self.data.freqs, self.data.amps, self.INVparams.all, method)
            elif method == 'CPZM':
                self.CPZMparams = CPZMparams(params, chi)
            else:
                self.DCMparams = DCMparams(params, chi)
                self.compare = ff.fit_raw_compare(self.data.freqs, self.data.amps, self.DCMparams.all, method)
                
        else:
            if method not in self.method:
                self.method.append(method)
                if method == 'INV':
                    self.INVparams = INVparams(params, chi)
                elif method == 'CPZM':
                    self.CPZMparams = CPZMparams(params, chi)
                else:
                    self.DCMparams = DCMparams(params, chi)
            else:
                print("repeated load parameter")

    def reload_params(self, method: str, params: np.ndarray, chi):
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
                self.compare = ff.fit_raw_compare(self.data.freqs, self.data.amps, self.DCMparams.all, 'DCM')
            elif method == 'INV':

                self.INVparams = INVparams(params, chi)
                self.compare = ff.fit_raw_compare(self.data.freqs, self.data.amps, self.INVparams.all, 'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams(params, chi)
        else:
            print('no')

    def power_calibrate(self, p):  
        # optional input calibration: linear function of input S21(frequency) to sample cables, meas. at room temp.

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

def from_columns(freqs, amps, phases):
    """Load data from columns provided individually."""
    linear_amps = 10 ** (amps / 20)
    return ResonatorData(freqs=freqs, amps=amps, phases=phases, linear_amps=linear_amps)

def from_file(filepath, data_column=None, fscale=1):
    if data_column is not None:
        s_col = data_column
    else:
        s_col = 1
    filename, extension = os.path.splitext(filepath)
    if extension.startswith('.s') and extension.endswith('p'):
        try:
            snp_file = open(filepath, 'r')
        except OSError as e:
            print(f'ERROR {e} when opening file')
            print(f'Data file: {filepath} could not be found/read')
            quit()
        file, inline, options, frequency_units, data_format = header_parse(file=snp_file)
        freqs, amps, phases, linear_amps = data_parse(s_col, inline, frequency_units, data_format, file, options)
        if frequency_units == 'hz':
            fscale = fscale / 1e9
        elif frequency_units == 'khz':
            fscale = fscale / 1e6
        elif frequency_units == 'mhz':
            fscale = fscale / 1e3
        freqs = freqs / fscale

        return ResonatorData(freqs, amps, phases, linear_amps)
    elif 'txt' in extension or 'csv' in extension:
        try:
            txt_file = open(filepath, 'r')
            file, line, options, frequency_units, data_format = header_parse(file=txt_file)
            data_lines = []
            while line:
                if 'END' in line:
                    break
                data_lines.append(line)
                line = file.readline().strip()
            data = np.loadtxt(data_lines, delimiter=',')
        except Exception as e:
            print(f'Exception: **{e}** encountered when attempting to load data file as .txt/.csv')
            print(f'Are you using a comma as your delimiter?')
            quit()

        freqs = data.T[0] / fscale
        amps = data.T[1]
        phases = data.T[2] * np.pi / 180
        linear_amps = 10 ** (amps / 20)
        return ResonatorData(freqs, amps, phases, linear_amps)
    else:
        print(f'File extension {extension} not supported.')
        print(f'Please use .s2p, .s1p, .txt, or .csv')
        quit()

def header_parse(file):
    data_format = None
    frequency_units = None
    comment_line = ['!']
    option_line = ['#']
    metadata = ['s21', 's11', 's12', 's22']
    dformats = ['db', 'ma', 'ri']
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.']

    options = []
    inline = file.readline()
    while any(comment in inline.lower() for comment in comment_line) \
            or any(option_lead in inline for option_lead in option_line) \
            or not any(number in inline.lower()[0] for number in nums):
        if any(option_lead in inline.lower() for option_lead in option_line) \
                or any(dformat in inline.lower() for dformat in dformats) \
                or any(measure in inline.lower() for measure in metadata):
            options.append(inline)

        inline = file.readline()

    for val in options:
        if 'db' in val.lower():
            data_format = 'db'
        elif 'ma' in val.lower():
            data_format = 'ma'
        elif 'ri' in val.lower():
            data_format = 'ri'

        if 'ghz' in val.lower():
            frequency_units = 'ghz'
        elif 'khz' in val.lower():
            frequency_units = 'khz'
        elif 'mhz' in val.lower():
            frequency_units = 'mhz'
        elif 'hz' in val.lower():
            frequency_units = 'hz'

    return file, inline, options, frequency_units, data_format

def data_parse(s_col, line, frequency_units, data_format, file, options):
    row = line.split()
    data_rows = [3, 4]
    if len(row) == 0:
        print("Data not found in file.")
        quit()

    if len(row) > 3:
        # If too many rows, use info from header to pull correct column
        # s_col has potential focus column
        if isinstance(s_col, int):
            data_rows[0] = s_col
            # data_rows[1] = self.c_col + 1 VNASweep has no c_col
        elif isinstance(s_col, (tuple, list, np.ndarray)):
            data_rows[0] = s_col[0]
            data_rows[1] = s_col[1]
        elif isinstance(s_col, str):
            for metadata in options:
                if 'Measurements: ' in metadata:
                    measurements = metadata.rsplit('Measurements: ')[1].strip('.:\n').lower().split(', ')
                    idx = (measurements.index(s_col.lower()) * 2) + 1
                    data_rows[0] = idx
                    data_rows[1] = idx + 1
                    break
        else:
            print("Could not interpret which data columns to use, using default")

    freqs = np.array(float(row[0]))
    if data_format == "db":
        amps = np.array(float(row[data_rows[0]]))
        phases = np.array(float(row[data_rows[1]]))
        line = file.readline().strip()

        while line:
            row = line.split()
            freqs = np.append(freqs, float(row[0]))
            amps = np.append(amps, float(row[data_rows[0]]))
            phases = np.append(phases, float(row[data_rows[1]]))
            line = file.readline().strip()
        phases = phases * np.pi / 180
        linear_amps = 10 ** (amps / 20)

    elif data_format == "ma":
        linear_amps = np.array(float(row[data_rows[0]]))
        phases = np.array(float(row[data_rows[1]]))
        line = file.readline().strip()

        while line:
            row = line.split()
            freqs = np.append(freqs, float(row[0]))
            linear_amps = np.append(linear_amps, float(row[data_rows[0]]))
            phases = np.append(phases, float(row[data_rows[1]]))
            line = file.readline().strip()

        phases = phases * np.pi / 180
        amps = np.log10(linear_amps) * 20

    elif data_format == "ri":
        real = np.array(float(row[data_rows[0]]))
        imaginary = np.array(float(row[data_rows[1]]))
        line = file.readline().strip()

        while line:
            row = line.split()
            freqs = np.append(freqs, float(row[0]))
            real = np.append(real, float(row[data_rows[0]]))
            imaginary = np.append(imaginary, float(row[data_rows[1]]))
            line = file.readline().strip()
        linear_amps = np.absolute(real + imaginary)
        phases = np.angle(real + 1j * imaginary, deg=True)
        amps = np.log10(linear_amps) * 20

    else:
        print("Data type in file not supported. Please use DB, MA, or RI.")
        quit()

    if frequency_units == "hz":
        freqs = freqs / 10 ** 9
    elif frequency_units == "khz":
        freqs = freqs / 10 ** 6
    elif frequency_units == "mhz":
        freqs = freqs / 10 ** 3
    elif frequency_units != "ghz":
        print(
            "Units for the frequency not found. Please include units for frequency in the header of the file.")
    return freqs, amps, phases, linear_amps


@attrs.define
class DCMparams(object):  # DCM fitting results
    params: np.ndarray
    chi: float
    num_photon: float = 0

    def __attrs_post_init__(self):
        self.Qc = self.params[2]
        self.Q = self.params[1]
        Qc = self.params[2] * np.exp(1j * self.params[4])
        Qi = (self.params[1] ** -1 - abs(np.real(Qc ** -1))) ** -1
        self.ReQc = 1 / np.real(Qc ** -1)
        self.Qi = Qi
        self.chi = self.chi
        self.fc = self.params[3]
        self.phi = ((self.params[4] + np.pi) % (2 * np.pi) - np.pi) / np.pi * 180
        self.A = self.params[0]
        self.theta = self.params[5]
        self.all = self.params


@attrs.define
class INVparams(object):  # INV fitting results
    params: np.ndarray
    chi: float
    num_photon: float = 0

    def __attrs_post_init__(self):
        self.Qc = self.params[2]
        self.Qi = self.params[1]
        Q = 1 / (self.params[1] ** -1 + self.params[2] ** -1)
        self.Q = Q
        self.chi = self.chi
        self.fc = self.params[3]
        self.phi = ((self.params[4] + np.pi) % (2 * np.pi) - np.pi) / np.pi * 180
        self.A = self.params[0]
        self.theta = self.params[5]
        self.all = self.params


@attrs.define
class CPZMparams(object):
    params: np.ndarray
    chi: float
    num_photon: float = 0

    def __attrs_post_init__(self):
        self.Qc = self.params[2]
        self.Qi = self.params[1]
        self.Qa = self.params[4]
        self.chi = self.chi
        self.fc = self.params[3]
