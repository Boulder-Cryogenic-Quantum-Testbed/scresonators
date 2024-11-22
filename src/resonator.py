import os
import numpy as np
from file_io import FileIO

from preprocessor import DataProcessor
from fit import Fitter
from fit_methods.factory import create_fit_method


class Resonator:

    def __init__(self, file_path, data_columns=["Frequency [Hz]", "Magnitude [dB]", "Phase [deg]"], 
                 preproces_method='circle', fit_method_name='DCM'):

        self.file_path = file_path
        self.data_columns = data_columns
        self.freqs, self.dB_amps, self.phase_deg = self.load_data()

        self.phase_rad = np.unwrap(np.deg2rad(self.phase_deg))
        self.linear_amps = 10 ** (self.dB_amps / 20)
        self.S21_data = self.linear_amps * np.exp(1j * self.phase_rad)   # S21 = magnitude * e^(i * phase)

        os_path = os.path.split(self.file_path)
        self.dir = os_path[0]
        self.file_name = os_path[1]

        self.preprocess_method = preproces_method
        self.data_processor = DataProcessor(self, normalize_pts=10, preprocess_method=self.preprocess_method) 

        self.fit_method_name = fit_method_name
        self.fit_method = create_fit_method(self.fit_method_name, 
                               MC_iteration=5, 
                               MC_rounds=100, 
                               MC_weight='no', 
                               MC_weightvalue=2, 
                               MC_fix=[], 
                               MC_step_const=0.6, 
                               manual_init=None, 
                               vary=None)

    def load_data(self):
        
        file_io = FileIO(self.file_path, self.data_columns)
        data_list = file_io.load_csv()
        freqs = data_list[0]
        dB_amps = data_list[1]
        phases = data_list[2]
        return freqs, dB_amps, phases   
    
    def initialize_data_processor(self, normalize_pts, preprocess_method):

        self.processor = DataProcessor(self, normalize_pts=normalize_pts, preprocess_method=preprocess_method) 
        
    def initialize_fit_method(self, 
                            fit_name: str = None,
                            MC_iteration: int = 5,
                            MC_rounds: int = 100,
                            MC_weight: str = 'no',
                            MC_weightvalue: int = 2,
                            MC_fix: list = [],
                            MC_step_const: float = 0.6,
                            manual_init: list = None,
                            vary: list = None):
        
        if fit_name is None:
            fit_name = self.fit_method_name

        kwargs = {
            'MC_iteration': MC_iteration,
            'MC_rounds': MC_rounds,
            'MC_weight': MC_weight,
            'MC_weightvalue': MC_weightvalue,
            'MC_fix': MC_fix,
            'MC_step_const': MC_step_const,
            'manual_init': manual_init,
            'vary': vary
        }
        
        self.fit_method = create_fit_method(fit_name, **kwargs)
    
    def fit(self):

        fitter = Fitter(self.fit_method, self.data_processor)
        output = fitter.fit()

        return output
    
    def save_fig(self):
        """
        To be implemented
        """

        return 0