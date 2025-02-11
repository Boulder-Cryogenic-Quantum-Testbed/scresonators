import os
import numpy as np
from file_io import FileIO
from pathlib import Path

from preprocessor import DataProcessor
from fit import Fitter
from fit_methods.factory import create_fit_method


class Resonator:

    def __init__(self, df=None, filepath=None, fileIO_obj=None, filename=None, preprocess_method='circle', fit_method_name="DCM", verbose=False):
        
        # accept either filepath as path obj or a fileIO obj directly 
        if filepath is None and fileIO_obj is None and df is None:
            raise TypeError("Provide one of arguments 'filepath', 'fileIO', or 'df' to load data.")
        elif fileIO_obj is not None:
            self.load_data_fileIO(fileIO_obj)
        elif filepath is not None:
            self.load_data_csv(filepath)
        elif df is not None:
            self.data_df = df
            self.data_cols = df.columns
        else:  
            pass
        
        """
            load_data_FileIO and load_data_csv add the following attributes:
                self.filepath  ->  string filepath to data
                self.pathlib_path  -> pathlib.Path filepath to data
                self.data_df  ->  pd.DataFrame of data
                self.data_cols  ->  pd.DataFrame.columns in a list instead of Index
                self.fileIO  ->  original FileIO object
        """

        # prepare data and path information
        self.freqs = self.data_df[self.data_cols[0]].to_numpy()
        self.magn_dB = self.data_df[self.data_cols[1]].to_numpy()
        self.phase_rad =  self.data_df[self.data_cols[2]].to_numpy()

        self.phase_rad = np.unwrap(self.phase_rad)
        self.magn_lin = 10 ** (self.magn_dB / 20)
        self.S21_data = self.magn_lin * np.exp(1j * self.phase_rad)  # S21 = magn_lin * e^(i * phase_rad)

        if hasattr(self, "pathlib_path") is True:
            self.dir = self.pathlib_path.parent
        else:
            self.dir = Path(os.getcwd())
            
        if filename is not None:
            self.filename = filename
        else:
            print(f"Since filepath and FileIO were not given, need to provide a filename!")
            
        
            
        
        self.preprocess_method = preprocess_method
        self.data_processor = DataProcessor(self, normalize_pts=10, preprocess_method=self.preprocess_method) 


        # TODO: check if this is how we want to automatically create fit methods
        if hasattr(self, "fit_method") is False:
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
        
    def load_data_fileIO(self, fileIO_obj):
        """
            takes a fileIO object and grabs all the useful parts 
        """
        self.pathlib_path = fileIO_obj.pathlib_path
        self.filepath = fileIO_obj.filepath
        self.filename = self.pathlib_path.name
        self.data_df = fileIO_obj.df
        self.data_cols = list(fileIO_obj.df.columns)
        self.fileIO = fileIO_obj  # just in case we want to access it later
        
        
    def load_data_csv(self, filepath):
        """
            takes a pathlib_path and loads csv data by converting it to a FileIO object
        """
        fileIO_obj = FileIO(filepath)
        self.load_data_fileIO(fileIO_obj)
        
    
    def initialize_data_processor(self, normalize_pts, preprocess_method):

        self.processor = DataProcessor(self, normalize_pts=normalize_pts, preprocess_method=preprocess_method) 
        
        
    # TODO: we should pass a "FitMethod" object instead of just the name of the fit method
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
        self.fitter = fitter
        output = self.fitter.fit()

        return output
    
    def save_fig(self):
        """
        To be implemented
        """

        return 0