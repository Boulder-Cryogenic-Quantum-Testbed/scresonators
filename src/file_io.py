import csv
import numpy as np
import pandas as pd

class FileIO:
    
    def __init__(self, filepath, data_columns=None):
        self.filepath = filepath
        self.data_columns = data_columns

    def load_csv(self):
        df = pd.read_csv(self.filepath, header=None, names=self.data_columns)
        var_list = []

        for col in self.data_columns:
            var_list.append(df[col].values)  

        return var_list