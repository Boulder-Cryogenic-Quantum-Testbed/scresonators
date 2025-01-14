import csv
import numpy as np
import pandas as pd

class FileIO:
    
    def __init__(self, filepath, df_headers=None):
        self.filepath = filepath
        self.df_headers = df_headers

    def load_csv(self):
        df = pd.read_csv(self.filepath, header=None, names=self.df_headers)
        var_list = []

        for col in self.df_headers:
            var_list.append(df[col].values)  

        return var_list