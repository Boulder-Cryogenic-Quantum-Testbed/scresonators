"""Dataset implementation using CSV for table data and JSON for metadata."""

import json
import os

import pandas as pd

import cryores.experiments.data.base as base


class CsvJsonDataset(base.Dataset):
    """Save Pandas DataFrame and JSON-compatible dictionaries as CSV and JSON.
    
    Every dataset will be specified by a name and have a corresponding .csv
    and .json file.

    """

    def __init__(self):
        self.data = None
        self.metadata = None

    def add_data(self, data: pd.DataFrame, metadata: dict) -> None:
        self.data = data
        self.metadata = metadata

    def save(self, path: str, name: str):
        if self.data is None:
            raise ValueError('No data specified.')
        if self.metadata is None:
            raise ValueError('No metadata specified.')

        csv_filename = self._filename(path, name, 'csv')
        json_filename = self._filename(path, name, 'json')

        try:
            with open(json_filename, 'w') as json_f:
                json.dump(self.metadata, json.f)
        except ValueError:
            print('Unable to encode the metadata as JSON!')
            raise

        self.data.to_csv(csv_filename)

    @classmethod
    def load(cls, path: str, name: str):
        csv_filename = self._filename(path, name, 'csv')
        json_filename = self._filename(path, name, 'json')

        data = pd.read_csv(csv_filename)
        with open(json_filename, 'r') as json_f:
            metadata = json.load(json_f)

        return cls(data=data, metadata=metadata)

    @staticmethod
    def _filename(path: str, name: str, ext: str) -> str:
        return os.path.join(path, '{}.{}'.format(name, ext))
