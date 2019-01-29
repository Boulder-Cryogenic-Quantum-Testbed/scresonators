import abc

import pandas as pd


class Dataset(abc.ABC):
    """Collection of tabular data and key-value parameters."""

    @abc.abstractmethod
    def add_data(self, data: pd.DataFrame, metadata: dict) -> None:
        pass

    @abc.abstractmethod
    def save(self, path, name):
        """Save the dataset to the given path with the prefix name.

        The name should not include any file extension, as this dataset may
        be represented by more than one file.
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str, name: str):
        """Create a Dataset from one or more files."""
