"""Simple in-memory dataset. Use for testing only."""

import json
import os

import pandas as pd

import cryores.experiments.data.base as base


class SimpleDataset(base.Dataset):
    """Store Pandas DataFrame and metadata in memory."""

    def __init__(self):
        self.data = None
        self.metadata = None

    def add_data(self, data: pd.DataFrame, metadata: dict) -> None:
        self.data = data
        self.metadata = metadata

    def save(self, path: str, name: str):
        """Do nothing."""

    @classmethod
    def load(cls, path: str, name: str):
        raise NotImplementedError('Load not implemented for in-memory dataset')
