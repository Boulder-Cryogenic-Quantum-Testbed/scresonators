import abc
from typing import NamedTuple

import pandas as pd
from labrad.units import dBm, GHz, Value

import cryores.experiments.data.base as data

class AnalysisResults(
        NamedTuple('AnalysisResults', [
            ('lowest_resonance', Value),
        ])):
    """Container for results of analysis."""


class Analyzer(abc.ABC):
    """Analysis code that accepts a Dataset and returns certain parameters."""

    @abc.abstractmethod
    def analyze(self, dataset: data.Dataset) -> AnalysisResults:
        pass
