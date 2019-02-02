"""Common experiment classes."""
from typing import Optional, Tuple

from labrad.units import dBm, GHz, Value

import cryores.analysis.base as analysis
import cryores.instruments.base as instruments
import cryores.experiments.data.base as data


class FrequencyScan:
    def __init__(self,
                 device: instruments.VnaInstrument,
                 dataset: data.Dataset,
                 analyzer: analysis.Analyzer) -> None:

        self.device = device
        self.dataset = dataset
        self.analyzer = analyzer
        self.results = None

    def frequency_sweep(
            self,
            frequency_start: Value,
            frequency_end: Value,
            npoints: int,
            power: Value,
            averages: int=1,
            save_to: Optional[Tuple[str, str]] = None) -> None:
        """Run a simple frequency sweep."""

        results = self.device.sweep(
            frequency_start=frequency_start,
            frequency_end=frequency_end,
            npoints=npoints,
            power=power,
            averages=averages,
            log_or_linear=instruments.LogOrLinear.LOG)

        # When creating metadata, always make sure it's JSON-compatible.
        metadata = {
            'frequency_start_GHz': frequency_start[GHz],
            'frequency_end_GHz': frequency_end[GHz],
            'npoints': npoints,
            'power_dBm': power[dBm],
            'averages': averages,
            'log_or_linear': instruments.LogOrLinear.LOG.value,
        }

        # Grab metadata from the device.
        metadata['device_params'] = self.device.get_parameters()

        # Store the data.
        self.dataset.add_data(results, metadata)

        if save_to is not None:
            path, name = save_to
            self.dataset.save(path, name)

        # Pass data to analysis code.
        self.results = self.analyzer.analyze(self.dataset)

    def get_results(self) -> analysis.AnalysisResults:
        if not self.results:
            raise RuntimeError('No analysis has run.')
        return self.results
