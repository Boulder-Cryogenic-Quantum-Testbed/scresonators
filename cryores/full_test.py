"""Example of a full experiment flow using a fake VNA."""
import math

from labrad.units import dBm, GHz

import cryores.analysis.analysis as analysis
import cryores.instruments.fake as fake
import cryores.experiments.base as experiments
import cryores.experiments.data.simple as simple_data


def test_fake_vna_to_analysis():

    # Create the device outside of the experiment.
    vna = fake.FakeVna()

    # This allows us to specify any device-specific setup first, e.g.
    vna.set_device_parameter(1234.5)
    
    experiment = experiments.FrequencyScan(
            device=vna,
            dataset=simple_data.SimpleDataset(),
            analyzer=analysis.ResonatorMinAnalyzer())

    # Run a simple frequency sweep.
    experiment.frequency_sweep(
            frequency_start=1.0*GHz,
            frequency_end=5.0*GHz,
            npoints=20,
            power=0.5*dBm,
            )

    results = experiment.get_results()
    assert math.isclose(results.lowest_resonance, 1.0)
