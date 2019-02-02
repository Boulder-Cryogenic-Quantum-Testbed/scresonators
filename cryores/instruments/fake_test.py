from labrad.units import dBm, GHz, Value

import cryores.instruments.fake as fake


class TestFakeVna:
    """Class-based unit test.

    All methods starting with "test_" are an individual, unique unit test.
    """

    def setup_method(self):
        """Run a setup procedure before each test below."""
        self.device = fake.FakeVna(peak_frequency_GHz=5.0)

    def test_sweep(self):
        # We can use self.device because pytest will always run setup_method
        # before running this test.
        self.device.init_device()

        # Set a parameter.
        self.device.set_device_parameter(1234.5)

        results = self.device.sweep(
            frequency_start=1.0 * GHz,
            frequency_end=10.0 * GHz,
            npoints=10,
            power=1.0 * dBm,
            averages=1,)

        params = self.device.get_parameters()
        assert params == {'custom_parameter': 1234.5}
