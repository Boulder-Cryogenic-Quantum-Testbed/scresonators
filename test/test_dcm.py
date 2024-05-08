import numpy as np
import pytest
from src.dcm import DCM
from src.fitter import Fitter

# Mocking Fitter.find_circle to return predefined values for the test
Fitter.find_circle = staticmethod(lambda x, y: (0.5, 0.5, 1.0))

@pytest.fixture
def dcm_instance():
    return DCM()

def test_find_initial_guess(dcm_instance):
    x = np.linspace(1, 10, 10)
    y1 = np.sin(x)
    y2 = np.cos(x)
    method = 'DCM'
    
    # Expected values are what you determine they should be based on your known input
    expected_init_guess = [10000, 1000, 5.5, -0.5]  # Example expected values
    expected_circle = (0.5, 0.5, 1.0)  # Based on our mocked find_circle

    init_guess, x_c, y_c, r = dcm_instance.find_initial_guess(x, y1, y2, method)
    
    assert isinstance(init_guess, list), "Initial guess should be a list"
    assert init_guess == expected_init_guess, "Unexpected result for initial guess"
    assert len(init_guess) == 4, "Initial guess should have four elements"
    assert (x_c, y_c, r) == expected_circle, "Circle calculation did not match expected values"
    # Add more specific assertions here based on what your function is expected to compute
