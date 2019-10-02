import resfit.fit_resonator.fit_functions as ff
import pytest

def test_model_params():
    guess = ff.ModelParams.from_params(Qi=1000,Qc=1000,f_res=5.5, phi=0.1)
    assert guess.Q == 500.0
    assert guess.kappa == 0.0055
    assert guess.Qi == 1000.0

def test_model_raises_if_over_constrained():
    with pytest.raises(Exception):
        ff.ModelParams.from_params(Qi=1000,
                                   Q=1000,
                                   Qc=1000,
                                   f_res=5.5,
                                   phi=0.1)