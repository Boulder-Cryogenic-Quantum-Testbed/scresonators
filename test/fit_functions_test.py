# import pytest
# import sys
# import os 
# import glob

# try:
#     path_to_resfit = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
# except:
#     print('Could not find relative path to scresonators. Please update manually.')
#     path_to_resfit = './scresonators'

# pglob = glob.glob(path_to_resfit)
# assert len(pglob), f'Path: {path_to_resfit} does not exist'
# sys.path.append(path_to_resfit)

# import fit_resonator.cavity_functions as ff

# def test_model_params():
#     guess = ff.ModelParams.from_params(Qi=1000,Qc=1000,f_res=5.5, phi=0.1)
#     assert guess.Q == 500.0
#     assert guess.kappa == 0.0055
#     assert guess.Qi == 1000.0

# def test_model_raises_if_over_constrained():
#     with pytest.raises(Exception):
#         ff.ModelParams.from_params(Qi=1000,
#                                    Q=1000,
#                                    Qc=1000,
#                                    f_res=5.5,
#                                    phi=0.1)