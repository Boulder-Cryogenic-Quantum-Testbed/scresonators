import numpy as np
import lmfit
from .fit_method import FitMethod
from ..utils import find_circle
from scipy.ndimage import gaussian_filter

class ReflectionMode(FitMethod):
    def __init__(self):
        pass

    @staticmethod
    def func(f, Q, Qc, f0):
        """Reflection mode fit function."""
        return 1-(2*Q/Qc)/(1-2j*Q*(f-f0)/f0)

    @staticmethod
    def fit_function(f, params):
        Q = params['Q'].value
        Qc = params['Qc'].value
        f0 = params['f0'].value

        return 1-(2*Q/Qc)/(1-2j*Q*(f-f0)/f0)

    
    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['f'])
        return model

    def find_initial_guess(self, fdata: np.ndarray, sdata: np.ndarray) -> lmfit.Parameters:
        x_c, y_c, r = find_circle(np.real(sdata), np.imag(sdata))  # the circle diameter is 2*r = 2*Q/Qc
        '''
        in order to robustly estimate the linewidth we smooth the data to eliminate noise
        then we calculate |dS/df| to find the region in frequency space where the resonator response is changing rapidly
        taking the derivative eliminates the need to consider phi for the purpose of estimating the linewidth 

        once we have |dS/df| we count up the segments in the frequency data where |dS/df| > cutoff
        this is implemented with the dot product 
        '''
        filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
        gradSmagnitude = np.abs(np.gradient(filtered_data, fdata))

        chiFunction = np.zeros(len(gradSmagnitude))
        cutoff = 0.5 * (np.min(gradSmagnitude) + np.max(gradSmagnitude))
        for n in range(len(gradSmagnitude)):
            if gradSmagnitude[n] > cutoff:
                chiFunction[n] = 1  # set to one if |dS/df| is above the cutoff at this point
        linewidth = np.dot(chiFunction[:-1], np.diff(fdata))

        f_c = fdata[np.argmax(gradSmagnitude) + 3]  # uncertainties can't be calculated when this guess is too good!!!
        Q_guess = 2 * f_c / (linewidth)
        print(f'Q_guess: {Q_guess}')
        Qc_guess = Q_guess / r

        # Create an lmfit.Parameters object to store initial guesses
        params = lmfit.Parameters()
        params.add('Q', value=Q_guess)
        params.add('Qc', value=Qc_guess)
        params.add('f0', value=f_c, min=f_c * 0.9, max=f_c * 1.1)

        return params

    def extractQi(self, params):
        Q = params['Q'].value
        Qc = params['Qc'].value
        params.add('inverseQi', value =1/Q - 1/Qc)
        inverseQi = params['inverseQi'].value
        params.add('Qi', value = 1/inverseQi)
        # if you get an error that points here check that all your parameters varied during the fit, set verbose = True
        params['inverseQi'].stderr = np.sqrt((params['Q'].stderr/Q**2)**2+(params['Qc'].stderr/Qc**2)**2)
        params['Qi'].stderr = params['inverseQi'].stderr/inverseQi**2
        return params