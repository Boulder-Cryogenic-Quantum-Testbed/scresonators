import numpy as np
import scipy.optimize as spopt
import plot as fp
import cavity_functions as ff

def extract_near_res(x_raw: np.ndarray,
                     y_raw: np.ndarray,
                     f_res: float,
                     kappa: float,
                     extract_factor: int = 1):
    """Extracts portions of spectrum of kappa within resonance.

    Args:
        x_raw: X-values of spectrum to extract from.
        y_raw: Y-values of spectrum to extract from.
        f_res: Resonant frequency about which to extract data.
        kappa: Width about f_res to extract.
        extract_factor: Multiplier for kappa.

    Returns:
        Extracted spectrum kappa about f_res.
    """
    # starting resonance to add to fit
    xstart = f_res - extract_factor / 2 * kappa
    # final resonance to add to fit
    xend = f_res + extract_factor / 2 * kappa
    x_temp = []
    y_temp = []
    # xdata is new set of data to be fit, within extract_factor times the 
    # bandwidth, ydata is S21 data to match indices with xdata
    for i, freq in enumerate(x_raw):
        if (freq > xstart and freq < xend):
            x_temp.append(freq)
            y_temp.append(y_raw[i])

    if len(x_temp) < 1:
        raise Exception(">Failed to extract data from designated bandwidth")

    return np.asarray(x_temp), np.asarray(y_temp)

def convert_params(from_method, params):
    if from_method == 'DCM':
        Qc = params[2] / np.cos(params[4])
        Qi = params[1] * Qc / (Qc - params[1])
        Qc_INV = params[2]
        Qi_INV = Qi / (1 + np.sin(params[4]) / Qc_INV / 2)
        return [1 / params[0], Qi_INV, Qc_INV, params[3], -params[4],
                 -params[5]]
    elif from_method == 'INV':
        Qc_DCM = params[2]
        Q_DCM = (np.cos(params[4]) / params[2] + 1 / params[1]) ** -1
        return [1 / params[0], Q_DCM, Qc_DCM, params[3], -params[4], -params[5]]
    
def find_circle(x, y):
    """Given a set of x,y data return a circle that fits data using LeastSquares
      Circle Fit Randy Bullock (2017)

    Args:
        x: Array of x position of data in complex plane (real)
        y: Array of y position of data in complex plane (imaginary)

    Returns:
        x (matrix1) and y (matrix2) center coordinates of the circle, and the 
        radius of the circle "R"
    """
    N = 0
    xavg = 0
    yavg = 0
    for i in range(0, len(x)):
        N = N + 1
        xavg = xavg + x[i]
    for i in range(0, len(y)):
        yavg = yavg + y[i]

    xavg = xavg / N
    yavg = yavg / N

    xnew = []
    ynew = []
    Suu = 0
    Svv = 0
    for i in range(0, len(x)):
        xnew.append(x[i] - xavg)
        Suu = Suu + (x[i] - xavg) * (x[i] - xavg)
    for i in range(0, len(y)):
        ynew.append(y[i] - yavg)
        Svv = Svv + (y[i] - yavg) * (y[i] - yavg)

    Suv = 0
    Suuu = 0
    Svvv = 0
    Suvv = 0
    Svuu = 0
    for i in range(0, len(xnew)):
        Suv = Suv + xnew[i] * ynew[i]
        Suuu = Suuu + xnew[i] * xnew[i] * xnew[i]
        Svvv = Svvv + ynew[i] * ynew[i] * ynew[i]
        Suvv = Suvv + xnew[i] * ynew[i] * ynew[i]
        Svuu = Svuu + ynew[i] * xnew[i] * xnew[i]
    Suv2 = Suv

    matrix1 = 0.5 * (Suuu + Suvv)
    matrix2 = 0.5 * (Svvv + Svuu)

    # row reduction for row 1
    Suv = Suv / Suu
    matrix1 = matrix1 / Suu

    # row subtraction for row 2 by row 1
    Svv = Svv - (Suv * Suv2)
    matrix2 = matrix2 - (Suv2 * matrix1)

    # row reduction for row 2
    matrix2 = matrix2 / Svv

    # row subtraction for row 1 by row 2
    matrix1 = matrix1 - (Suv * matrix2)

    # at this point matrix1 is x_c and matrix2 is y_c
    alpha = (matrix1 * matrix1) + (matrix2 * matrix2) + (Suu + Svv) / N
    R = alpha ** (0.5)

    matrix1 = matrix1 + xavg
    matrix2 = matrix2 + yavg

    return matrix1, matrix2, R

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    return val, idx
