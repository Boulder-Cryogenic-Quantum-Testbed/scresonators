import numpy as np

def unpack_s2p_df(s2p_df, return_bare_minimum=False):
    """
        This function through a provided DataFrame and returns a full set (or bare
            minimum) of arrays that compose an s-parameter measurement.
            
        Currently, the function loops through the columns and compares the col
            string manually to the following list:
            
        ['dB', 'lin', 'rad', 'deg', 'compl', 'cmpl', 'real', 'imag']
        
        
    
    """
    ## start by initializing all possible formats as None, then replace one-by-one
    magn_lin, magn_dB = None, None
    phase_rad, phase_deg = None, None
    real, imag = None, None
    cmpl = None
    
    # check_list = ['dB', 'lin', 'rad', 'deg', 'compl', 'cmpl', 'real', 'imag']
    # complete_lists = [ ["dB", "lin", "deg", "rad"], ["cmpl"], ["real", "imag"]]
    
    # loop over all columns in DataFrame
    #   for each column, identify the format.
    #   as soon as it is identified, exit the 
    #   if statement (due to using elif), then 
    #   check if we have enough info overall,
    #   and if so, break & convert the rest

    for col in s2p_df:
        if "freq" in col.lower():
            freqs = s2p_df[col].values
        
        elif "db" in col.lower():
            magn_dB = s2p_df[col].values
            magn_lin = 10**(magn_dB/20)

        elif "lin" in col.lower():
            magn_lin = s2p_df[col].values
            magn_dB = 20*np.log10(magn_lin)
            
        elif "rad" in col.lower():
            phase_rad = s2p_df[col].values
            phase_deg = np.rad2deg(phase_rad)
            
        elif "deg" in col.lower():
            phase_deg = s2p_df[col].values
            phase_rad = np.deg2rad(phase_deg)
            
        elif "compl" in col.lower() or "cmpl" in col.lower():
            real, imag = np.real(cmpl), np.imag(cmpl)
            magn_lin = np.abs(cmpl)
            phase_rad = np.unwrap(np.angle(cmpl))
            
        elif "real" in col.lower():
            real = s2p_df[col].values
            
        elif "imag" in col.lower():
            imag = s2p_df[col].values
        
        else:
            print(f"column of s2p_df is not in recognized list!")
            print(f"    col is not in ['dB', 'lin', 'rad', 'deg', 'compl', 'cmpl', 'real', or 'imag'].")
            print(f"        {col=}")    
            
        #########################
        ##### as soon as we have a set of variables that fully encompasses
        ##### the dataset, just stop searching & convert the last ones
        #########################
        
    s2p_dict_all_vals = {
        "Frequency"  : freqs,
        
        "magn_lin"   : magn_lin,
        "magn_dB"    : magn_dB,
        "phase_rad"  : phase_rad,
        "phase_deg"  : phase_deg ,
        "real"       : real,
        "imag "      : imag,
        "cmpl"       : cmpl,
    }        
    
    try:
        if "magn_dB" in s2p_dict_all_vals and "phase_rad" in s2p_dict_all_vals:
            cmpl = magn_lin * np.exp(1j * phase_rad)
            real, imag = np.real(cmpl), np.imag(cmpl)
            magn_lin = np.abs(cmpl)
            phase_deg = np.deg2rad(phase_rad)
            print("found magn_dB, phase")
        
        elif "magn_lin" in s2p_dict_all_vals and "phase_deg" in s2p_dict_all_vals:
            cmpl = magn_lin * np.exp(1j * phase_rad)
            real, imag = np.real(cmpl), np.imag(cmpl)
            magn_dB = 20*np.log10(magn_lin)
            phase_rad = np.angle(cmpl)
            print("found magn_lin, phase_deg")
            
        if "cmpl" in s2p_dict_all_vals:
            real, imag = np.real(cmpl), np.imag(cmpl)
            magn_lin = np.abs(cmpl)
            magn_dB = 20*np.log10(magn_lin)
            phase_rad, phase_deg  = np.angle(cmpl), np.rad2deg(np.angle(cmpl))
            print("found cmpl")
            
        elif "real" in s2p_dict_all_vals and "imag" in s2p_dict_all_vals:
            cmpl = real + 1j*imag
            magn_lin = np.abs(cmpl)
            magn_dB = 20*np.log10(magn_lin)
            phase_rad, phase_deg  = np.angle(cmpl), np.rad2deg(np.angle(cmpl))
            print("found real, imag")
        
    except Exception as e:
        raise e
    
    
        
    s2p_dict_all_vals.update({
        "Frequency"  : freqs,
        
        "magn_lin"   : magn_lin,
        "magn_dB"    : magn_dB,
        "phase_rad"  : phase_rad,
        "phase_deg"  : phase_deg ,
        "real"       : real,
        "imag"      : imag,
        "cmpl"       : cmpl,
    })
    
    
    # construct final dictionary using dict comprehension to remove all None elements
    
    if return_bare_minimum is True:
        s2p_dict = { "Frequency" : freqs, "magn_dB" : magn_dB, "phase_rad" : phase_rad}
    else:
        s2p_dict = { key : val for key, val in s2p_dict_all_vals.items() if val is not None}
    
    return s2p_dict




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


# ~~~ from John's utils.py
def remove_delay(fdata: np.ndarray, sdata: np.ndarray, delay):
    return np.exp(2j*np.pi*delay*fdata)*sdata
