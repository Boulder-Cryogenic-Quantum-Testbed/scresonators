import numpy as np

def find_nearest(array: np.ndarray, value: float) -> tuple:
    """
    Finds the nearest value and its index in an array to a given value.

    Args:
        array (np.ndarray): The array to search.
        value (float): The value to find the nearest match for.

    Returns:
        tuple: A tuple containing the nearest value and its index in the array.
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_circle(x: np.ndarray, y: np.ndarray) -> tuple:
    """Finds a circle that best fits the given x, y data using Least Squares Circle Fit.
    
    Args:
        x (np.ndarray): Array of x positions of data in the complex plane (real).
        y (np.ndarray): Array of y positions of data in the complex plane (imaginary).
        
    Returns:
        tuple: Coordinates of the circle's center (x, y) and its radius (R).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    x, y = np.asarray(x), np.asarray(y)
    x_mean, y_mean = np.mean(x), np.mean(y)

    # Check if there are at least three unique points
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        raise ValueError("At least three unique points are required to define a circle.")

    # Check for collinearity by calculating the area of the triangle formed by the first three unique points
    if len(x) >= 3:
        area = 0.5 * abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
        if area == 0:
            raise ValueError("The points are collinear and cannot define a circle.")

    # Center the data
    u, v = x - x_mean, y - y_mean

    # Matrix form of the system
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Suuv)])
    uc, vc = np.linalg.solve(A, B)

    # Radius and center
    center_x = uc + x_mean
    center_y = vc + y_mean
    radius = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(x))

    return center_x, center_y, radius

def phase_centered(f, fr, Ql, theta, delay=0.):
    """
    Yields the phase response of a strongly overcoupled resonator
    in reflection, centered around the origin, with a potential linear
    background phase slope due to signal delay.

    Args:
        f: Frequency array.
        fr: Resonance frequency.
        Ql: Loaded quality factor (Ql ≈ Qc for Qi >> Qc).
        theta: Offset phase.
        delay (optional): Time delay between output and input signal,
                          leading to a linearly frequency-dependent phase shift.

    Returns:
        np.ndarray: Phase response for each frequency in `f`.
    """
    return theta - 2 * np.pi * delay * (f - fr) + 2. * np.arctan(2. * Ql * (1. - f / fr))

def phase_dist(angle):
    """
    Maps an angle from [-2pi, +2pi] to a phase distance on a circle [0, pi].
    
    Args:
        angle (np.ndarray or float): Angle(s) to be mapped, in radians.
        
    Returns:
        np.ndarray or float: The phase distance(s) corresponding to the input angle(s),
                             guaranteed to be in the range [0, pi].
    """
    return np.pi - np.abs(np.pi - np.abs(angle))

def periodic_boundary(angle):
    """
    Maps an arbitrary angle to the interval [-np.pi, np.pi).
    
    Args:
        angle (float or np.ndarray): Angle(s) in radians to be mapped.
    
    Returns:
        float or np.ndarray: The mapped angle(s) within [-np.pi, np.pi).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def normalize(f_data: np.ndarray, z_data: np.ndarray, delay: float, a: float, alpha: float) -> np.ndarray:
    """
    Normalizes scattering data to a canonical position with the off-resonant
    point at (1, 0), adjusting for amplitude and phase offset, but not for
    rotation around the off-resonant point.

    Args:
        f_data (np.ndarray): The frequency data array, not used in the current implementation
                             but included for future extensions or adjustments.
        z_data (np.ndarray): The complex scattering data to be normalized.
        delay (float): The cable delay; currently not used in normalization but
                       included for consistency with calibration parameters.
        a (float): The normalization amplitude factor.
        alpha (float): The phase offset to be corrected.

    Returns:
        np.ndarray: The normalized scattering data.
    """
    z_norm = (z_data / a) * np.exp(-1j * alpha)
    return z_norm

