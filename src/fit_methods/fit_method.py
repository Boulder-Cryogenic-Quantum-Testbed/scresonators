from abc import ABC, abstractmethod
import numpy as np

class FitMethod(ABC):
    @abstractmethod
    def func(self, x: np.ndarray, *params) -> np.ndarray:
        """
        The fit function for the method.

        Args:
            x (np.ndarray): The independent variable data.
            *params: The parameters for the fit function.

        Returns:
            np.ndarray: The evaluated fit function.
        """
        pass

    @abstractmethod
    def find_initial_guess(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> tuple:
        """
        Finds an initial guess for the parameters based on the data.

        Args:
            x (np.ndarray): The independent variable data.
            y1 (np.ndarray): The real part of the dependent variable data.
            y2 (np.ndarray): The imaginary part of the dependent variable data.

        Returns:
            tuple: The initial guess for the parameters, and optionally other relevant information.
        """
        pass

    @abstractmethod
    def min_fit(self, params, xdata: np.ndarray, ydata: np.ndarray):
        """
        Minimizes the parameters for the fit.

        Args:
            params: Initial guess for the parameters.
            xdata (np.ndarray): The independent variable data.
            ydata (np.ndarray): The dependent variable data.

        Returns:
            tuple: The optimized parameters and their confidence intervals.
        """
        pass
