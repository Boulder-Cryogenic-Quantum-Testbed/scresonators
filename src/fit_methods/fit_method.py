from abc import ABC, abstractmethod
import numpy as np
import lmfit

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
    def find_initial_guess(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        """
        Finds an initial guess for the parameters based on the data.

        Args:
            x (np.ndarray): The independent variable data.
            y (np.ndarray): The dependent variable data.

        Returns:
            Parameters: The initial guess for the parameters
        """
        pass

    @abstractmethod
    def create_model(self) -> lmfit.Model:
        """
        Creates an lmfit model of the method function

        Returns:
            Model: object with minimizing methods
        """
        pass

    @abstractmethod
    def generate_highres_fit(self, x: np.ndarray, fit_params: dict, num_fit_points: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a higher-resolution complex fit using the fit method.

        Args:
            x (np.ndarray): The independent variable data.
            fit_params (dict): Dictionary of fit parameters.
            num_fit_points (int): Number of points for higher resolution.

        Returns:
            tuple: Tuple of (high_res_x, high_res_y).
                high_res_x (np.ndarray): High-resolution independent variable data.
                high_res_y (np.ndarray): High-resolution dependent variable data.
        """
        pass