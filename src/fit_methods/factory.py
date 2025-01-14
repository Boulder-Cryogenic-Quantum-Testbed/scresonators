# fit_methods/factory.py
from .fit_method import FitMethod
from .dcm import DCM

def create_fit_method(method_name: str, **kwargs) -> FitMethod:
    if method_name == 'DCM':
        return DCM(**kwargs)  # Pass all arguments to the DCM constructor
    else:
        raise ValueError(f"Unknown fit method: {method_name}")