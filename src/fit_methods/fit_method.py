from abc import ABC, abstractmethod

# FitMethod Interface
class FitMethod(ABC):
    @abstractmethod
    def fit(self, data):
        pass