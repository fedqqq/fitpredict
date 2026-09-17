"""fitpredict public package."""

from fitpredict.prediction import predict
from fitpredict.training import FitHistory, FitResult, fit

__all__ = ["FitHistory", "FitResult", "fit", "predict"]
