"""numericalderivative module."""

from .FunctionWithArguments import FunctionWithArguments
from .DerivativeBenchmark import (
    DerivativeBenchmarkProblem,
    InverseProblem,
    ExponentialProblem,
    LogarithmicProblem,
    SquareRootProblem,
    AtanProblem,
    SinProblem,
    ScaledExponentialProblem,
    GMSWExponentialProblem,
    SXXNProblem1,
    SXXNProblem2,
    SXXNProblem3,
    SXXNProblem4,
    OliverProblem1,
    OliverProblem2,
    OliverProblem3,
    BuildBenchmark,
)
from .DumontetVignes import DumontetVignes
from .FiniteDifferenceOptimalStep import FiniteDifferenceOptimalStep
from .GillMurraySaundersWright import GillMurraySaundersWright
from .SteplemanWinarsky import SteplemanWinarsky
from .FiniteDifferenceFormula import FiniteDifferenceFormula

__all__ = [
    "FiniteDifferenceFormula",
    "FunctionWithArguments",
    "DerivativeBenchmarkProblem",
    "InverseProblem",
    "ExponentialProblem",
    "LogarithmicProblem",
    "SquareRootProblem",
    "AtanProblem",
    "SinProblem",
    "ScaledExponentialProblem",
    "GMSWExponentialProblem",
    "SXXNProblem1",
    "SXXNProblem2",
    "SXXNProblem3",
    "SXXNProblem4",
    "OliverProblem1",
    "OliverProblem2",
    "OliverProblem3",
    "DumontetVignes",
    "FiniteDifferenceOptimalStep",
    "GillMurraySaundersWright",
    "SteplemanWinarsky",
    "BuildBenchmark",
]
__version__ = "1.0"
