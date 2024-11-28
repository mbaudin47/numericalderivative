# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
A benchmark for derivatives of functions. 
"""

import numpy as np


class DerivativeBenchmark:
    def __init__(
        self,
        name,
        function,
        first_derivative,
        second_derivative,
        third_derivative,
        fourth_derivative,
        x,
    ):
        """
        Create a benchmark for numerical derivatives of a function

        Parameters
        ----------
        function : function
            The function
        first_derivative : function
            The first derivative of the function
        second_derivative : function
            The second derivative of the function
        third_derivative : function
            The third derivative of the function
        fourth_derivative : function
            The fourth derivative of the function
        x : float
            The point where the derivative should be computed
        """
        self.name = name
        self.function = function
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative
        self.third_derivative = third_derivative
        self.fourth_derivative = fourth_derivative
        self.x = x


class ExponentialDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self):
        """
        Create an exponential derivative benchmark

        The function is:

        f(x) = exp(x)

        for any x.
        """

        def my_exp(x):
            return np.exp(x)

        def my_exp_prime(x):
            return np.exp(x)

        def my_exp_2d_derivative(x):
            return np.exp(x)

        def my_exp_3d_derivative(x):
            return np.exp(x)

        def my_exp_4th_derivative(x):
            return np.exp(x)

        x = 1.0
        super().__init__(
            "exp",
            my_exp,
            my_exp_prime,
            my_exp_2d_derivative,
            my_exp_3d_derivative,
            my_exp_4th_derivative,
            x,
        )


class LogarithmicDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self):
        """
        Create a logarithmic derivative benchmark

        The function is:

        f(x) = log(x)

        for any x > 0.
        """

        def my_log(x):
            return np.log(x)

        def my_log_prime(x):
            return 1.0 / x

        def my_log_2nd_derivative(x):
            return -1.0 / x**2

        def my_log_3d_derivative(x):
            return 2.0 / x**3

        def my_log_4th_derivative(x):
            return -6.0 / x**4

        x = 1.0

        super().__init__(
            "log",
            my_log,
            my_log_prime,
            my_log_2nd_derivative,
            my_log_3d_derivative,
            my_log_4th_derivative,
            x,
        )


class SquareRootDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self):
        """
        Create a square root derivative benchmark

        The function is:

        f(x) = log(x)

        for any x >= 0.
        """

        def my_squareroot(x):
            return np.sqrt(x)

        def my_squareroot_prime(x):
            return 1.0 / (2.0 * np.sqrt(x))

        def my_square_root_2nd_derivative(x):
            return -1.0 / (4.0 * x**1.5)

        def my_square_root_3d_derivative(x):
            return 3.0 / (8.0 * x**2.5)

        def my_square_root_4th_derivative(x):
            return -15.0 / (16.0 * x**3.5)

        x = 1.0
        super().__init__(
            "sqrt",
            my_squareroot,
            my_squareroot_prime,
            my_square_root_2nd_derivative,
            my_square_root_3d_derivative,
            my_square_root_4th_derivative,
            x,
        )


class AtanDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self):
        """
        Create an arctangent derivative benchmark

        The function is:

        f(x) = atan(x)

        for any x.
        """

        def my_atan(x):
            return np.arctan(x)

        def my_atan_prime(x):
            return 1.0 / (1.0 + x**2)

        def my_atan_2nd_derivative(x):
            return -2.0 * x / (1.0 + x**2) ** 2

        def my_atan_3d_derivative(x):
            return (6 * x**2 - 2) / (1.0 + x**2) ** 3

        def my_atan_4th_derivative(x):
            return -24.0 * x * (x**2 - 1) / (1.0 + x**2) ** 4

        x = 0.5

        super().__init__(
            "atan",
            my_atan,
            my_atan_prime,
            my_atan_2nd_derivative,
            my_atan_3d_derivative,
            my_atan_4th_derivative,
            x,
        )


class SinDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self):
        """
        Create a sine derivative benchmark

        The function is:

        f(x) = sin(x)

        for any x.
        """

        def my_sin(x):
            return np.sin(x)

        def my_sin_prime(x):
            return np.cos(x)

        def my_sin_2nd_derivative(x):
            return -np.sin(x)

        def my_sin_3d_derivative(x):
            return -np.cos(x)

        def my_sin_4th_derivative(x):
            return np.sin(x)

        x = 1.0
        super().__init__(
            "sin",
            my_sin,
            my_sin_prime,
            my_sin_2nd_derivative,
            my_sin_3d_derivative,
            my_sin_4th_derivative,
            x,
        )


class ScaledExponentialDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self, alpha=1.0e6):
        """
        Create a scaled exponential derivative benchmark

        The function is:

        f(x) = exp(-x / alpha)

        for any x.

        Parameters
        ----------
        alpha : float, > 0
            The parameter
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha = {alpha} should be > 0")
        self.alpha = alpha

        def scaled_exp(x):
            return np.exp(-x / alpha)

        def scaled_exp_prime(x):
            return -np.exp(-x / alpha) / alpha

        def scaled_exp_2nd_derivative(x):
            return np.exp(-x / alpha) / (alpha**2)

        def scaled_exp_3d_derivative(x):
            return -np.exp(-x / alpha) / (alpha**3)

        def scaled_exp_4th_derivative(x):
            return np.exp(-x / alpha) / (alpha**4)

        x = 1.0
        super().__init__(
            "scaled exp",
            scaled_exp,
            scaled_exp_prime,
            scaled_exp_2nd_derivative,
            scaled_exp_3d_derivative,
            scaled_exp_4th_derivative,
            x,
        )


class GillMurraySaundersWrightExponentialDerivativeBenchmark(DerivativeBenchmark):
    def __init__(self, alpha=1.0e6):
        """
        Create an exponential derivative benchmark

        See eq. 4 page 312 in (Gill, Murray, Saunders & Wright, 1983)

        f(x) = (exp(x) - 1)^2 + (1 / sqrt(1 + x^2) - 1)^2

        Parameters
        ----------
        alpha : float, > 0
            The parameter

        References
        ----------
        - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983).
          Computing forward-difference intervals for numerical optimization.
          SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha = {alpha} should be > 0")
        self.alpha = alpha

        def gms_exp(x):
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = expm1**2 + t**2
            return y

        def gms_exp_prime(x):
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = 2 * np.exp(x) * expm1 - 2 * x * t / s**1.5
            return y

        def gms_exp_2nd_derivative(x):
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = (
                6.0 * t * x**2 / s**2.5
                + 2 * x**2 / s**3
                - 2 * t / s**1.5
                + 2 * np.exp(2 * x)
                + 2 * np.exp(x) * expm1
            )
            return y

        def gms_exp_3d_derivative(x):
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = 2 * (
                -15 * x**3 * t / s ** (7 / 2)
                - 9 * x**3 / s**4
                + 9 * x * t / s ** (5 / 2)
                + 3 * x / s**3
                + expm1 * np.exp(x)
                + 3 * np.exp(2 * x)
            )
            return y

        def gms_exp_4th_derivative(x):
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = 2 * (
                105 * x**4 * t / s ** (9 / 2)
                + 87 * x**4 / s**5
                - 90 * x**2 * t / s ** (7 / 2)
                - 54 * x**2 / s**4
                + 9 * t / s ** (5 / 2)
                + expm1 * np.exp(x)
                + 7 * np.exp(2 * x)
                + 3 / s**3
            )
            return y

        x = 1.0

        super().__init__(
            "GMS",
            gms_exp,
            gms_exp_prime,
            gms_exp_2nd_derivative,
            gms_exp_3d_derivative,
            gms_exp_4th_derivative,
            x,
        )


def BuildBenchmarkList():
    """
    Create a list of benchmark problems.

    Returns
    -------
    benchmark_list : list(DerivativeBenchmark)
        A collection of benchmark problems.
    """
    benchmark_list = [
        ExponentialDerivativeBenchmark(),
        LogarithmicDerivativeBenchmark(),
        SquareRootDerivativeBenchmark(),
        AtanDerivativeBenchmark(),
        ScaledExponentialDerivativeBenchmark(),
        GillMurraySaundersWrightExponentialDerivativeBenchmark(),
    ]
    return benchmark_list
