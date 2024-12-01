# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
A benchmark for derivatives of functions. 
"""

import numpy as np


class DerivativeBenchmarkProblem:
    """
    Create a benchmark problem for numerical derivatives of a function

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
    
    References
    ----------
    - Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.
    - Adaptive numerical differentiation. R. S. Stepleman and N. D. Winarsky. Journal: Math. Comp. 33 (1979), 1257-1264 
    """
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
        self.name = name
        self.function = function
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative
        self.third_derivative = third_derivative
        self.fourth_derivative = fourth_derivative
        self.x = x

    def get_name(self):
        """
        Return the name of the problem

        Returns
        -------
        name : str
            The name
        """
        return self.name

    def get_x(self):
        """
        Return the input point of the problem

        Returns
        -------
        x : float
            The input point
        """
        return self.x

    def get_function(self):
        """
        Return the function of the problem

        Returns
        -------
        function : function
            The function
        """
        return self.function

    def get_first_derivative(self):
        """
        Return the first derivative of the function of the problem

        Returns
        -------
        first_derivative : function
            The first derivative of the function
        """
        return self.first_derivative

    def get_second_derivative(self):
        """
        Return the second derivative of the function of the problem

        Returns
        -------
        second_derivative : function
            The second derivative of the function
        """
        return self.second_derivative

    def get_third_derivative(self):
        """
        Return the third derivative of the function of the problem

        Returns
        -------
        third_derivative : function
            The third derivative of the function
        """
        return self.third_derivative

    def get_fourth_derivative(self):
        """
        Return the fourth derivative of the function of the problem

        Returns
        -------
        fourth_derivative : function
            The fourth derivative of the function
        """
        return self.fourth_derivative

class ExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    The function is:

    .. math::

        f(x) = \exp(x)

    for any x.

    See problem #1 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """
    def __init__(self):

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


class LogarithmicProblem(DerivativeBenchmarkProblem):
    r"""
    Create a logarithmic derivative benchmark problem

    The function is:

    .. math::

        f(x) = \log(x)

    for any x > 0.

    See problem #2 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """
    def __init__(self):

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


class SquareRootProblem(DerivativeBenchmarkProblem):
    r"""
    Create a square root derivative benchmark problem

    The function is:

    .. math::

        f(x) = \sqrt{x}

    for any x >= 0.

    See problem #3 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """
    def __init__(self):

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


class AtanProblem(DerivativeBenchmarkProblem):
    r"""
    Create an arctangent derivative benchmark problem

    The function is:

    .. math::

        f(x) = \arctan(x)

    for any x.

    See problem #4 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """
    def __init__(self):

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


class SinProblem(DerivativeBenchmarkProblem):
    r"""
    Create a sine derivative benchmark problem

    The function is:

    .. math::

        f(x) = \sin(x)

    for any x.

    See problem #5 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """
    def __init__(self):

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


class ScaledExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create a scaled exponential derivative benchmark problem

    The function is:

    .. math::

        f(x) = \exp(-x / \alpha)

    for any x where :math:`\alpha` is a parameter.

    Parameters
    ----------
    alpha : float, > 0
        The parameter
    """
    def __init__(self, alpha=1.0e6):
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


class GMSWExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See eq. 4 page 312 in (Gill, Murray, Saunders & Wright, 1983)

    .. math::

        f(x) = \left(\exp(x) - 1\right)^2 + \left(\frac{1}{\sqrt{1 + x^2}} - 1\right)^2

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
    def __init__(self, alpha=1.0e6):
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


def BuildBenchmark():
    """
    Create a list of benchmark problems.

    Returns
    -------
    benchmark_list : list(DerivativeBenchmarkProblem)
        A collection of benchmark problems.
    """
    benchmark_list = [
        ExponentialProblem(),
        LogarithmicProblem(),
        SquareRootProblem(),
        AtanProblem(),
        SinProblem(),
        ScaledExponentialProblem(),
        GMSWExponentialProblem(),
    ]
    return benchmark_list
