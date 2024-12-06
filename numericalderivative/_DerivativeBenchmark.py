# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
A benchmark for derivatives of functions. 
"""

import numpy as np


class DerivativeBenchmarkProblem:
    """Create a benchmark problem for numerical derivatives of a function"""

    def __init__(
        self,
        name,
        function,
        first_derivative,
        second_derivative,
        third_derivative,
        fourth_derivative,
        x,
        interval,
    ):
        """
        Create a benchmark problem for numerical derivatives of a function

        This provides the function and the exact first derivative.
        This makes it possible to check the approximation of the first
        derivative using a finite difference formula.
        This class also provides the second, third and fourth derivative.
        This makes it possible to compute the optimal step for
        various finite difference formula.

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
            The point where the derivative should be computed for a single test.
        interval : list of 2 floats
            The lower and upper bounds of the benchmark problem.
            This can be useful for benchmarking on several points.
            We must have interval[0] <= interval[1].

        References
        ----------
        - Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.
        - Adaptive numerical differentiation. R. S. Stepleman and N. D. Winarsky. Journal: Math. Comp. 33 (1979), 1257-1264
        """
        self.name = name
        self.function = function
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative
        self.third_derivative = third_derivative
        self.fourth_derivative = fourth_derivative
        self.x = x
        if interval[0] > interval[1]:
            raise ValueError(
                f"The lower bound {interval[0]} of the interval should be "
                f"lower or equal to the upper bound {interval[1]}."
            )
        self.interval = interval

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

    def get_interval(self):
        """
        Return the interval of the problem

        Returns
        -------
        interval : list of 2 floats
            The interval
        """
        return self.interval

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


class PolynomialProblem(DerivativeBenchmarkProblem):
    r"""
    Create a polynomial derivative benchmark problem

    The function is:

    .. math::

        f(x) = x^\alpha

    for any :math:`x > 0` where :math:`\alpha \in \mathbb{R}` is a nonzero parameter.
    The test point is :math:`x = 1`.

    This test can be difficult depending on the value of :math:`\alpha`.
    For example, if :math:`\alpha = 2`, then the third derivative is zero.
    This produces an infinite exact step for the first derivative
    central finite difference formula.

    """

    def __init__(self, alpha=2):

        def function(x):
            return x**self.alpha

        def function_prime(x):
            return self.alpha * x ** (self.alpha - 1)

        def function_2nd_derivative(x):
            return self.alpha * (self.alpha - 1) * x ** (self.alpha - 2)

        def function_3d_derivative(x):
            return (
                self.alpha * (self.alpha - 1) * (self.alpha - 2) * x ** (self.alpha - 3)
            )

        def function_4th_derivative(x):
            return (
                self.alpha
                * (self.alpha - 1)
                * (self.alpha - 2)
                * (self.alpha - 4)
                * x ** (self.alpha - 4)
            )

        x = 1.0
        interval = [-12.0, 12.0]
        if alpha == 0.0:
            raise ValueError(f"The parameter alpha = {alpha} must be nonzero.")
        self.alpha = alpha
        super().__init__(
            "polynomial",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            x,
            interval,
        )


class ExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    The function is:

    .. math::

        f(x) = \exp(x)

    for any :math:`x`.
    The test point is :math:`x = 1`.


    See problem #1 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """

    def __init__(self):

        def exp(x):
            return np.exp(x)

        def exp_prime(x):
            return np.exp(x)

        def exp_2d_derivative(x):
            return np.exp(x)

        def exp_3d_derivative(x):
            return np.exp(x)

        def exp_4th_derivative(x):
            return np.exp(x)

        x = 1.0
        interval = [0.0, 12.0]
        super().__init__(
            "exp",
            exp,
            exp_prime,
            exp_2d_derivative,
            exp_3d_derivative,
            exp_4th_derivative,
            x,
            interval,
        )


class LogarithmicProblem(DerivativeBenchmarkProblem):
    r"""
    Create a logarithmic derivative benchmark problem

    The function is:

    .. math::

        f(x) = \log(x)

    for any :math:`x > 0`.
    The test point is :math:`x = 1`.

    See problem #2 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """

    def __init__(self):

        def log(x):
            return np.log(x)

        def log_prime(x):
            return 1.0 / x

        def log_2nd_derivative(x):
            return -1.0 / x**2

        def log_3d_derivative(x):
            return 2.0 / x**3

        def log_4th_derivative(x):
            return -6.0 / x**4

        x = 1.0
        interval = [0.01, 12.0]
        super().__init__(
            "log",
            log,
            log_prime,
            log_2nd_derivative,
            log_3d_derivative,
            log_4th_derivative,
            x,
            interval,
        )


class SquareRootProblem(DerivativeBenchmarkProblem):
    r"""
    Create a square root derivative benchmark problem

    The function is:

    .. math::

        f(x) = \sqrt{x}

    for any :math:`x \geq 0`.
    The test point is :math:`x = 1`.

    The square root function is difficult to differentiate at x = 0:
    its first derivative is infinite.

    See problem #3 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """

    def __init__(self):

        def squareroot(x):
            return np.sqrt(x)

        def squareroot_prime(x):
            return 1.0 / (2.0 * np.sqrt(x))

        def square_root_2nd_derivative(x):
            return -1.0 / (4.0 * x**1.5)

        def square_root_3d_derivative(x):
            return 3.0 / (8.0 * x**2.5)

        def square_root_4th_derivative(x):
            return -15.0 / (16.0 * x**3.5)

        x = 1.0
        interval = [0.01, 12.0]
        super().__init__(
            "sqrt",
            squareroot,
            squareroot_prime,
            square_root_2nd_derivative,
            square_root_3d_derivative,
            square_root_4th_derivative,
            x,
            interval,
        )


class AtanProblem(DerivativeBenchmarkProblem):
    r"""
    Create an arctangent derivative benchmark problem

    The function is:

    .. math::

        f(x) = \arctan(x)

    for any :math:`x`.
    The test point is :math:`x = 1/2`.

    See problem #4 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """

    def __init__(self):

        def atan(x):
            return np.arctan(x)

        def atan_prime(x):
            return 1.0 / (1.0 + x**2)

        def atan_2nd_derivative(x):
            return -2.0 * x / (1.0 + x**2) ** 2

        def atan_3d_derivative(x):
            return (6 * x**2 - 2) / (1.0 + x**2) ** 3

        def atan_4th_derivative(x):
            return -24.0 * x * (x**2 - 1) / (1.0 + x**2) ** 4

        x = 0.5
        interval = [-12.0, 12.0]
        super().__init__(
            "atan",
            atan,
            atan_prime,
            atan_2nd_derivative,
            atan_3d_derivative,
            atan_4th_derivative,
            x,
            interval,
        )


class SinProblem(DerivativeBenchmarkProblem):
    r"""
    Create a sine derivative benchmark problem

    The function is:

    .. math::

        f(x) = \sin(x)

    for any :math:`x`.
    The test point is :math:`x = 1`.

    See problem #5 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.
    """

    def __init__(self):

        def sin(x):
            return np.sin(x)

        def sin_prime(x):
            return np.cos(x)

        def sin_2nd_derivative(x):
            return -np.sin(x)

        def sin_3d_derivative(x):
            return -np.cos(x)

        def sin_4th_derivative(x):
            return np.sin(x)

        x = 1.0
        interval = [-np.pi, np.pi]
        super().__init__(
            "sin",
            sin,
            sin_prime,
            sin_2nd_derivative,
            sin_3d_derivative,
            sin_4th_derivative,
            x,
            interval,
        )


class ScaledExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create a scaled exponential derivative benchmark problem

    The function is:

    .. math::

        f(x) = \exp(-x / \alpha)

    for any :math:`x` where :math:`\alpha` is a parameter.
    The test point is :math:`x = 1`.

    This problem is interesting because the optimal step for the central
    finite difference formula of the first derivative is 6.694, which
    is much larger than we may expect.

    Parameters
    ----------
    alpha : float, nonzero 0
        The parameter
    """

    def __init__(self, alpha=1.0e6):
        if alpha == 0.0:
            raise ValueError(f"alpha = {alpha} should be nonzero")
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
        interval = [0.0, 12.0]
        super().__init__(
            "scaled exp",
            scaled_exp,
            scaled_exp_prime,
            scaled_exp_2nd_derivative,
            scaled_exp_3d_derivative,
            scaled_exp_4th_derivative,
            x,
            interval,
        )


class GMSWExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See eq. 4 page 312 in (Gill, Murray, Saunders & Wright, 1983)

    The function is:

    .. math::

        f(x) = \left(\exp(x) - 1\right)^2 + \left(\frac{1}{\sqrt{1 + x^2}} - 1\right)^2

    for any :math:`x`.
    The test point is :math:`x = 1`.
    The optimal finite difference step for the forward finite difference
    formula of the first derivative is approximately :math:`10^{-3}`.

    Parameters
    ----------
    alpha : float, > 0
        The parameter

    References
    ----------
    - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
    """

    def __init__(self):

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
        interval = [0.0, 12.0]
        super().__init__(
            "GMSW",
            gms_exp,
            gms_exp_prime,
            gms_exp_2nd_derivative,
            gms_exp_3d_derivative,
            gms_exp_4th_derivative,
            x,
            interval,
        )


class SXXNProblem1(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See page 14 in (Shi, Xie, Xuan & Nocedal, 2022)

    The function is:

    .. math::

        f(x) = \left(\exp(x) - 1\right)^2

    for any :math:`x`.
    The test point is :math:`x = -8`.

    According to (Shi, Xie, Xuan & Nocedal, 2022), this function
    has "extremely small first and second order derivatives at t = -8".
    A naive choice of the step for forward differences can result in
    extremely large step and huge error.

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self):
        def sxxn_exp1(x):
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = expm1**2
            return y

        def sxxn_exp1_prime(x):
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = 2 * np.exp(x) * expm1
            return y

        def sxxn_exp1_2nd_derivative(x):
            y = 2 * np.exp(x) * (2 * np.exp(x) - 1)
            return y

        def sxxn_exp1_3d_derivative(x):
            y = 2 * np.exp(x) * (4 * np.exp(x) - 1)
            return y

        def sxxn_exp1_4th_derivative(x):
            y = 2 * np.exp(x) * (8 * np.exp(x) - 1)
            return y

        x = -8.0
        interval = [-12.0, 12.0]
        super().__init__(
            "SXXN1",
            sxxn_exp1,
            sxxn_exp1_prime,
            sxxn_exp1_2nd_derivative,
            sxxn_exp1_3d_derivative,
            sxxn_exp1_4th_derivative,
            x,
            interval,
        )


class SXXNProblem2(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See page 14 in (Shi, Xie, Xuan & Nocedal, 2022)

    The function is:

    .. math::

        f(x) = \exp(\alpha x)

    for any :math:`x` and :math:`\alpha` is a parameter.
    The test point is :math:`x = 0.01`.

    The function is similar to ScaledExponentialProblem,
    but the scaling and the test point are different.

    According to (Shi, Xie, Xuan & Nocedal, 2022), this problem is
    interesting because the function has high order derivatives
    which increase rapidly.
    Therefore, a finite difference formula can be inaccurate
    if the step size is chosen to be large.

    Parameters
    ----------
    alpha : float, > 0
        The parameter.

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self, alpha=1.0e2):
        self.alpha = alpha

        def sxxn_exp2(x):
            y = np.exp(self.alpha * x)
            return y

        def sxxn_exp2_prime(x):
            y = self.alpha * np.exp(self.alpha * x)
            return y

        def sxxn_exp2_2nd_derivative(x):
            y = self.alpha**2 * np.exp(self.alpha * x)
            return y

        def sxxn_exp2_3d_derivative(x):
            y = self.alpha**3 * np.exp(self.alpha * x)
            return y

        def sxxn_exp2_4th_derivative(x):
            y = self.alpha**4 * np.exp(self.alpha * x)
            return y

        x = 0.01
        interval = [-1.0, 1.0]

        super().__init__(
            "SXXN2",
            sxxn_exp2,
            sxxn_exp2_prime,
            sxxn_exp2_2nd_derivative,
            sxxn_exp2_3d_derivative,
            sxxn_exp2_4th_derivative,
            x,
            interval,
        )


class SXXNProblem3(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See page 14 in (Shi, Xie, Xuan & Nocedal, 2022)

    The function is:

    .. math::

        f(x) = x^4 + 3x^2 - 10x

    for any :math:`x`.
    The test point is :math:`x = 0.99999`.

    According to (Shi, Xie, Xuan & Nocedal, 2022), this problem
    is difficult because f'(1) = 0.

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self):
        def sxxn_3(x):
            y = x**4 + 3 * x**2 - 10 * x
            return y

        def sxxn_3_prime(x):
            y = 4 * x**3 + 6 * x - 10
            return y

        def sxxn_3_2nd_derivative(x):
            y = 12 * x**2 + 6
            return y

        def sxxn_3_3d_derivative(x):
            y = 24 * x
            return y

        def sxxn_3_4th_derivative(x):
            y = 24
            return y

        x = 0.99999
        interval = [-12.0, 12.0]

        super().__init__(
            "SXXN3",
            sxxn_3,
            sxxn_3_prime,
            sxxn_3_2nd_derivative,
            sxxn_3_3d_derivative,
            sxxn_3_4th_derivative,
            x,
            interval,
        )


class SXXNProblem4(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See page 14 in (Shi, Xie, Xuan & Nocedal, 2022)

    The function is:

    .. math::

        f(x) = 10000 \; x^3 + 0.01 \; x^2 + 5x

    for any :math:`x`.
    The test point is :math:`x = 10^{-9}`.

    According to (Shi, Xie, Xuan & Nocedal, 2022), this problem
    is difficult because the function is approximately symmetric with
    respect to :math:`x = 0`.

    The fourth derivative is zero, which produces an infinite optimal
    second derivative step for central finite difference formula.

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self):
        def sxxn_4(x):
            y = 1.0e4 * x**3 + 0.01 * x**2 + 5 * x
            return y

        def sxxn_4_prime(x):
            y = 3.0e4 * x**2 + 0.02 * x + 5
            return y

        def sxxn_4_2nd_derivative(x):
            y = 6.0e4 * x + 0.02
            return y

        def sxxn_4_3d_derivative(x):
            y = 6.0e4
            return y

        def sxxn_4_4th_derivative(x):
            y = 0
            return y

        x = 1.0e-9
        interval = [-12.0, 12.0]

        super().__init__(
            "SXXN4",
            sxxn_4,
            sxxn_4_prime,
            sxxn_4_2nd_derivative,
            sxxn_4_3d_derivative,
            sxxn_4_4th_derivative,
            x,
            interval,
        )


class OliverProblem1(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See table 1 page 151 in (Oliver, 1980)

    The function is:

    .. math::

        f(x) = \exp(4 * x)

    for any :math:`x`.
    The test point is :math:`x = 1`.
    This is the ScaledExponentialProblem with :math:`\alpha = -1/4`.

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self):
        alpha = -1.0 / 4.0
        problem = ScaledExponentialProblem(alpha)
        interval = [-12.0, 12.0]

        super().__init__(
            "Oliver1",
            problem.get_function(),
            problem.get_first_derivative(),
            problem.get_second_derivative(),
            problem.get_third_derivative(),
            problem.get_fourth_derivative(),
            problem.get_x(),
            interval,
        )


class OliverProblem2(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See table 1 page 151 in (Oliver, 1980)

    The function is:

    .. math::

        f(x) = \exp(x^2)

    for any :math:`x`.
    The test point is :math:`x = 1`.

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self):
        def function(x):
            y = np.exp(x**2)
            return y

        def function_prime(x):
            y = 2.0 * np.exp(x**2) * x
            return y

        def function_2nd_derivative(x):
            y = 2.0 * np.exp(x**2) * (2 * x**2 + 1)
            return y

        def function_3d_derivative(x):
            y = 4.0 * np.exp(x**2) * x * (2 * x**2 + 3)
            return y

        def function_4th_derivative(x):
            y = 4.0 * np.exp(x**2) * (4 * x**4 + 12 * x**2 + 3)
            return y

        x = 1.0
        interval = [-12.0, 12.0]

        super().__init__(
            "Oliver2",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            x,
            interval,
        )


class OliverProblem3(DerivativeBenchmarkProblem):
    r"""
    Create an logarithmic derivative benchmark problem

    See table 1 page 151 in (Oliver, 1980)

    The function is:

    .. math::

        f(x) = x^2 \ln(x)

    for any :math:`x`.
    The test point is :math:`x = 1`.

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self):
        def function(x):
            y = x**2 * np.log(x)
            return y

        def function_prime(x):
            y = x + 2.0 * x * np.log(x)
            return y

        def function_2nd_derivative(x):
            y = 2.0 * np.log(x) + 3.0
            return y

        def function_3d_derivative(x):
            y = 2.0 / x
            return y

        def function_4th_derivative(x):
            y = -2.0 / x**2
            return y

        x = 1.0
        interval = [0.01, 12.0]

        super().__init__(
            "Oliver3",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            x,
            interval,
        )


class InverseProblem(DerivativeBenchmarkProblem):
    r"""
    Create an inverse derivative benchmark problem

    See table 1 page 151 in (Oliver, 1980)

    The function is:

    .. math::

        f(x) = \frac{1}{x}

    for any nonzero :math:`x`.
    The test point is :math:`x = 1`.

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self):
        def function(x):
            y = 1.0 / x
            return y

        def function_prime(x):
            y = -1.0 / x**2
            return y

        def function_2nd_derivative(x):
            y = 2.0 / x**3
            return y

        def function_3d_derivative(x):
            y = -6.0 / x**4
            return y

        def function_4th_derivative(x):
            y = 24.0 / x**5
            return y

        x = 1.0
        interval = [0.01, 12.0]

        super().__init__(
            "inverse",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            x,
            interval,
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
        PolynomialProblem(),
        InverseProblem(),
        ExponentialProblem(),
        LogarithmicProblem(),
        SquareRootProblem(),
        AtanProblem(),
        SinProblem(),
        ScaledExponentialProblem(),
        GMSWExponentialProblem(),
        SXXNProblem1(),
        SXXNProblem2(),
        SXXNProblem3(),
        SXXNProblem4(),
        OliverProblem1(),
        OliverProblem2(),
        OliverProblem3(),
    ]
    return benchmark_list
