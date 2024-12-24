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
        fifth_derivative,
        x,
        interval,
    ):
        """
        Create a benchmark problem for numerical derivatives of a function

        This provides the function and the exact first derivative.
        This makes it possible to check the approximation of the first
        derivative using a finite difference formula.
        This class also provides the second, third, fourth and fifth derivative.
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
        fifth_derivative : function
            The fifth derivative of the function
        x : float
            The point where the derivative should be computed for a single test.
        interval : list of 2 floats
            The lower and upper bounds of the benchmark problem.
            This can be useful for benchmarking on several points.
            We must have interval[0] <= interval[1].

        Examples
        --------
        The next example creates a benchmark problem.

        >>> import numericalderivative as nd
        >>> problem = nd.ExponentialProblem()
        >>> x = problem.get_x()
        >>> function = problem.get_function()
        >>> first_derivative = problem.get_first_derivative()

        Print a problem.

        >>> problem = nd.ExponentialProblem()
        >>> print(problem)

        The next example creates a benchmark experiment.

        >>> import numericalderivative as nd
        >>> benchmark = nd.BuildBenchmark()
        >>> number_of_problems = len(benchmark)
        >>> for i in range(number_of_problems):
        >>>     problem = benchmark[i]
        >>>     name = problem.get_name()
        >>>     print(f"Problem #{i}: {name}")

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
        self.fifth_derivative = fifth_derivative
        self.x = x
        if interval[0] > interval[1]:
            raise ValueError(
                f"The lower bound {interval[0]} of the interval should be "
                f"lower or equal to the upper bound {interval[1]}."
            )
        self.interval = interval

    def __str__(self, x=None) -> str:
        report = ""
        report += f"DerivativeBenchmarkProblem\n"
        report += f"name = {self.name}\n"
        report += f"x = {self.x}\n"
        report += f"f(x) = {self.function(self.x)}\n"
        report += f"f'(x) = {self.first_derivative(self.x)}\n"
        try:
            report += f"f''(x) = {self.second_derivative(self.x)}\n"
        except:
            report += f"f''(x) = undefined\n"
        try:
            report += f"f^(3)(x) = {self.third_derivative(self.x)}\n"
        except:
            report += f"f^(3)(x) = undefined\n"
        try:
            report += f"f^(4)(x) = {self.fourth_derivative(self.x)}\n"
        except:
            report += f"f^(4)(x) = undefined\n"
        try:
            report += f"f^(5)(x) = {self.fifth_derivative(self.x)}\n"
        except:
            report += f"f^(5)(x) = undefined\n"
        return report

    def _repr_html_(self, x=None) -> str:
        report = ""
        report += f"<b>DerivativeBenchmarkProblem</b>\n"
        report += f"<ul>\n"
        report += f"<li>name = {self.name}</li>\n"
        report += f"<li>x = {self.x}</li>\n"
        report += f"<li>f(x) = {self.function(self.x)}</li>\n"
        report += f"<li>f'(x) = {self.first_derivative(self.x)}</li>\n"
        try:
            report += f"<li>f''(x) = {self.second_derivative(self.x)}</li>\n"
        except:
            report += f"<li>f''(x) = undefined</li>\n"
        try:
            report += f"<li>f^(3)(x) = {self.third_derivative(self.x)}</li>\n"
        except:
            report += f"<li>f^(3)(x) = undefined</li>\n"
        try:
            report += f"<li>f^(4)(x) = {self.fourth_derivative(self.x)}</li>\n"
        except:
            report += f"<li>f^(4)(x) = undefined</li>\n"
        try:
            report += f"<li>f^(5)(x) = {self.fifth_derivative(self.x)}</li>\n"
        except:
            report += f"<li>f^(5)(x) = undefined</li>\n"
        report += f"</ul>\n"
        return report

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

    def get_fifth_derivative(self):
        """
        Return the fifth derivative of the function of the problem

        Returns
        -------
        fifth_derivative : function
            The fifth derivative of the function
        """
        return self.fifth_derivative


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
    For example, the :class:`~numericalderivative.DumontetVignes` algorithm
    does not perform correctly for this problem because it is
    based on the hypothesis that the third derivative is zero.

    The central finite difference for the first derivative
    is exact for this problem for any value of the differentiation step.

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, alpha=2, x=1.0, interval=[-12.0, 12.0]):

        def function(x):
            return x**self.alpha

        def function_prime(x):
            if self.alpha == 0.0:
                y = 0.0
            else:
                y = self.alpha * x ** (self.alpha - 1)
            return y

        def function_2nd_derivative(x):
            if self.alpha == 0.0 or self.alpha == 1.0:
                y = 0.0
            else:
                y = self.alpha * (self.alpha - 1) * x ** (self.alpha - 2)
            return y

        def function_3d_derivative(x):
            if self.alpha == 0.0 or self.alpha == 1.0 or self.alpha == 2.0:
                y = 0.0
            else:
                y = (
                    self.alpha
                    * (self.alpha - 1)
                    * (self.alpha - 2)
                    * x ** (self.alpha - 3)
                )
            return y

        def function_4th_derivative(x):
            if (
                self.alpha == 0.0
                or self.alpha == 1.0
                or self.alpha == 2.0
                or self.alpha == 3.0
            ):
                y = 0.0
            else:
                y = (
                    self.alpha
                    * (self.alpha - 1)
                    * (self.alpha - 2)
                    * (self.alpha - 3)
                    * x ** (self.alpha - 4)
                )
            return y

        def function_5th_derivative(x):
            if (
                self.alpha == 0.0
                or self.alpha == 1.0
                or self.alpha == 2.0
                or self.alpha == 3.0
                or self.alpha == 4.0
            ):
                y = 0.0
            else:
                y = (
                    self.alpha
                    * (self.alpha - 1)
                    * (self.alpha - 2)
                    * (self.alpha - 3)
                    * (self.alpha - 4)
                    * x ** (self.alpha - 5)
                )
            return y

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
            function_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, x=1.0, interval=[0.0, 12.0]):

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

        def function_5th_derivative(x):
            return np.exp(x)

        super().__init__(
            "exp",
            exp,
            exp_prime,
            exp_2d_derivative,
            exp_3d_derivative,
            exp_4th_derivative,
            function_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, x=1.0, interval=[0.01, 12.0]):

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

        def log_5th_derivative(x):
            return 24 / x**5

        super().__init__(
            "log",
            log,
            log_prime,
            log_2nd_derivative,
            log_3d_derivative,
            log_4th_derivative,
            log_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, x=1.0, interval=[0.01, 12.0]):

        problem = PolynomialProblem(0.5)
        super().__init__(
            "sqrt",
            problem.get_function(),
            problem.get_first_derivative(),
            problem.get_second_derivative(),
            problem.get_third_derivative(),
            problem.get_fourth_derivative(),
            problem.get_fifth_derivative(),
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, x=0.5, interval=[-12.0, 12.0]):

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

        def atan_5th_derivative(x):
            return 24.0 * (5 * x**4 - 10 * x**2 + 1) / (1.0 + x**2) ** 5

        super().__init__(
            "atan",
            atan,
            atan_prime,
            atan_2nd_derivative,
            atan_3d_derivative,
            atan_4th_derivative,
            atan_5th_derivative,
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

    This function can be difficult to differentiate at the points
    :math:`x = \pm \pi` because the second derivative is zero at these
    points.

    See problem #5 in (Dumontet & Vignes, 1977) page 23.
    See (Stepleman & Wirnarsky, 1979) page 1263.

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, x=1.0, interval=[-np.pi, np.pi]):

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

        def sin_5th_derivative(x):
            return np.cos(x)

        super().__init__(
            "sin",
            sin,
            sin_prime,
            sin_2nd_derivative,
            sin_3d_derivative,
            sin_4th_derivative,
            sin_5th_derivative,
            x,
            interval,
        )


class ScaledExponentialProblem(DerivativeBenchmarkProblem):
    r"""
    Create a scaled exponential derivative benchmark problem

    The function is:

    .. math::

        f(x) = \exp(\alpha x)

    for any :math:`x` where :math:`\alpha` is a parameter.
    The test point is :math:`x = 1`.

    This problem is interesting because the optimal step for the central
    finite difference formula of the first derivative is 6.694, which
    is much larger than we may expect.

    Parameters
    ----------
    alpha : float, nonzero 0
        The parameter
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
    """

    def __init__(self, alpha=-1.0e-6, x=1.0, interval=[0.0, 12.0]):
        if alpha == 0.0:
            raise ValueError(f"alpha = {alpha} should be nonzero")
        self.alpha = alpha

        problem = SXXNProblem2(alpha)
        super().__init__(
            "scaled exp",
            problem.get_function(),
            problem.get_first_derivative(),
            problem.get_second_derivative(),
            problem.get_third_derivative(),
            problem.get_fourth_derivative(),
            problem.get_fifth_derivative(),
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
    For this point, the value of the function is zero.
    Hence, the absolute error of the function evaluation cannot be
    computed from a given relative error for this test point.
    The optimal finite difference step for the forward finite difference
    formula of the first derivative is approximately :math:`10^{-3}`.

    Parameters
    ----------
    alpha : float, > 0
        The parameter
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].
        The interval should not contain x = 0 because the first
        derivative is zero at this point.
        This may create an infinite relative error.

    References
    ----------
    - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
    """

    def __init__(self, x=1.0, interval=[0.001, 12.0]):

        sxxn1 = SXXNProblem1()
        sxxn1_function = sxxn1.get_function()
        sxxn1_1st_derivative = sxxn1.get_first_derivative()
        sxxn1_2nd_derivative = sxxn1.get_second_derivative()
        sxxn1_3d_derivative = sxxn1.get_third_derivative()
        sxxn1_4th_derivative = sxxn1.get_fourth_derivative()
        sxxn1_5th_derivative = sxxn1.get_fifth_derivative()

        def gmsw_exp(x):
            y1 = sxxn1_function(x)
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            y2 = t**2
            y = y1 + y2
            return y

        def gmsw_exp_prime(x):
            y1 = sxxn1_1st_derivative(x)
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            y2 = -2 * x * t / s**1.5
            y = y1 + y2
            return y

        def gmsw_exp_2nd_derivative(x):
            y1 = sxxn1_2nd_derivative(x)
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            y2 = 6.0 * t * x**2 / s**2.5 + 2 * x**2 / s**3 - 2 * t / s**1.5
            y = y1 + y2
            return y

        def gmsw_exp_3d_derivative(x):
            y1 = sxxn1_3d_derivative(x)
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            y2 = 2 * (
                -15 * x**3 * t / s ** (7 / 2)
                - 9 * x**3 / s**4
                + 9 * x * t / s ** (5 / 2)
                + 3 * x / s**3
            )
            y = y1 + y2
            return y

        def gmsw_exp_4th_derivative(x):
            y1 = sxxn1_4th_derivative(x)
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            y2 = 2 * (
                105 * x**4 * t / s ** (9 / 2)
                + 87 * x**4 / s**5
                - 90 * x**2 * t / s ** (7 / 2)
                - 54 * x**2 / s**4
                + 9 * t / s ** (5 / 2)
                + 3 / s**3
            )
            y = y1 + y2
            return y

        def gmsw_exp_5th_derivative(x):
            y1 = sxxn1_5th_derivative(x)
            s = 1 + x**2
            t = 1.0 / np.sqrt(s) - 1
            y2 = (
                450 * t * x / s ** (7 / 2)
                - 270 * x / s**4
                - 1890 * t * x**5 / s ** (11 / 2)
                - 1950 * x**5 / s**6
                + 2100 * t * x**3 / s ** (9 / 2)
                + 1740 * x**3 / s**5
            )
            y = y1 + y2
            return y

        super().__init__(
            "GMSW",
            gmsw_exp,
            gmsw_exp_prime,
            gmsw_exp_2nd_derivative,
            gmsw_exp_3d_derivative,
            gmsw_exp_4th_derivative,
            gmsw_exp_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self, x=-8.0, interval=[-12.0, 12.0]):
        def sxxn1(x):
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = expm1**2
            return y

        def sxxn1_prime(x):
            expm1 = np.expm1(x)  # np.exp(x) - 1
            y = 2 * np.exp(x) * expm1
            return y

        def sxxn1_2nd_derivative(x):
            y = 2 * np.exp(x) * (2 * np.exp(x) - 1)
            return y

        def sxxn1_3d_derivative(x):
            y = 2 * np.exp(x) * (4 * np.exp(x) - 1)
            return y

        def sxxn1_4th_derivative(x):
            y = 2 * np.exp(x) * (8 * np.exp(x) - 1)
            return y

        def sxxn1_5th_derivative(x):
            y = 2 * np.exp(x) * (16 * np.exp(x) - 1)
            return y

        super().__init__(
            "SXXN1",
            sxxn1,
            sxxn1_prime,
            sxxn1_2nd_derivative,
            sxxn1_3d_derivative,
            sxxn1_4th_derivative,
            sxxn1_5th_derivative,
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

    The function is similar to :class:`~numericalderivative.ScaledExponentialProblem`,
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
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self, alpha=1.0e2, x=0.01, interval=[-1.0, 1.0]):
        self.alpha = alpha

        def sxxn2(x):
            y = np.exp(self.alpha * x)
            return y

        def sxxn2_prime(x):
            y = self.alpha * np.exp(self.alpha * x)
            return y

        def sxxn2_2nd_derivative(x):
            y = self.alpha**2 * np.exp(self.alpha * x)
            return y

        def sxxn2_3d_derivative(x):
            y = self.alpha**3 * np.exp(self.alpha * x)
            return y

        def sxxn2_4th_derivative(x):
            y = self.alpha**4 * np.exp(self.alpha * x)
            return y

        def sxxn2_5th_derivative(x):
            y = self.alpha**5 * np.exp(self.alpha * x)
            return y

        super().__init__(
            "SXXN2",
            sxxn2,
            sxxn2_prime,
            sxxn2_2nd_derivative,
            sxxn2_3d_derivative,
            sxxn2_4th_derivative,
            sxxn2_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self, x=0.99999, interval=[-12.0, 12.0]):
        def sxxn3(x):
            y = x**4 + 3 * x**2 - 10 * x
            return y

        def sxxn3_prime(x):
            y = 4 * x**3 + 6 * x - 10
            return y

        def sxxn3_2nd_derivative(x):
            y = 12 * x**2 + 6
            return y

        def sxxn3_3d_derivative(x):
            y = 24 * x
            return y

        def sxxn3_4th_derivative(x):
            y = 24
            return y

        def sxxn3_5th_derivative(x):
            y = 0.0
            return y

        super().__init__(
            "SXXN3",
            sxxn3,
            sxxn3_prime,
            sxxn3_2nd_derivative,
            sxxn3_3d_derivative,
            sxxn3_4th_derivative,
            sxxn3_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    """

    def __init__(self, x=1.0e-9, interval=[-12.0, 12.0]):
        def sxxn_4(x):
            y = 1.0e4 * x**3 + 0.01 * x**2 + 5 * x
            return y

        def sxxn4_prime(x):
            y = 3.0e4 * x**2 + 0.02 * x + 5
            return y

        def sxxn4_2nd_derivative(x):
            y = 6.0e4 * x + 0.02
            return y

        def sxxn4_3d_derivative(x):
            y = 6.0e4
            return y

        def sxxn4_4th_derivative(x):
            y = 0
            return y

        def sxxn4_5th_derivative(x):
            y = 0
            return y

        super().__init__(
            "SXXN4",
            sxxn_4,
            sxxn4_prime,
            sxxn4_2nd_derivative,
            sxxn4_3d_derivative,
            sxxn4_4th_derivative,
            sxxn4_5th_derivative,
            x,
            interval,
        )


class OliverProblem1(DerivativeBenchmarkProblem):
    r"""
    Create an exponential derivative benchmark problem

    See table 1 page 151 in (Oliver, 1980)

    The function is:

    .. math::

        f(x) = \exp(4 x)

    for any :math:`x`.
    The test point is :math:`x = 1`.
    This is the :class:`~numericalderivative.ScaledExponentialProblem`
    with :math:`\alpha = 4`.

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self, alpha=4.0, x=1.0, interval=[-12.0, 12.0]):
        problem = SXXNProblem2(alpha, x)

        super().__init__(
            "Oliver1",
            problem.get_function(),
            problem.get_first_derivative(),
            problem.get_second_derivative(),
            problem.get_third_derivative(),
            problem.get_fourth_derivative(),
            problem.get_fifth_derivative(),
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self, x=1.0, interval=[-12.0, 12.0]):
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

        def function_5th_derivative(x):
            y = 8.0 * np.exp(x**2) * (4 * x**4 + 20 * x**2 + 15)
            return y

        super().__init__(
            "Oliver2",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            function_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self, x=1.0, interval=[0.01, 12.0]):
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

        def function_5th_derivative(x):
            y = 4.0 / x**3
            return y

        super().__init__(
            "Oliver3",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            function_5th_derivative,
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

    Parameters
    ----------
    x : float
        The point where the derivative should be computed for a single test.
    interval : list of 2 floats
        The lower and upper bounds of the benchmark problem.
        This can be useful for benchmarking on several points.
        We must have interval[0] <= interval[1].

    References
    ----------
    - Oliver, J. (1980). An algorithm for numerical differentiation of a function of one real variable. _Journal of Computational and Applied Mathematics, 6,_ 145–160.

    """

    def __init__(self, x=1.0, interval=[0.01, 12.0]):
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

        def function_5th_derivative(x):
            y = -120.0 / x**6
            return y

        super().__init__(
            "inverse",
            function,
            function_prime,
            function_2nd_derivative,
            function_3d_derivative,
            function_4th_derivative,
            function_5th_derivative,
            x,
            interval,
        )


def build_benchmark():
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


# %%
def benchmark_method(
    function,
    derivative_function,
    test_points,
    compute_first_derivative,
    verbose=False,
):
    """
    Compute the first derivative on a set of test points

    Parameters
    ----------
    function : function
        The function.
    derivative_function : function
        The exact first derivative of the function.
    test_points : list(float)
        The list of x points where the derivative is to be evaluated
    compute_first_derivative : function
        The method to compute the first derivative.
        The calling sequence must be `f_prime_approx, f_eval = compute_first_derivative(function, x)`
        where `f_prime_approx` is the approximate value of the first derivative,
        `f_eval` is the number of function evaluations, `function` is the
        function and `x` is the point.
    verbose : bool
        Set to True to print intermediate messages.

    Returns
    -------
    average_relative_error : float, > 0
        The average relative error between the approximate first derivative
        and the exact first derivative
    average_feval : float
        The average number of function evaluations
    data : list(floats)
        For each test point, a list of 3 floats: x, relative error, number of
        function evaluations.
    """
    number_of_test_points = len(test_points)
    relative_error_array = np.zeros(number_of_test_points)
    feval_array = np.zeros(number_of_test_points)
    for i in range(number_of_test_points):
        x = test_points[i]
        try:
            f_prime_approx, number_of_function_evaluations = compute_first_derivative(
                function, x
            )
            exact_first_derivative = derivative_function(x)
            absolute_error = abs(f_prime_approx - exact_first_derivative)
            relative_error = absolute_error / abs(exact_first_derivative)
        except:
            number_of_function_evaluations = -1
            absolute_error = np.nan
            relative_error = np.nan
        if verbose:
            print(
                f"x = {x:.3f}, "
                f"abs. error = {absolute_error:.3e}, "
                f"rel. error = {relative_error:.3e}, "
                f"Func. eval. = {number_of_function_evaluations}"
            )
        relative_error_array[i] = relative_error
        feval_array[i] = number_of_function_evaluations

    average_relative_error = np.mean(relative_error_array)
    average_feval = np.mean(feval_array)
    # Compute the dataset of the benchmark
    data = []
    for i in range(number_of_test_points):
        x = test_points[i]
        data.append([x, relative_error_array[i], feval_array[i]])
    return average_relative_error, average_feval, data
