# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Various finite difference formulas.
"""

import numpy as np
import numericalderivative as nd


class FiniteDifferenceFormula:
    """Compute a derivative of the function using finite difference formula"""

    def __init__(self, function, x, args=None):
        """
        Compute a derivative of the function using finite difference formula

        This class can only be used trough its children classes.

        See also
        --------
        FirstDerivativeForward, FirstDerivativeCentral, SecondDerivativeCentral, ThirdDerivativeCentral

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.

        Returns
        -------
        None.


        Examples
        --------
        >>> import numericalderivative as nd
        >>> import numpy as np
        >>>
        >>> def scaled_exp(x):
        >>>     alpha = 1.e6
        >>>     return np.exp(-x / alpha)
        >>>
        >>> x = 1.0
        >>> formula = nd.FirstDerivativeForward(scaled_exp, x)
        >>> function = formula.get_function()
        >>> x = formula.get_x()
        """
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x

    def get_function(self):
        """
        Return the function

        Returns
        -------
        function : function
            The function
        """
        return self.function

    def get_x(self):
        """
        Return the input point

        Returns
        -------
        x : list
            The point
        """
        return self.x


class FirstDerivativeForward(FiniteDifferenceFormula):
    """Compute the first derivative using forward finite difference formula"""

    @staticmethod
    def compute_error(step, second_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the total error for forward finite difference for f'.

        The total error is the sum of the
        rounding error in the finite difference formula and the truncation
        error in the Taylor expansion:

        .. math::

            e(h) = \frac{2 \epsilon_f}{h} + \frac{h}{2} |f''(x)|

        where :math:`\epsilon_f > 0` is the absolute precision of the
        function evaluation.

        Parameters
        ----------
        step : float
            The differentiation step h.
        second_derivative_value : float
            The value of the second derivative at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        absolute_error : float
            The optimal absolute error.

        """
        # Compute rounding error
        rounding_error = 2 * absolute_precision / step
        # Compute truncation error
        truncation_error = step * abs(second_derivative_value) / 2.0
        # Compute error
        total_error = rounding_error + truncation_error
        return total_error

    @staticmethod
    def compute_step(second_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the exact optimal step for forward finite difference for f'.

        This is the step which is optimal to approximate the first derivative
        f'(x) using the forward finite difference formula (see `compute()`).
        The optimal step is (see Eq. 6 in Gill, Murray, Saunders, & Wright, 1983)):

        .. math::

            h^\star = 2  \left( \frac{\epsilon_f}{|f''(x)|} \right)^{1/2}

        where :math:`\epsilon_f > 0` is the absolute precision.
        The total absolute error corresponding to the optimal step is:

        .. math::

            e(h^\star) = 2  \left( \epsilon_f |f''(x)| \right)^{1/2}

        Parameters
        ----------
        second_derivative_value : float
            The value of the second derivative at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        optimal_step : float
            The optimal differentiation step h.
        absolute_error : float
            The optimal absolute error.

        """
        if second_derivative_value == 0.0:
            optimal_step = np.inf
        else:
            optimal_step = 2.0 * np.sqrt(
                absolute_precision / abs(second_derivative_value)
            )
        absolute_error = 2.0 * np.sqrt(
            absolute_precision * abs(second_derivative_value)
        )
        return optimal_step, absolute_error

    def __init__(self, function, x, args=None):
        """
        Compute the first derivative using forward finite difference formula

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.

        Returns
        -------
        None.

        Examples
        --------
        >>> import numericalderivative as nd
        >>> import numpy as np
        >>>
        >>> def scaled_exp(x):
        >>>     alpha = 1.e6
        >>>     return np.exp(-x / alpha)
        >>>
        >>> x = 1.0
        >>> formula = nd.FirstDerivativeForward(scaled_exp, x)
        >>> step = 1.0e-3  # A first guess
        >>> f_prime_approx = formula.compute(step)

        Compute the step using an educated guess.

        >>> second_derivative_value = 1.0  # A guess
        >>> step, absolute_error = formula.compute_step(second_derivative_value)
        >>> f_prime_approx = formula.compute(step)

        Compute the absolute error from a given step.

        >>> absolute_error = formula.compute_error(step, second_derivative_value)
        """
        super().__init__(function, x, args)

    def compute(self, step):
        r"""
        Compute an approximate first derivative using finite differences

        This method uses the formula (see Eq. 1, page 311 in (GMS&W, 1983)
        and (Faires & Burden, 1998) page 164):

        .. math::

            f'(x) = \frac{f(x + h) - f(x)}{h} + \frac{h}{2} f''(\xi)

        where :math:`h > 0` is the step and :math:`\xi \in (x, x + h)`.

        Parameters
        ----------
        step : float, > 0
            The finite difference step

        Returns
        -------
        second_derivative : float
            An estimate of f''(x).

        References
        ----------
        - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983).  Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
        - Faires, J. D., & Burden, R. L. (1998). Numerical methods, 2d edition. Cengage Learning.
        """
        step = (self.x + step) - self.x  # Magic trick
        if step <= 0.0:
            raise ValueError("Zero computed step. Cannot perform finite difference.")
        #
        x1 = self.x + step
        first_derivative = (self.function(x1) - self.function(self.x)) / step
        return first_derivative


class FirstDerivativeCentral(FiniteDifferenceFormula):
    """Compute the first derivative using central finite difference formula"""

    @staticmethod
    def compute_error(step, third_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the total error for central finite difference for f'

        The total error is the sum of the
        rounding error in the finite difference formula and the truncation
        error in the Taylor expansion:

        .. math::

            e(h) = \frac{\epsilon_f}{h} + \frac{h^2}{6} |f'''(x)|

        where :math:`\epsilon_f > 0` is the absolute precision of the function
        evaluation.

        Parameters
        ----------
        step : float
            The differentiation step h.
        third_derivative_value : float
            The value of the third derivative at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        absolute_error : float
            The optimal absolute error.

        """
        # Compute rounding error
        rounding_error = absolute_precision / step
        # Compute truncation error
        truncation_error = step**2 * abs(third_derivative_value) / 6.0
        # Compute error
        total_error = rounding_error + truncation_error
        return total_error

    @staticmethod
    def compute_step(third_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the exact optimal step for central finite difference for f'.

        This is the step which is optimal to approximate the first derivative
        f'(x) using the centered finite difference formula (see `compute()`).
        The optimal step is:

        .. math::

            h^\star = \left( \frac{3 \epsilon_f}{|f'''(x)|} \right)^{1/3}

        The total absolute error corresponding to the optimal step is:

        .. math::

            e(h^\star) = \frac{3^{\frac{2}{3}}}{2} \left( \epsilon_f^2 |f'''(x)| \right)^{1/3}

        Parameters
        ----------
        third_derivative_value : float
            The value of the third derivative at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        optimal_step : float
            The optimal differentiation step h.
        absolute_error : float
            The optimal absolute error.

        """
        if third_derivative_value == 0.0:
            optimal_step = np.inf
        else:
            optimal_step = (3.0 * absolute_precision / abs(third_derivative_value)) ** (
                1.0 / 3.0
            )
        absolute_error = (
            (3.0 ** (2.0 / 3.0))
            / 2.0
            * (absolute_precision**2 * abs(third_derivative_value)) ** (1.0 / 3.0)
        )
        return optimal_step, absolute_error

    def __init__(self, function, x, args=None):
        """
        Compute the first derivative using central finite difference formula

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.

        Returns
        -------
        None.

        """
        super().__init__(function, x, args)

    def compute(self, step):
        r"""
        Compute first derivative using central finite difference.

        This is based on the central finite difference formula
        (see (Faires & Burden, 1998) page 166) :

        .. math::

            f'(x) = \frac{f(x + h) - f(x - h)}{2h} - \frac{h^2}{6} f'''(\xi)

        where :math:`h > 0` is the step and :math:`\xi \in (x, x + h)`.

        Parameters
        ----------
        step : float, > 0
            The finite difference step

        Returns
        -------
        first_derivative : float
            The approximate first derivative at point x.

        References
        ----------
        - Faires, J. D., & Burden, R. L. (1998). Numerical methods, 2d edition. Cengage Learning.
        """
        step = (self.x + step) - self.x  # Magic trick
        if step <= 0.0:
            raise ValueError("Zero computed step. Cannot perform finite difference.")
        x1 = self.x + step
        x2 = self.x - step
        first_derivative = (self.function(x1) - self.function(x2)) / (x1 - x2)
        return first_derivative


class SecondDerivativeCentral(FiniteDifferenceFormula):
    """Compute the second derivative using central finite difference formula"""

    @staticmethod
    def compute_error(step, fourth_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the total error for central finite difference for f''

        The total error is the sum of the
        rounding error in the finite difference formula and the truncation
        error in the Taylor expansion:

        .. math::

            e(h) = \frac{4 \epsilon_f}{h^2} + \frac{h^2}{12} \left|f^{(4)}(x)\right|

        where :math:`\epsilon_f > 0` is the absolute precision of the function
        evaluation.

        Parameters
        ----------
        step : float
            The differentiation step h.
        fourth_derivative_value : float
            The value of the fourth derivative at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        absolute_error : float
            The optimal absolute error.

        """
        # Compute rounding error
        rounding_error = 4 * absolute_precision / step**2
        # Compute truncation error
        truncation_error = step**2 * abs(fourth_derivative_value) / 12.0
        # Compute error
        total_error = rounding_error + truncation_error
        return total_error

    @staticmethod
    def compute_step(fourth_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the optimal step for the central finite difference for f''.

        This step minimizes the total error of the second derivative
        central finite difference (see `compute()`).
        The optimal step is (see Eq. 8bis, page 314 in Gill, Murray, Saunders, & Wright, 1983)):

        .. math::

            h^\star = \left(\frac{48 \epsilon_f}{|f^{(4)}(x)|} \right)^{1/4}

        The total absolute error corresponding to the optimal step is:

        .. math::

            e(h^\star) = \frac{2 \sqrt{3}}{3} \sqrt{\epsilon_f |f^{(4)}(x)|}

        Parameters
        ----------
        fourth_derivative_value : float
            The fourth derivative f^(4) at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        optimal_step : float
            The finite difference step.
        absolute_error : float
            The absolute error.

        """
        #
        if fourth_derivative_value == 0.0:
            optimal_step = np.inf
        else:
            optimal_step = (
                48.0 * absolute_precision / abs(fourth_derivative_value)
            ) ** (1.0 / 4.0)
        absolute_error = (
            2.0
            * np.sqrt(3.0)
            / 3.0
            * np.sqrt(absolute_precision * abs(fourth_derivative_value))
        )
        return optimal_step, absolute_error

    def __init__(self, function, x, args=None):
        """
        Compute the second derivative using central finite difference formula

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.

        Returns
        -------
        None.

        """
        super().__init__(function, x, args)

    def compute(self, step):
        r"""
        Compute an approximate second derivative using finite differences.

        The formula is (see (Faires & Burden, 1998) page 171):

        .. math::

            f''(x) = \frac{f(x + h) - 2 f(x) + f(x - h)}{h^2}
                     + \frac{h^2}{12} f^{(4)}(\xi)

        where :math:`h > 0` is the step and :math:`\xi \in (x, x + h)`.

        This second derivative can be used to compute an
        approximate optimal step for the forward first finite difference.
        Please use compute_step() to do this.

        Parameters
        ----------
        step : float
            The step.

        Returns
        -------
        second_derivative : float
            An estimate of f''(x).

        References
        ----------
        - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
        - Faires, J. D., & Burden, R. L. (1998). Numerical methods, 2d edition. Cengage Learning.
        """
        step = (self.x + step) - self.x  # Magic trick
        if step <= 0.0:
            raise ValueError("Zero computed step. Cannot perform finite difference.")
        # Eq. 8 page 314 in (GMS&W, 1983)
        second_derivative = (
            self.function(self.x + step)
            - 2 * self.function(self.x)
            + self.function(self.x - step)
        ) / (step**2)
        return second_derivative


class ThirdDerivativeCentral(FiniteDifferenceFormula):
    """Compute the third derivative using central finite difference formula"""

    @staticmethod
    def compute_error(step, fifth_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the total error for central finite difference for f'''

        The total error is the sum of the
        rounding error in the finite difference formula and the truncation
        error in the Taylor expansion:

        .. math::

            e(h) = \frac{3 \epsilon_f}{h^3} + \frac{h^2}{4} \left|f^{(5)}(x)\right|

        where :math:`\epsilon_f > 0` is the absolute precision of the function
        evaluation.

        Parameters
        ----------
        step : float
            The differentiation step h.
        fifth_derivative_value : float
            The value of the fifth derivative at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        absolute_error : float
            The optimal absolute error.

        """
        # Compute rounding error
        rounding_error = 3 * absolute_precision / step**3
        # Compute truncation error
        truncation_error = step**2 * abs(fifth_derivative_value) / 4
        # Compute error
        total_error = rounding_error + truncation_error
        return total_error

    @staticmethod
    def compute_step(fifth_derivative_value=1.0, absolute_precision=1.0e-16):
        r"""
        Compute the optimal step for the central finite difference for f'''.

        This step minimizes the total error of the second derivative
        central finite difference (see `compute()`).
        The optimal step is:

        .. math::

            h^\star = \left(\frac{18 \epsilon_f}{|f^{(5)}(x)|} \right)^{1/5}

        The total absolute error corresponding to the optimal step is:

        .. math::

            e(h^\star) = \frac{5}{12} 2^{2/5} 3^{4/5} \epsilon_f^{2/5} \left|f^{(5)}(x)\right|^{3/5}

        Parameters
        ----------
        fifth_derivative_value : float
            The fourth derivative f^(4) at point x.
        absolute_precision : float, optional
            The absolute error of the function f at the point x.
            This is equal to abs(relative_precision * f(x)) where
            relative_precision is the relative accuracy and f(x) is the function
            value of f at point x.

        Returns
        -------
        optimal_step : float
            The finite difference step.
        absolute_error : float
            The absolute error.

        """
        #
        if fifth_derivative_value == 0.0:
            optimal_step = np.inf
        else:
            optimal_step = (
                18.0 * absolute_precision / abs(fifth_derivative_value)
            ) ** (1.0 / 5.0)
        absolute_error = (
            5 * 2**(2/5) * 3**(4/5) / 12
            * absolute_precision ** (2/5) * abs(fifth_derivative_value) ** (3/5)
        )
        return optimal_step, absolute_error

    def __init__(self, function, x, args=None):
        """
        Compute the third derivative using central finite difference formula

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.

        Returns
        -------
        None.

        """
        super().__init__(function, x, args)

    def compute(self, step):
        r"""
        Estimate the 3d derivative f'''(x) using finite differences.

        This is based on the central finite difference formula
        (see (Betts, 2010) table 1.7 page 47 and (Dumontet & Vignes, 1977) eq. 27
        page 19):

        .. math::

            f^{(3)}(x) = &\frac{- f(x - 2h)  + 2 f(x - h) - 2 f(x + h) + f(x + 2h)}{2h^3} \\
                         &- \frac{h^2}{4} f^{(5)}(x) + O(h^3)

        where :math:`h > 0` is the step.

        Parameters
        ----------
        step : float
            The step used for the finite difference formula.

        Returns
        -------
        third_derivative : float
            The approximate f'''(x).

        References
        ----------
        - Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.
        - Betts, J. T. (2010). _Practical methods for optimal control and estimation using nonlinear programming_. Society for Industrial and Applied Mathematics.
        """
        t = np.zeros(4)
        t[0] = self.function(self.x + 2 * step)
        t[1] = -self.function(self.x - 2 * step)  # Fixed wrt paper
        t[2] = -2.0 * self.function(self.x + step)
        t[3] = 2.0 * self.function(self.x - step)  # Fixed wrt paper
        third_derivative = np.sum(t) / (2 * step**3)  # Eq. 27 and 35 in (D&V, 1977)
        return third_derivative
