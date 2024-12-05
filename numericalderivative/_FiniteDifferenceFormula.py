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
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        return self.function.get_number_of_evaluations()


class FirstDerivativeForward(FiniteDifferenceFormula):
    """Compute the first derivative using forward finite difference formula"""

    @staticmethod
    def compute_step(second_derivative_value, absolute_precision=1.0e-16):
        r"""
        Compute the exact optimal step for forward finite difference for f'.

        This is the step which is optimal to approximate the first derivative
        f'(x) using the forward finite difference formula :

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x)}{h}

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
        # Eq. 6 in Gill, Murray, Saunders, & Wright (1983).
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

        """
        super().__init__(function, x, args)

    def compute(self, step):
        r"""
        Compute an approximate first derivative using finite differences

        This method uses the formula:

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x)}{h}

        where :math:`h` is the step.

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
        """
        step = (self.x + step) - self.x  # Magic trick
        if step <= 0.0:
            raise ValueError("Zero computed step. Cannot perform finite difference.")
        # Eq. 1, page 311 in (GMS&W, 1983)
        x1 = self.x + step
        first_derivative = (self.function(x1) - self.function(self.x)) / step
        return first_derivative


class FirstDerivativeCentral(FiniteDifferenceFormula):
    """Compute the first derivative using central finite difference formula"""

    @staticmethod
    def compute_step(third_derivative_value, absolute_precision=1.0e-16):
        r"""
        Compute the exact optimal step for central finite difference for f'.

        This is the step which is optimal to approximate the first derivative
        f'(x) using the centered finite difference formula :

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x - h)}{2 h}

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
            * absolute_precision ** (2.0 / 3.0)
            * abs(third_derivative_value) ** (1.0 / 3.0)
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

        This is based on the central finite difference formula:

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}

        where :math:`h` is the step.

        Parameters
        ----------
        step : float, > 0
            The finite difference step

        Returns
        -------
        first_derivative : float
            The approximate first derivative at point x.
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
    def compute_step(fourth_derivative_value, absolute_precision=1.0e-16):
        r"""
        Compute the optimal step for the finite difference for f''.

        This step minimizes the total error of the second derivative
        central finite difference :

        .. math::

            f''(x) \approx \frac{f(x + k) - 2 f(x) + f(x - k)}{k^2}

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
        # Eq. 8bis, page 314 in Gill, Murray, Saunders, & Wright (1983).
        if fourth_derivative_value == 0.0:
            optimal_step = np.inf
        else:
            optimal_step = (
                12.0 * absolute_precision / abs(fourth_derivative_value)
            ) ** (1.0 / 4.0)
        absolute_error = 2.0 * np.sqrt(
            absolute_precision * abs(fourth_derivative_value) / 12.0
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

        The formula is:

        .. math::

            f''(x) \approx \frac{f(x + h) - 2 f(x) + f(x - h)}{h^2}

        where :math:`h` is the step.

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
        Estimate the 3d derivative f'''(x) using finite differences.

        This is based on the central finite difference formula:

        .. math::

            f^{(3)}(x) \approx \frac{f(x + 2h) - f(x - 2h) -2 f(x + h) + 2 f(x - h)}{2h^3}

        where :math:`h` is the step.

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
        """
        t = np.zeros(4)
        t[0] = self.function(self.x + 2 * step)
        t[1] = -self.function(self.x - 2 * step)  # Fixed wrt paper
        t[2] = -2.0 * self.function(self.x + step)
        t[3] = 2.0 * self.function(self.x - step)  # Fixed wrt paper
        third_derivative = np.sum(t) / (2 * step**3)  # Eq. 27 and 35 in (D&V, 1977)
        return third_derivative
