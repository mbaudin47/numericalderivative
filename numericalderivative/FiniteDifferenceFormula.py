# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Provides various finite difference formulas.
"""

import numpy as np
from .NumericalDerivative import NumericalDerivative


class FiniteDifferenceFormula(NumericalDerivative):
    def __init__(self, function, x, args=None) -> None:
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
        super().__init__(function, x, args)

    def function_eval(self, x):
        """
        Evaluates the function at point x

        Manages the extra input arguments, if any.

        Parameters
        ----------
        x : float
            The input point.

        Returns
        ----------
        y : float
            The output point.
        """
        if self.args is None:
            function_value = self.function(x)
        else:
            function_value = self.function(x, *self.args)
        self.number_of_function_evaluations += 1
        return function_value

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        ----------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        return self.number_of_function_evaluations

    def compute_third_derivative(self, step):
        """
        Estimate the 3d derivative f'''(x) using finite differences.

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
        - Dumontet, J., & Vignes, J. (1977). 
          Détermination du pas optimal dans le calcul des dérivées sur ordinateur. 
          RAIRO. Analyse numérique, 11 (1), 13-25.
        """
        t = np.zeros(4)
        t[0] = self.function_eval(self.x + 2 * step)
        t[1] = -self.function_eval(self.x - 2 * step)  # Fixed wrt paper
        t[2] = -2.0 * self.function_eval(self.x + step)
        t[3] = 2.0 * self.function_eval(self.x - step)  # Fixed wrt paper
        third_derivative = np.sum(t) / (2 * step**3)  # Eq. 27 and 35 in (D&V, 1977)
        return third_derivative

    def compute_first_derivative_central(self, step):
        """
        Compute first derivative using central finite difference.

        This is based on the central finite difference formula:

        f'(x) ~ (f(x + h) - f(x - h)) / (2h)

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
        first_derivative = (self.function_eval(x1) - self.function_eval(x2)) / (x1 - x2)
        return first_derivative

    def compute_first_derivative_forward(self, step):
        """
        Compute an approximate first derivative using finite differences

        This method uses the formula:

        f'(x) ~ (f(x + h) - f(x)) / h

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
        - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). 
          Computing forward-difference intervals for numerical optimization. 
          SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
        """
        step = (self.x + step) - self.x  # Magic trick
        if step <= 0.0:
            raise ValueError("Zero computed step. Cannot perform finite difference.")
        # Eq. 1, page 311 in (GMS&W, 1983)
        x1 = self.x + step
        first_derivative = (self.function(x1) - self.function(self.x)) / step
        return first_derivative

    def compute_second_derivative_central(self, step):
        """
        Compute an approximate second derivative using finite differences.

        The formula is:

        f''(x) ~ (f(x + k) - 2 f(x) + f(x - k)) / k^2

        This second derivative can be used to compute an
        approximate optimal step for the forward first finite difference.
        Please use FiniteDifferenceOptimalStep.compute_step_first_derivative_forward()
        to do this.

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
        - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). 
          Computing forward-difference intervals for numerical optimization. 
          SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
        """
        step = (self.x + step) - self.x  # Magic trick
        if step <= 0.0:
            raise ValueError("Zero computed step. Cannot perform finite difference.")
        # Eq. 8 page 314 in (GMS&W, 1983)
        second_derivative = (
            self.function_eval(self.x + step)
            - 2 * self.function_eval(self.x)
            + self.function_eval(self.x - step)
        ) / (step**2)
        return second_derivative
