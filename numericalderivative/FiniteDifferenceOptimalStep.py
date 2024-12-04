# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Compute the optimum step for finite differences

"""
import numpy as np


class FiniteDifferenceOptimalStep:
    """
    Compute the optimum step for finite differences

    Parameters
    ----------
    absolute_precision : float, optional
        The absolute error of the function f at the point x.
        This is equal to abs(relative_precision * f(x)) where
        relative_precision is the relative accuracy and f(x) is the function
        value of f at point x.

    References
    ----------
    - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.

    """
    def __init__(self, absolute_precision=1.0e-16):
        if absolute_precision <= 0.0:
            raise ValueError(
                f"The absolute precision = {absolute_precision} " "but it must be > 0"
            )
        self.absolute_precision = absolute_precision

    def compute_step_first_derivative_forward(self, second_derivative_value):
        """
        Compute the exact optimal step for forward finite difference for f'.

        This is the step which is optimal to approximate the first derivative
        f'(x) using the forward finite difference formula :

        f'(x) ~ (f(x + h) - f(x)) / h

        Parameters
        ----------
        second_derivative_value : float
            The value of the second derivative at point x.

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
                self.absolute_precision / abs(second_derivative_value)
            )
        absolute_error = 2.0 * np.sqrt(
            self.absolute_precision * abs(second_derivative_value)
        )
        return optimal_step, absolute_error

    def compute_step_first_derivative_central(self, third_derivative_value):
        """
        Compute the exact optimal step for central finite difference for f'.

        This is the step which is optimal to approximate the first derivative
        f'(x) using the centered finite difference formula :

        f'(x) ~ (f(x + h) - f(x - h)) / (2 * h)

        Parameters
        ----------
        third_derivative_value : float
            The value of the third derivative at point x.

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
            optimal_step = (
                3.0 * self.absolute_precision / abs(third_derivative_value)
            ) ** (1.0 / 3.0)
        absolute_error = (
            (3.0 ** (2.0 / 3.0))
            / 2.0
            * self.absolute_precision ** (2.0 / 3.0)
            * abs(third_derivative_value) ** (1.0 / 3.0)
        )
        return optimal_step, absolute_error

    def compute_step_second_derivative(self, fourth_derivative_value):
        """
        Compute the optimal step for the finite difference for f''.

        This step minimizes the total error of the second derivative
        finite difference :

        f''(x) ~ (f(x + k) - 2 f(x) + f(x - k)) / (k ** 2)

        Parameters
        ----------
        fourth_derivative_value : float
            The fourth derivative f^(4) at point x.

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
            optimal_step = (12.0 * self.absolute_precision / abs(fourth_derivative_value)) ** (
                1.0 / 4.0
            )
        absolute_error = 2.0 * np.sqrt(
            self.absolute_precision * abs(fourth_derivative_value) / 12.0
        )
        return optimal_step, absolute_error

