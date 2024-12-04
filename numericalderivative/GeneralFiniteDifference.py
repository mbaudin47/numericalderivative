# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Creates a finite difference formula of arbitrary differentiation differentiation_order or accuracy
"""

import numpy as np
import sys
import numericalderivative as nd


class GeneralFiniteDifference:
    """
    Defines a general finite difference formula

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is to be evaluated.
    differentiation_order : int
        The order of the derivative.
        For example differentiation_order = 1 is the first derivative.
    formula_accuracy : int
        The order of precision of the formula.
    direction : str, optional
        The direction of the formula.
        The direction can be "forward", "backward" or "centered".
        The default is "centered".
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    """

    def __init__(
        self,
        function,
        x,
        differentiation_order,
        formula_accuracy,
        direction="centered",
        args=None,
    ) -> None:
        if differentiation_order <= 0:
            raise ValueError(f"Invalid differentiation order {differentiation_order}")
        self.differentiation_order = differentiation_order
        if formula_accuracy <= 0:
            raise ValueError(f"Invalid formula accuracy {formula_accuracy}")
        self.formula_accuracy = formula_accuracy
        if (
            direction != "forward"
            and direction != "backward"
            and direction != "centered"
        ):
            raise ValueError(f"Invalid direction {direction}.")
        self.direction = direction
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x

    def compute_indices(self):
        """
        Computes the min and max indices for a finite difference formula.

        This function is used by compute_coefficients() to compute the
        derivative of arbitrary differentiation_order and arbitrary differentiation_order of accuracy.

        Parameters
        ----------
        None

        Raises
        ------
        ValueError
            If direction is "centered", d + formula_accuracy must be odd.

        Returns
        -------
        imin : int
            The minimum indice of the f.d. formula.
        imax : int
            The maximum indice of the f.d. formula.

        Examples
        --------

        >>> import numericalderivative as nd
        >>>
        >>> def scaled_exp(x):
        >>>     alpha = 1.e6
        >>>     return np.exp(-x / alpha)
        >>>
        >>> x = 1.0e-2
        >>> differentiation_order = 3  # Compute f'''
        >>> formula_accuracy = 6  # Use differentiation_order 6 formula
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy).compute_indices()
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "forward").compute_indices()
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "backward").compute_indices()
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "centered").compute_indices()
        """
        if self.direction == "forward":
            imin = 0
            imax = self.differentiation_order + self.formula_accuracy - 1
        elif self.direction == "backward":
            imin = -(self.differentiation_order + self.formula_accuracy - 1)
            imax = 0
        elif self.direction == "centered":
            if (self.differentiation_order + self.formula_accuracy) % 2 == 0:
                raise ValueError(
                    "d+formula_accuracy must be odd for a centered formula."
                )
            imax = (self.differentiation_order + self.formula_accuracy - 1) // 2
            imin = -imax
        else:
            raise ValueError(f"Invalid direction {self.direction}")
        return (imin, imax)

    def compute_coefficients(self):
        """
        Computes the coefficients of the finite difference formula.

        Parameters
        ----------
        None

        Raises
        ------
        ValueError
            If direction is "centered", differentiation_order + formula_accuracy must be odd.

        Returns
        -------
        c : np.array(differentiation_order + formula_accuracy)
            The coefficicients of the finite difference formula.

        Examples
        --------
        >>> import numericalderivative as nd
        >>>
        >>> def scaled_exp(x):
        >>>     alpha = 1.e6
        >>>     return np.exp(-x / alpha)
        >>>
        >>> x = 1.0e-2
        >>> differentiation_order = 3  # Compute f'''
        >>> formula_accuracy = 6  # Use differentiation_order 6 formula
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy).compute_coefficients()
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "forward").compute_coefficients()
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "backward").compute_coefficients()
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "centered").compute_coefficients()
        """
        # Compute matrix
        imin, imax = nd.compute_indices()
        indices = list(range(imin, imax + 1))
        A = np.vander(indices, increasing=True).T
        # Compute right-hand side
        b = np.zeros((self.differentiation_order + self.formula_accuracy))
        b[self.differentiation_order] = 1.0
        # Solve
        c = np.linalg.solve(A, b)
        return c

    def finite_differences(self, h=None):
        """
        Computes the degree d derivative of f at point x.

        Uses a finite difference formula with differentiation_order formula_accuracy.
        If the step is not provided, uses the approximately optimal
        step size.
        If direction is "centered", if d is even and if formula_accuracy is odd,
        then the differentiation_order of precision is actually formula_accuracy + 1.

        Parameters
        ----------
        h : float
            The step.

        Raises
        ------
        ValueError
            If direction is "centered", d + formula_accuracy must be odd.

        Returns
        -------
        z : float
            A approximation of the d-th derivative of f at point x.

        Examples
        --------
        >>> import numpy as np
        >>> x = 1.0
        >>> differentiation_order = 3  # Compute f'''
        >>> formula_accuracy = 2  # Use differentiation_order 2 precision
        >>> y = nd.GeneralFiniteDifference(np.sin, x, differentiation_order, formula_accuracy).finite_differences()
        >>> y = nd.GeneralFiniteDifference(np.sin, x, differentiation_order, formula_accuracy, "forward").finite_differences()

        """
        # Compute the optimal step size
        if h is None:
            if (
                self.direction == "centered"
                and self.differentiation_order % 2 == 0
                and self.formula_accuracy % 2 == 1
            ):
                eps = sys.float_info.epsilon
                h = eps ** (
                    1.0 / (self.differentiation_order + self.formula_accuracy + 1)
                )
            else:
                eps = sys.float_info.epsilon
                h = eps ** (1.0 / (self.differentiation_order + self.formula_accuracy))
        # Compute the function values
        imin, imax = np.compute_indices()
        y = np.zeros((self.differentiation_order + self.formula_accuracy))
        for i in range(imin, imax + 1):
            y[i - imin] = self.function(self.x + i * h)
        # Compute the coefficients
        c = np.compute_coefficients()
        # Apply the formula
        z = 0.0
        for i in range(imin, imax + 1):
            z += c[i - imin] * y[i - imin]
        factor = (
            np.math.factorial(self.differentiation_order)
            / h**self.differentiation_order
        )
        z *= factor
        return z
