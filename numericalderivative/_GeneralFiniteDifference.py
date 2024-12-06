# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Creates a finite difference formula of arbitrary differentiation differentiation_order or accuracy
"""

import numpy as np
import sys
import numericalderivative as nd
import math


class GeneralFiniteDifference:
    """Create a general finite difference formula"""

    def __init__(
        self,
        function,
        x,
        differentiation_order,
        formula_accuracy,
        direction="centered",
        args=None,
    ):
        """
        Create a general finite difference formula

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
            For the central F.D. formula, if the differentiation order is even,
            then the formula accuracy is necessarily even.
            If required increase the formula accuracy by 1 unit.
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

        References
        ----------
        - David Eberly. « Derivative approximation by finite differences ». 2008.
        - Nicholas Maxwell. « Notes on the derivation of Finite Difference kernels on regularly spaced grids, using arbitrary sample points ». 2010.
        - Bengt Fornberg. « Classroom Note :Calculation of Weights in Finite Difference Formulas ». In : SIAM Review 40.3 (1998), p. 685-691.
        - B. Fornberg. « Finite difference method ». In : Scholarpedia 6.10 (2011), p. 9685
        - H.Z. Hassan, A.A. Mohamad et G.E. Atteia. « An algorithm for the finite difference approximation of derivatives with arbitrary degree and order of accuracy ». In : Journal of Computational and Applied Mathematics 236.10 (2012), p. 2622-2631. issn : 0377-0427.
        """
        if differentiation_order <= 0:
            raise ValueError(f"Invalid differentiation order {differentiation_order}")
        self.differentiation_order = differentiation_order
        if formula_accuracy <= 0:
            raise ValueError(f"Invalid formula accuracy {formula_accuracy}")
        if (
            direction != "forward"
            and direction != "backward"
            and direction != "centered"
        ):
            raise ValueError(f"Invalid direction {direction}.")
        self.direction = direction
        if (
            self.direction == "centered"
            and self.differentiation_order % 2 == 0
            and self.formula_accuracy % 2 == 1
        ):
            raise ValueError(
                f"Invalid accuracy for a centered formula with even differentiation order."
                f" Please increase formula_accuracy by 1."
            )
        self.formula_accuracy = formula_accuracy
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x
        #
        # Setup the formula
        _ = self.compute_indices()
        # Compute the coefficients
        _ = self.compute_coefficients()

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
            self.imin = 0
            self.imax = self.differentiation_order + self.formula_accuracy - 1
        elif self.direction == "backward":
            self.imin = -(self.differentiation_order + self.formula_accuracy - 1)
            self.imax = 0
        elif self.direction == "centered":
            if (self.differentiation_order + self.formula_accuracy) % 2 == 0:
                raise ValueError(
                    "d+formula_accuracy must be odd for a centered formula."
                )
            self.imax = (self.differentiation_order + self.formula_accuracy - 1) // 2
            self.imin = -self.imax
        else:
            raise ValueError(f"Invalid direction {self.direction}")
        return (self.imin, self.imax)

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
        imin, imax = self.compute_indices()
        indices = list(range(imin, imax + 1))
        A = np.vander(indices, increasing=True).T
        # Compute right-hand side
        b = np.zeros((self.differentiation_order + self.formula_accuracy))
        b[self.differentiation_order] = 1.0
        # Solve
        self.coefficients = np.linalg.solve(A, b)
        return self.coefficients

    def compute_step(
        self,
        absolute_precision=sys.float_info.epsilon,
        higher_order_derivative_value=1.0,
    ):
        """
        Computes the optimal step

        See (Baudin, 2023) eq. (9.16) page 224.

        Parameters
        ----------
        absolute_precision : float, > 0
            The absolute precision of the function evaluation
        higher_order_derivative_value : float
            The value of the derivative of order differentiation_order + formula_accuracy.
            For example, if differentiation_order = 2 and the formula_accuracy = 3, then
            this must be the derivative of order 3 + 2 = 5.

        Returns
        -------
        step : float
            The finite difference step.

        References
        ----------
        - M. Baudin (2023). Méthodes numériques. Dunod.
        """
        # TODO: fix the constant introduced by the Taylor formula truncation error
        step = (
            self.differentiation_order
            * absolute_precision
            / (self.formula_accuracy * abs(higher_order_derivative_value))
        ) ** (1.0 / (self.differentiation_order + self.formula_accuracy))
        return step

    def finite_differences(self, step):
        """
        Computes the degree d derivative of f at point x.

        Uses a finite difference formula.
        If direction is "centered", if differentiation_order is even and if formula_accuracy is odd,
        then the order of precision is actually formula_accuracy + 1.

        Parameters
        ----------
        step : float
            The finite difference step.

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
        # Compute the function values
        y = np.zeros((self.differentiation_order + self.formula_accuracy))
        for i in range(self.imin, self.imax + 1):
            y[i - self.imin] = self.function(self.x + i * step)
        # Apply the formula
        z = 0.0
        for i in range(self.imin, self.imax + 1):
            z += self.coefficients[i - self.imin] * y[i - self.imin]
        factor = (
            math.factorial(self.differentiation_order)
            / step**self.differentiation_order
        )
        z *= factor
        return z
