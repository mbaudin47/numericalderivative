# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
The root class for all algorithms.
"""


class NumericalDerivative:
    """
    Compute an approximately optimal step for the forward finite difference first derivative.

    This is the root class and can only be used from its children.

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is approximated.
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    None.

    See also
    --------
    FiniteDifferenceFormula, SteplemanWinarsky, GillMurraySaundersWright, DumontetVignes

    """
    def __init__(
        self,
        function,
        x,
        args=None,
    ):
        self.function = function
        self.x = x
        self.args = args
        self.number_of_function_evaluations = 0

    def function_eval(self, x):
        """
        Evaluates the function at point x

        Manages the extra input arguments, if any.

        Parameters
        ----------
        x : float
            The input point.

        Returns
        -------
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
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        return self.number_of_function_evaluations

    def compute_first_derivative(self, step):
        """
        Compute an approximate first derivative using finite differences

        Parameters
        ----------
        step : float, > 0
            The finite difference step

        Returns
        -------
        f_prime_approx : float, > 0
            The approximate first derivative of the function
        """
        if step <= 0.0:
            raise ValueError(f"The step = {step} must be greater than zero.")
        raise NotImplementedError()

    def compute_step(self):
        """
        Compute an approximate optimal step

        Returns
        -------
        step : float, > 0
            The finite difference step for the first derivative
        number_of_iterations : int
            The number of iterations used to compute the step
        """
        raise NotImplementedError()
