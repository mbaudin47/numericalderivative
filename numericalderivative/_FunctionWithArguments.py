# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
A function with extra arguments.
"""


class FunctionWithArguments:
    """
    Evaluate a function with extra arguments.

    Parameters
    ----------
    function : function
        The function to differentiate.
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

    Examples
    --------
    Define and evaluate a function with arguments.

    >>> import numericalderivative as nd
    >>>
    >>> def scaled_exp_with_2_args(x, alpha, beta):
    >>>     return beta * np.exp(-x / alpha)
    >>>
    >>> alpha = 1.e6
    >>> beta = 2.0
    >>> args = [alpha, beta]
    >>> function = nd.FunctionWithArguments(
    ...     scaled_exp_with_2_args, args
    >>> )
    >>> y = function(x)
    >>> for i in range(10):
    >>>     y = function(x)
    >>> counter = function.get_number_of_evaluations()

    See also
    --------
    FiniteDifferenceFormula, SteplemanWinarsky, GillMurraySaundersWright, DumontetVignes

    """

    def __init__(
        self,
        function,
        args=None,
    ):
        self.function = function
        self.args = args
        self.number_of_evaluations = 0

    def __call__(self, x):
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
        self.number_of_evaluations += 1
        return function_value

    def get_function(self):
        """
        Return the function

        Returns
        -------
        function : function
            The function
        """
        return self.function

    def get_args(self):
        """
        Return the extra arguments

        Returns
        -------
        args : list
            The arguments
        """
        return self.args

    def get_number_of_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_evaluations : int
            The number of function evaluations.
        """
        return self.number_of_evaluations
