# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Class to define Stepleman and Winarsky algorithm
"""

import numpy as np
import numericalderivative as nd
import math


class SteplemanWinarsky:
    r"""
    Compute an approximately optimal step for the central F.D. formula of the first derivative

    Uses central finite difference to compute an approximate value of f'(x).
    The approximate optimal step for f'(x) is computed using a monotony property.

    The central F.D. is:

    .. math::

        f'(x) \approx d(h) := \frac{f(x + h) - f(x - h)}{2 h}

    where :math:`f` is the function, :math:`x \in \mathbb{R}` is the
    input point and :math:`h > 0` is the step.
    In the previous equation, the function :math:`d` is the central finite difference
    formula which is considered in this method.
    The goal of the method is to compute an approximately optimal
    :math:`h^\star` for the central F.D. formula using function evaluations
    only.
    Alternatively, :class:`~numericalderivative.DumontetVignes` can be used.

    The method introduces the function :math:`g` defined by:

    .. math::

        g(t) = f(x + t) - f(x - t)

    for any :math:`t \in \mathbb{R}`.
    We introduce the monotonic sequence of step sizes :math:`\{h_i\}_{i \geq 0}` defined
    by the equation:

    .. math::

        h_{i + 1} = \frac{h_i}{\beta}, \quad i=0,1,2,...

    Therefore, under some smoothness hypotheses on :math:`g`,
    there exists :math:`N \in \mathbb{N}` such that for any
    :math:`i \geq N`, we have:

    .. math::

        \left|d(h_{i + 1}) - d(h_i)\right|
        \leq \left|d(h_{i}) - d(h_{i - 1})\right|.

    The previous theorem states that the sequence
    :math:`\left(\left|d(h_{i}) - d(h_{i - 1})\right|\right)_{i \geq N}`
    is monotonic in exact arithmetic.

    The method starts from an initial step :math:`h_0 > 0`.
    It then reduces this step iteratively until the monotonicity property
    is broken.

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is to be evaluated.
    beta : float, > 1.0
        The reduction factor of h at each iteration.
        A value of beta closer to 1 can improve the accuracy of the optimum
        step, but may increase the number of iterations.
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    verbose : bool, optional
        Set to True to print intermediate messages.

    Returns
    -------
    None.

    References
    ----------
    - Adaptive numerical differentiation. R. S. Stepleman and N. D. Winarsky. Journal: Math. Comp. 33 (1979), 1257-1264

    See also
    --------
    FirstDerivativeCentral, SteplemanWinarskyInitialize, DumontetVignes

    Examples
    --------
    Compute the step of a badly scaled function.

    >>> import numericalderivative as nd
    >>>
    >>> def scaled_exp(x):
    >>>     alpha = 1.e6
    >>>     return np.exp(-x / alpha)
    >>>
    >>> x = 1.0e-2
    >>> initial_step = 1.0e8
    >>> algorithm = nd.SteplemanWinarsky(scaled_exp, x)
    >>> h_optimal, number_of_iterations = algorithm.find_step(initial_step)
    >>> f_prime_approx = algorithm.compute_first_derivative(h_optimal)
    """

    def __init__(self, function, x, beta=4.0, args=None, verbose=False):
        self.first_derivative_central = nd.FirstDerivativeCentral(function, x, args)
        self.functionwithargs = nd.FunctionWithArguments(function, args)
        self.step_history = []
        self.function = function
        self.x = x
        self.args = args
        self.verbose = verbose
        if beta <= 1.0:
            raise ValueError(f"beta must be greater than 1. Here beta = {beta}.")
        self.beta = beta
        return

    def find_step(self, initial_step, iteration_maximum=53):
        r"""
        Compute an approximate optimum step for central derivative using monotony properties.

        This function computes an approximate optimal step h for the central
        finite difference.

        Parameters
        ----------
        initial_step : float, > 0.0
            The initial value of the differentiation step.
            The algorithm produces a sequence of decreasing steps.
            Hence, the initial step should be an upper bound of the true
            optimal step.
            To find such a suitable initial step, the example below provides
            an heuristic designed by the authors of the method.
            If the order of magnitude of the third derivative can be guessed, then
            :meth:`~numericalderivative.FirstDerivativeCentral.compute_step` can be used.
            Moreover, :meth:`~numericalderivative.SteplemanWinarskyInitialize.find_initial_step`
            can help to find an appropriate initial step.
        iteration_maximum : int, optional
            The number of iterations.

        Returns
        -------
        estim_step : float
            A step size which is near to optimal.
        number_of_iterations : int
            The number of iterations required to reach that optimum.

        Examples
        --------
        The following heuristic can be used to set the initial
        step (see (Stepleman and Winarsky, 1979) eq. 3.9 page 1261):

        >>> beta = 4.0
        >>> relative_precision = 1.0e-16
        >>> x = 1.0
        >>> initial_step = beta * relative_precision ** (1.0 / 3.0) * abs(x)
        """
        if self.verbose:
            print("+ find_step()")
        if initial_step <= 0.0:
            raise ValueError(
                f"initial_step must be greater than 0. Here initial_step = {initial_step}."
            )
        if iteration_maximum <= 1:
            raise ValueError(
                f"iteration_maximum must be greater than 0. "
                f"Here iteration_maximum = {iteration_maximum}."
            )
        fractional_part, _ = math.modf(iteration_maximum)
        if fractional_part != 0.0:
            raise ValueError(
                f"The maximum number of iterations must be an integer, "
                f"but its fractional part is {fractional_part}"
            )
        if self.verbose:
            print(f"initial_step={initial_step:.3e}")
        h_previous = initial_step
        f_prime_approx_previous = self.first_derivative_central.compute(h_previous)
        diff_previous = np.inf
        estim_step = 0.0
        found_monotony_break = False
        self.step_history = [initial_step]
        for number_of_iterations in range(iteration_maximum):
            h_current = h_previous / self.beta
            self.step_history.append(h_current)
            f_prime_approx_current = self.first_derivative_central.compute(h_current)
            # eq. 2.3
            diff_current = abs(f_prime_approx_current - f_prime_approx_previous)
            if self.verbose:
                print(
                    "  number_of_iterations=%d, h=%.4e, |FD(h_current) - FD(h_previous)|=%.4e"
                    % (number_of_iterations, h_current, diff_current)
                )
            if diff_current == 0.0:
                if self.verbose:
                    print("  Stop because zero difference.")
                found_monotony_break = True
                # Zero difference achieved at step h : go back one step
                estim_step = h_current * self.beta
                break

            if diff_previous < diff_current:
                if self.verbose:
                    print("  Stop because no monotony anymore.")
                found_monotony_break = True
                # Monotony breaked at step h : go back one step
                estim_step = h_current * self.beta
                break

            f_prime_approx_previous = f_prime_approx_current
            h_previous = h_current
            diff_previous = diff_current

        if not found_monotony_break:
            raise ValueError(
                "No monotony break was found after %d iterations." % (iteration_maximum)
            )
        return estim_step, number_of_iterations

    def compute_first_derivative(self, step):
        """
        Compute an approximate value of f'(x) using central finite difference.

        The denominator is, however, implemented using the equation 3.4
        in Stepleman & Winarsky (1979).

        Parameters
        ----------
        step : float, > 0
            The step size.

        Returns
        -------
        f_prime_approx : float
            The approximation of f'(x).
        """
        f_prime_approx = self.first_derivative_central.compute(step)
        return f_prime_approx

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        finite_difference_feval = (
            self.first_derivative_central.get_function().get_number_of_evaluations()
        )
        function_eval = self.functionwithargs.get_number_of_evaluations()
        total_feval = finite_difference_feval + function_eval
        return total_feval

    def get_step_history(self):
        """
        Return the history of steps during the search.

        Let n be the number of iterations.
        The best step h is not the last one (with index n-1), since this is the step
        which breaks the monotony.
        The best step has index n - 2.

        Returns
        -------
        step_history : list(float)
            The list of steps h during intermediate iterations of the search.
            This is updated by :meth:`~numericalderivative.SteplemanWinarsky.find_step`.

        """
        return self.step_history

    def get_function(self):
        """
        Return the function

        Returns
        -------
        function : function
            The function to differentiate.
        """
        return self.function

    def get_x(self):
        """
        Return the point

        Returns
        -------
        x : float
            The point where the derivative is to be evaluated.
        """
        return self.x

    def get_args(self):
        """
        Return the arguments

        Returns
        -------
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.
        """
        return self.args

    def get_beta(self):
        """
        Return the multiplier

        Returns
        -------
        beta : float, > 1.0
            The reduction factor of h at each iteration.
            A value of beta closer to 1 can improve the accuracy of the optimum
            step, but may increase the number of iterations.
        """
        return self.beta


class SteplemanWinarskyInitialize:
    r"""
    Compute an initial step for a search algorithm

    Parameters
    ----------
    algorithm : :class:`~numericalderivative.SteplemanWinarsky`
        The algorithm to find a step
    relative_precision : float, > 0, optional
        The relative precision of evaluation of f.

    Returns
    -------
    None.

    References
    ----------
    - Adaptive numerical differentiation. R. S. Stepleman and N. D. Winarsky. Journal: Math. Comp. 33 (1979), 1257-1264

    Examples
    --------

    The next example computes the step of a badly scaled function.
    We first compute an appropriate initial step using :meth:`~numericalderivative.SteplemanWinarskyInitialize.find_initial_step`
    and set it as the input of :meth:`~numericalderivative.SteplemanWinarsky.find_step`.

    >>> import numericalderivative as nd
    >>>
    >>> def scaled_exp(x):
    >>>     alpha = 1.e6
    >>>     return np.exp(-x / alpha)
    >>>
    >>> x = 1.0e-2
    >>> initial_step = 1.0e8
    >>> algorithm = nd.SteplemanWinarsky(scaled_exp, x)
    >>> initialize = nd.SteplemanWinarskyInitialize(algorithm)
    >>> initial_step, number_of_iterations = initialize.find_initial_step(
    >>>     1.0e-5,
    >>>     1.0e7,
    >>> )
    >>> h_optimal, number_of_iterations = algorithm.find_step(initial_step)
    >>> f_prime_approx = algorithm.compute_first_derivative(h_optimal)
    """

    def __init__(self, algorithm, relative_precision=1.0e-15, verbose=False):
        if relative_precision <= 0.0:
            raise ValueError(
                f"The relative precision must be > 0. "
                f"here precision = {relative_precision}"
            )
        self.relative_precision = relative_precision
        self.algorithm = algorithm
        self.x = algorithm.get_x()
        self.function = algorithm.get_function()
        self.args = algorithm.get_args()
        self.functionwithargs = nd.FunctionWithArguments(self.function, self.args)
        self.verbose = verbose
        # Initialize step history
        self.step_history = []
        return

    def number_of_lost_digits(self, h):
        r"""
        Compute the number of (base 10) lost digits.

        The loss of figures produced by the substraction in the finite
        difference formula is (see (Stepleman & Winarsky, 1979), eq. 3.10 page 1261):

        .. math::

            \delta(h) = \left|\frac{2hd(h)}{f(x)}\right|

        where :math:`h > 0` is the step and :math:`d(h)` is the central
        finite difference formula.
        The number of decimal digits losts in the substraction is
        (see (Stepleman & Winarsky, 1979), eq. 3.11 page 1261):

        .. math::

            N(h) = -\log_{10}(\delta(h))

        where :math:`\log_{10}` is the base-10 decimal digit.

        Parameters
        ----------
        h : float
            Differentiation step.

        Returns
        -------
        number_of_digits : float
            The number of digits lost by cancellation.

        """
        d = self.algorithm.compute_first_derivative(h)
        function_value = self.functionwithargs(self.x)
        # eq. 3.10 page 1261
        if function_value == 0.0:
            delta = abs(2 * h * d)
        else:
            delta = abs(2 * h * d / function_value)
        # eq. 3.11
        number_of_digits = -np.log10(delta)
        return number_of_digits

    def find_initial_step(
        self,
        h_min,
        h_max,
        maximum_bisection=53,
        log_scale=True,
    ):
        r"""
        Compute the initial step using bisection.

        Search for an initial step :math:`h_0` such that:

        .. math::

            0 < N(h_0) < T := \log_{10}\left(\frac{\epsilon_r^{-1 / 3}}{\beta}\right)

        where :math:`N` is the number of lost digits (as computed by
        :meth:`number_of_lost_digits()`), :math:`h_0` is the initial step and
        :math:`\epsilon_r` is the relative precision of the function evaluation.
        This heuristic is based on the hypothesis that the absolute value of
        the third derivative is close to 1.

        The value returned by :meth:`find_initial_step()`
        can be used as input of :meth:`~numericalderivative.SteplemanWinarsky.find_step()`.

        This algorithm can fail if the required finite difference step is
        so large that the points :math:`x \pm h` fall beyond the mathematical input
        domain of the function.

        Parameters
        ----------
        h_min : float
            The lower bound to bracket the initial differentiation step.
        h_max : float, > h_min
            The upper bound to bracket the initial differentiation step.
            We must have N(h_min) > N(h_max) where N is the number of lost digits.
        maximum_bisection : int, optional
            The maximum number of bisection iterations.
        log_scale : bool, optional
            Set to True to bisect in log scale.

        Returns
        -------
        initial_step : float
            The initial step.
        number_of_iterations : int
            The number of required iterations.

        """
        if self.verbose:
            print("+ find_initial_step()")
        if h_min <= 0.0:
            raise ValueError(f"h_min  = {h_min} must be greater than zero.")
        if h_min >= h_max:
            raise ValueError(
                f"h_min  = {h_min} > h_max = {h_max}." "Please update the bounds."
            )
        beta = self.algorithm.get_beta()
        if maximum_bisection <= 0:
            raise ValueError(
                f"maximum_bisection  = {maximum_bisection} must be greater than 1."
            )
        if self.verbose:
            print(f"+ h_min = {h_min:.3e}, h_max = {h_max:.3e}")
        # eq. 3.15
        if self.verbose:
            print(f"+ relative_precision = {self.relative_precision:.3e}")
        # eq. 3.15
        n_treshold = np.log10(self.relative_precision ** (-1.0 / 3.0) / beta)
        if n_treshold <= 0.0:
            raise ValueError(
                f"The upper bound of the number of lost digits is {n_treshold} <= 0.0."
                " Increase absolute precision."
            )
        if self.verbose:
            print(
                "Searching for h such that "
                f"0 < N(h) <= n_treshold = {n_treshold:.3f}"
            )
        # Compute N(h_min)
        n_min = self.number_of_lost_digits(h_min)
        if n_min < 0.0:
            raise ValueError(
                f"The number of lost digits for h_min is {n_min} < 0." " Reduce h_min."
            )
        if n_min >= 0.0 and n_min <= n_treshold:
            if self.verbose:
                print(f"h_min is OK; n(h_min) = {n_min}. Stop.")
            initial_step = h_min
            number_of_iterations = 0
            return initial_step, number_of_iterations

        # Compute N(h_max)
        n_max = self.number_of_lost_digits(h_max)
        if self.verbose:
            print(f"n_min = {n_min:.3f}, " f"n_max = {n_max:.3f}")
        if n_max > n_treshold:
            raise ValueError(
                f"The number of lost digits for h_max is {n_max} > {n_treshold}."
                " Increase h_max or decrease relative_precision."
            )
        if n_max >= 0.0 and n_max <= n_treshold:
            if self.verbose:
                print(f"h_max is OK; n(h_max) = {n_max}. Stop.")
            initial_step = h_max
            number_of_iterations = 0
            return initial_step, number_of_iterations

        if n_min < n_max:
            raise ValueError("N(h_min) < N(h_max)")

        # Now : n_min > n_treshold > 0 > n_max
        found = False
        self.step_history = []
        for number_of_iterations in range(maximum_bisection):
            if self.verbose:
                print(
                    f"+ Iter {number_of_iterations} / {maximum_bisection}, "
                    f"h_min = {h_min:.3e}, "
                    f"h_max = {h_max:.3e}"
                )
            if log_scale:
                initial_step = 10 ** ((np.log10(h_max) + np.log10(h_min)) / 2.0)
            else:
                initial_step = (h_max + h_min) / 2.0
            self.step_history.append(initial_step)
            n_digits = self.number_of_lost_digits(initial_step)
            if self.verbose:
                print(
                    f"  h = {initial_step:.3e}, "
                    f"  Number of lost digits = {n_digits:.3f}"
                )
            if n_digits > 0 and n_digits < n_treshold:
                found = True
                if self.verbose:
                    print("  h is just right : stop !")
                break
            elif n_digits < 0.0:
                if self.verbose:
                    print("  h is too large: reduce it")
                h_max = initial_step
            else:
                if self.verbose:
                    print("  h is small: increase it")
                h_min = initial_step
        if not found:
            raise ValueError(
                "The maximum number of bisection "
                f"iterations {maximum_bisection} has been reached."
            )
        return initial_step, number_of_iterations

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        finite_difference_feval = (
            self.first_derivative_central.get_function().get_number_of_evaluations()
        )
        function_eval = self.functionwithargs.get_number_of_evaluations()
        total_feval = finite_difference_feval + function_eval
        return total_feval

    def get_step_history(self):
        """
        Return the history of steps during the search.

        Let n be the number of iterations.
        The best step h is not the last one (with index n-1), since this is the step
        which breaks the monotony.
        The best step has index n - 2.

        Returns
        -------
        step_history : list(float)
            The list of steps h during intermediate iterations of the search.
            This is updated by :meth:`~numericalderivative.SteplemanWinarskyInitialize.find_initial_step`.

        """
        return self.step_history
