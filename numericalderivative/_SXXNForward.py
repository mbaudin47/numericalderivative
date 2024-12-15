# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Class to define Shi, Xie, Xuan & Nocedal algorithm for the forward formula
"""

import numpy as np
import numericalderivative as nd


class SXXNForward:
    r"""
    Compute an approximately optimal step for the forward F.D. formula of the first derivative

    Uses forward finite difference to compute an approximate value of f'(x).

    The algorithm considers the test ratio:

    .. math::

        r(h) = \frac{\left|f(x + 4h) - 4f(x + h) + 3f(x)\right|]}{8 \epsilon_f}

    where :math:`h > 0` is the step and :math:`\epsilon_f> 0` is the absolute precision of evaluation
    of the function.
    The goal of the algorithm is to find the step such that:

    .. math::

        r_\ell \leq r(h) \leq r_u

    where :math:`r_\ell > 0` is the lower bound of the test ratio
    and :math:`r_u` is the upper bound.

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is to be evaluated.
    relative_precision : float, > 0, optional
        The relative precision of evaluation of f. The default is 1.0e-16.
    minimum_test_ratio : float, > 1
        The minimum value of the test ratio.
    maximum_test_ratio : float, > minimum_test_ratio
        The maximum value of the test ratio.
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

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.
    """

    def __init__(
        self,
        function,
        x,
        relative_precision=1.0e-16,
        minimum_test_ratio=1.5,
        maximum_test_ratio=6.0,
        args=None,
        verbose=False,
    ):
        if relative_precision <= 0.0:
            raise ValueError(
                f"The relative precision must be > 0. "
                f"here precision = {relative_precision}"
            )
        self.relative_precision = relative_precision
        self.verbose = verbose
        self.first_derivative_forward = nd.FirstDerivativeForward(function, x, args)
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x
        self.step_history = []
        if minimum_test_ratio <= 1.0:
            raise ValueError(
                f"The minimum test ratio must be > 1, "
                f"but minimum_test_ratio = {minimum_test_ratio}"
            )
        if maximum_test_ratio <= minimum_test_ratio:
            raise ValueError(
                f"The maximum test ratio must be greater than the minimum, "
                f"but minimum_test_ratio = {minimum_test_ratio} "
                f" and maximum_test_ratio = {maximum_test_ratio}"
            )
        self.minimum_test_ratio = minimum_test_ratio
        self.maximum_test_ratio = maximum_test_ratio
        return

    def get_ratio_min_max(self):
        r"""
        Return the minimum and maximum of the test ratio

        Returns
        -------
        minimum_test_ratio : float, > 0
            The lower bound of the test ratio.
        maximum_test_ratio : float, > 0
            The upper bound of the test ratio.
        """
        return [self.minimum_test_ratio, self.maximum_test_ratio]

    def compute_test_ratio(self, step, absolute_precision, function_values=None):
        r"""
        Compute the test ratio

        Parameters
        ----------
        step : float, > 0
            The finite difference step
        absolute_precision : float, > 0
            The absolute precision
        function_values : list(3 floats)
            The function values f(x), f(x + h), f(x + 4h).
            If function_values is None, then compute the funtion
            values.

        Returns
        -------
        test_ratio : float, > 0
            The test ratio
        """
        if function_values is None:
            f0 = self.function(self.x)
            f1 = self.function(self.x + step)
            f4 = self.function(self.x + 4.0 * step)
            function_values = [f0, f1, f4]

        f0, f1, f4 = function_values
        test_ratio = abs(f4 - 4 * f1 + 3 * f0) / (8 * absolute_precision)
        return test_ratio

    def compute_step(
        self,
        iteration_maximum=50,
        logscale=True,
    ):
        r"""
        Compute an approximate optimum step for central derivative using monotony properties.

        This function computes an approximate optimal step h for the central
        finite difference.

        Parameters
        ----------
        iteration_maximum : int, optional
            The number of number_of_iterations. The default is 53.
        logscale : bool, optional
            Set to True to use a logarithmic scale when updating the step k
            during the search. Set to False to use a linear scale when
            updating the step k during the search.

        Returns
        -------
        estim_step : float
            A step size which is near to optimal.
        number_of_iterations : int
            The number of iterations required to reach that optimum.

        """
        if iteration_maximum < 1:
            raise ValueError(
                f"The maximum number of iterations must be > 1, "
                f"but iteration_maximum = {iteration_maximum}"
            )

        # Initialize
        number_of_iterations = 1
        f0 = self.function(self.x)
        absolute_precision = self.relative_precision * abs(f0)
        estim_step = 2.0 / np.sqrt(3.0) * np.sqrt(absolute_precision)
        f1 = self.function(self.x + estim_step)
        f4 = self.function(self.x + 4.0 * estim_step)
        lower_step_bound = 0.0
        upper_step_bound = np.inf
        self.step_history = []
        print("iteration_maximum =", iteration_maximum)
        for number_of_iterations in range(iteration_maximum):
            self.step_history.append(estim_step)
            test_ratio = self.compute_test_ratio(
                estim_step, absolute_precision, [f0, f1, f4]
            )
            if self.verbose:
                print(
                    f"+ Iteration = {number_of_iterations}, "
                    f"lower_step_bound = {lower_step_bound:.3e}, "
                    f"upper_step_bound = {upper_step_bound:.3e}, "
                    f"estim_step = {estim_step:.3e}, "
                    f"r = {test_ratio:.4f}"
                )
            if test_ratio < self.minimum_test_ratio:
                if self.verbose:
                    print(
                        "    - test_ratio < self.minimum_test_ratio. "
                        "Set lower bound to h."
                    )
                lower_step_bound = estim_step
            elif test_ratio > self.maximum_test_ratio:
                if self.verbose:
                    print(
                        "    - test_ratio > self.minimum_test_ratio. "
                        "Set upper bound to h."
                    )
                upper_step_bound = estim_step
            else:
                break
            if upper_step_bound == np.inf:
                if self.verbose:
                    print("    - upper_step_bound == np.inf. Increase h.")
                estim_step *= 4.0
                f1 = f4
                f4 = self.function(self.x + 4.0 * estim_step)
            elif lower_step_bound == 0.0:
                if self.verbose:
                    print("    - upper_step_bound == np.inf. Decrease h.")
                estim_step /= 4.0
                f4 = f1
                f1 = self.function(self.x + estim_step)
            else:
                if self.verbose:
                    print("    - Bisection.")
                if logscale:
                    log_step = (
                        np.log(lower_step_bound) + np.log(upper_step_bound)
                    ) / 2.0
                    estim_step = np.exp(log_step)
                else:
                    estim_step = (lower_step_bound + upper_step_bound) / 2.0
                f1 = self.function(self.x + estim_step)
                f4 = self.function(self.x + 4 * estim_step)

        return estim_step, number_of_iterations

    def compute_first_derivative(self, step):
        """
        Compute an approximate value of f'(x) using centered finite difference.

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
        f_prime_approx = self.first_derivative_forward.compute(step)
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
            self.first_derivative_forward.get_function().get_number_of_evaluations()
        )
        function_eval = self.function.get_number_of_evaluations()
        total_feval = finite_difference_feval + function_eval
        return total_feval

    def get_step_history(self):
        """
        Return the history of steps during the bissection search.

        Returns
        -------
        step_history : list(float)
            The list of steps k during intermediate iterations of the bissection search.
            This is updated by :meth:`compute_third_derivative`.

        """
        return self.step_history
