# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Class to define Gill, Murray, Saunders and Wright algorithm
"""

import numpy as np
from .FiniteDifferenceOptimalStep import FiniteDifferenceOptimalStep
from .NumericalDerivative import NumericalDerivative
from .FiniteDifferenceFormula import FiniteDifferenceFormula


class GillMurraySaundersWright(NumericalDerivative):
    """
    Compute an approximately optimal step for the forward finite difference first derivative.

    The method is based on three steps:

    - compute an approximate optimal step for the second derivative using central finite difference formula,
    - compute the approximate second derivative using central finite difference formula,
    - compute the approximate optimal step for the first derivative using the forward finite difference formula.

    Finally, this approximately optimal step can be use to compute the
    first derivative using the forward finite difference formula.

    This algorithm is a simplified version of the algorithm published in
    (Gill, Murray, Saunders & Wright, 1983) section 3.2 page 316.
    While (Gill, Murray, Saunders & Wright, 1983) simultaneously considers
    the finite difference step of the forward, backward formula for the
    first derivative and the central formula for the second derivative,
    this algorithm only searches for the optimal step for the central
    formula for the second derivative.

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is approximated.
    relative_precision : float, optional
        The relative error of the function f at the point x.
    c_threshold_min : float, optional, > 0
        The minimum value of the condition error.
    c_threshold_max : float, optional, > c_threshold_min
        The maximum value of the condition error.
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    verbose : bool
        Set to True to print intermediate messages

    Returns
    -------
    None.

    References
    ----------
    - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
    """
    def __init__(
        self,
        function,
        x,
        relative_precision=1.0e-16,
        c_threshold_min=0.001,
        c_threshold_max=0.1,
        args=None,
        verbose=False,
    ):
        if relative_precision <= 0.0:
            raise ValueError(
                f"The relative precision must be > 0. "
                f"here relative precision = {relative_precision}"
            )
        self.relative_precision = relative_precision
        if c_threshold_max <= c_threshold_min:
            raise ValueError(
                f"c_threshold_max = {c_threshold_max} must be greater than "
                f"c_threshold_min = {c_threshold_min}"
            )
        self.c_threshold_min = c_threshold_min
        self.c_threshold_max = c_threshold_max
        self.verbose = verbose
        self.finite_difference = FiniteDifferenceFormula(function, x, args)
        super().__init__(function, x, args)
        self.y = self.function_eval(self.x)
        self.absolute_precision = abs(relative_precision * self.y)

    def compute_condition(self, k):
        """
        Compute the condition error for given step k.

        This is the condition error of the finite difference formula
        of the second derivative finite difference :

        f''(x) ~ (f(x + k) - 2 f(x) + f(x - k)) / (k ** 2)

        The condition error is a decreasing function of k.

        Parameters
        ----------
        k : float
            The step used for the finite difference approximation
            of the second derivative.

        Returns
        -------
        c : float
            The condition error.

        """
        # Eq. 8 page 314
        # We do not use compute_2nd_derivative because y=f(x) is known.
        # This way, we compute it only once.
        phi = (
            self.function_eval(self.x + k) - 2 * self.y + self.function_eval(self.x - k)
        ) / (k**2)
        # Eq. 11 page 315
        if phi == 0.0:
            c = np.inf
        else:
            c = 4.0 * self.absolute_precision / (k**2 * abs(phi))
        return c

    def compute_step_for_second_derivative(
        self, kmin, kmax, iteration_maximum=50, logscale=True
    ):
        """
        Compute the optimal step k suitable to approximate the second derivative.

        Then the approximate value of the second derivative can be computed using
        compute_2nd_derivative().

        Parameters
        ----------
        kmin : float, > 0
            The minimum step k for the second derivative.
        kmax : float, > kmin
            The maximum step k for the second derivative.
        iteration_maximum : in, optional
            The maximum number of iterations.
        logscale : bool, optional
            Set to True to use a logarithmic scale to update k.
            Set to False to use a linear scale.
            The default is True.

        Returns
        -------
        step_second_derivative : float, > 0
            The optimum step step_second_derivative.
        number_of_iterations : int
            The number of iterations required to compute step_second_derivative.

        """
        if kmin >= kmax:
            raise ValueError(f"kmin = {kmin} must be less than kmax = {kmax}.")
        # Check C(kmin)
        cmin = self.compute_condition(kmin)
        if self.verbose:
            print(f"kmin = {kmin:.3e}, c(kmin) = {cmin:.3e}")
        if cmin >= self.c_threshold_min and cmin <= self.c_threshold_max:
            iteration = 0
            return kmin, iteration
        elif cmin < self.c_threshold_min:
            raise ValueError(
                f"C(kmin) = {cmin} < c_threshold_min = {self.c_threshold_min}. "
                "Please decrease kmin. "
            )
        # Check C(kmax)
        cmax = self.compute_condition(kmax)
        if self.verbose:
            print(f"kmax = {kmax:.3e}, c(kmax) = {cmax:.3e}")
        if cmax >= self.c_threshold_min and cmax <= self.c_threshold_max:
            iteration = 0
            return kmax, iteration
        elif cmax > self.c_threshold_max:
            raise ValueError(
                f"C(kmax) = {cmax} > c_threshold_max = {self.c_threshold_max}. "
                "Please increase kmax. "
            )
        # Now c_threshold_min <= c(kmin) and c_threshold_max < c(kmin)
        # which implies: c_threshold_max < c(kmin)
        # and c(kmax) <= c_threshold_max and c(kmax) < c_threshold_min
        # which implies: c(kmax) < c_threshold_min
        # In summary: c(kmax) < c_threshold_min < c_threshold_max < c(kmin).
        for number_of_iterations in range(iteration_maximum):
            if logscale:
                logk = (np.log(kmin) + np.log(kmax)) / 2.0
                step_second_derivative = np.exp(logk)
            else:
                step_second_derivative = (kmin + kmax) / 2.0
            c = self.compute_condition(step_second_derivative)
            if self.verbose:
                print(f"Iter #{number_of_iterations}, "
                      f"kmin = {kmin:.3e}, "
                      f"kmax = {kmax:.3e}, "
                      f"k = {step_second_derivative:.3e}, "
                      f"c(k) = {c:.3e}"
                      )
            if c > self.c_threshold_min and c <= self.c_threshold_max:
                if self.verbose:
                    print(f"  c in [{self.c_threshold_min}, {self.c_threshold_max}]: stop!"
                        )
                found = True
                break
            elif c < self.c_threshold_min:
                if self.verbose:
                    print(f"  c(k) < c_threshold_min: reduce kmax.")
                kmax = step_second_derivative
            else:
                if self.verbose:
                    print(f"  c(k) >= c_threshold_min: increase kmin.")
                kmin = step_second_derivative
        if not found:
            raise ValueError(
                f"Unable to find satisfactory step_second_derivative "
                f"after {iteration_maximum} iterations."
                "Please increase iteration_maximum"
            )
        return step_second_derivative, number_of_iterations

    def compute_step(self, kmin, kmax, iteration_maximum=50, logscale=True):
        """
        Compute the optimal step suitable to approximate the fist derivative.

        This method computes the approximately optimal step for the second derivative.
        Then the approximate value of the second derivative can be computed using
        compute_2nd_derivative().

        Parameters
        ----------
        kmin : float, > 0
            The minimum step k for the second derivative.
        kmax : float, > kmin
            The maximum step k for the second derivative.
        iteration_maximum : in, optional
            The maximum number of iterations. The default is 50.
        logscale : bool, optional
            Set to True to use a logarithmic scale to update k.
            Set to False to use a linear scale.
            The default is True.

        Returns
        -------
        step : float, > 0
            The optimum step for the first derivative.
        number_of_iterations : int
            The number of iterations required to compute the step.

        """
        step_second_derivative, number_of_iterations = (
            self.compute_step_for_second_derivative(
                kmin, kmax, iteration_maximum, logscale
            )
        )
        # Compute an approximate 2nd derivative from the approximately optimal step
        second_derivative_value = (
            self.finite_difference.compute_second_derivative_central(
                step_second_derivative
            )
        )
        # Plug the step for second derivative, evaluate the second derivative,
        # and plug it into the formula.
        fd_step = FiniteDifferenceOptimalStep(self.absolute_precision)
        step, _ = fd_step.compute_step_first_derivative_forward(second_derivative_value)
        return step, number_of_iterations

    def compute_first_derivative(self, step):
        """
        Compute an approximate first derivative using finite differences

        This method uses the formula:

        f'(x) ~ (f(x + h) - f(x)) / h


        Parameters
        ----------
        step : float, > 0
            The step size.

        Returns
        -------
        f_prime_approx : float
            The approximation of f'(x).
        """
        f_prime_approx = self.finite_difference.compute_first_derivative_forward(step)
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
            self.finite_difference.get_number_of_function_evaluations()
        )
        total_feval = finite_difference_feval + self.number_of_function_evaluations
        return total_feval
