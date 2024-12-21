#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Benchmark Gill, Murray, Saunders and Wright method
==================================================

The goal of this example is to benchmark the :class:`~numericalderivative.GillMurraySaundersWright`
on a collection of test problems.
These problems are created by the :meth:`~numericalderivative.build_benchmark()` 
static method, which returns a list of problems.

References
----------
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
"""

# %%
import numpy as np
import pylab as pl
import tabulate
import numericalderivative as nd

# %%
# The next function is an oracle which returns the absolute precision
# of the value of the function.


# %%
def absolute_precision_oracle(function, x, relative_precision):
    """
    Return the absolute precision of the function value

    This oracle may fail if the function value is zero.

    Parameters
    ----------
    function : function
        The function
    x : float
        The test point
    relative_precision : float, > 0, optional
        The relative precision of evaluation of f.

    Returns
    -------
    absolute_precision : float, >= 0
        The absolute precision
    """
    function_value = function(x)
    if function_value == 0.0:
        raise ValueError(
            "The function value is zero: " "cannot compute the absolute precision"
        )
    absolute_precision = relative_precision * abs(function_value)
    return absolute_precision


class GillMurraySaundersWrightMethod:
    def __init__(self, kmin, kmax, relative_precision):
        """
        Create a GillMurraySaundersWright method to compute the approximate first derivative

        Notice that the algorithm is parametrized here based on
        the relative precision of the value of the function f.
        Then an oracle computes the absolute precision depending on
        the function, the point x and the relative precision.

        Parameters
        ----------
        kmin : float, kmin > 0
            A minimum bound for the finite difference step of the third derivative.
            If no value is provided, the default is to compute the smallest
            possible kmin using number_of_digits and x.
        kmax : float, kmax > kmin > 0
            A maximum bound for the finite difference step of the third derivative.
            If no value is provided, the default is to compute the largest
            possible kmax using number_of_digits and x.
        relative_precision : float, > 0, optional
            The relative precision of evaluation of f.
        """
        self.kmin = kmin
        self.kmax = kmax
        self.relative_precision = relative_precision

    def compute_first_derivative(self, function, x):
        """
        Compute the first derivative using GillMurraySaundersWright

        Parameters
        ----------
        function : function
            The function
        x : float
            The test point

        Returns
        -------
        f_prime_approx : float
            The approximate value of the first derivative of the function at point x
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        absolute_precision = absolute_precision_oracle(
            function, x, self.relative_precision
        )
        algorithm = nd.GillMurraySaundersWright(
            function, x, absolute_precision=absolute_precision
        )
        step, _ = algorithm.find_step(kmin, kmax)
        f_prime_approx = algorithm.compute_first_derivative(step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        return f_prime_approx, number_of_function_evaluations


# %%
# The next example computes the approximate derivative on the
# :class:`~numericalderivative.ExponentialProblem`.

# %%
print("+ Benchmark on several points")
number_of_test_points = 20
kmin = 1.0e-16
kmax = 1.0e-1
problem = nd.ExponentialProblem()
print(problem)
interval = problem.get_interval()
test_points = np.linspace(interval[0], interval[1], number_of_test_points)
relative_precision = 1.0e-15
method = GillMurraySaundersWrightMethod(kmin, kmax, relative_precision)
average_relative_error, average_feval, data = nd.benchmark_method(
    problem.get_function(),
    problem.get_first_derivative(),
    test_points,
    method.compute_first_derivative,
    True,
)
print("Average relative error =", average_relative_error)
print("Average number of function evaluations =", average_feval)
tabulate.tabulate(data, headers=["x", "Rel. err.", "F. Eval."], tablefmt="html")


# %%
# Map from the problem name to kmax

# %%
kmax_map = {
    "polynomial": 1.0,
    "inverse": 1.0e0,
    "exp": 1.0e-1,
    "log": 1.0e-3,  # x > 0
    "sqrt": 1.0e-3,  # x > 0
    "atan": 1.0e0,
    "sin": 1.0e0,
    "scaled exp": 1.0e5,
    "GMSW": 1.0e0,
    "SXXN1": 1.0e0,
    "SXXN2": 1.0e0,  # Fails
    "SXXN3": 1.0e0,
    "SXXN4": 1.0e0,
    "Oliver1": 1.0e0,
    "Oliver2": 1.0e0,
    "Oliver3": 1.0e-3,
}

# %%
# Benchmark the :class:`~numericalderivative.GillMurraySaundersWright` class
# on a collection of problems.

# %%
number_of_test_points = 100
relative_precision = 1.0e-15
delta_x = 1.0e-9
data = []
function_list = nd.build_benchmark()
number_of_functions = len(function_list)
average_relative_error_list = []
average_feval_list = []
for i in range(number_of_functions):
    problem = function_list[i]
    function = problem.get_function()
    first_derivative = problem.get_first_derivative()
    name = problem.get_name()
    kmax = kmax_map[name]
    kmin = 1.0e-16 * kmax
    lower_x_bound, upper_x_bound = problem.get_interval()
    if name == "sin":
        # Change the lower and upper bound so that the points +/-pi
        # are excluded (see below for details).
        lower_x_bound += delta_x
        upper_x_bound -= delta_x
    test_points = np.linspace(lower_x_bound, upper_x_bound, number_of_test_points)
    print(f"Function #{i}, {name}")
    method = GillMurraySaundersWrightMethod(kmin, kmax, relative_precision)
    average_relative_error, average_feval, _ = nd.benchmark_method(
        function,
        first_derivative,
        test_points,
        method.compute_first_derivative,
    )
    average_relative_error_list.append(average_relative_error)
    average_feval_list.append(average_feval)
    data.append(
        (
            name,
            kmin,
            kmax,
            average_relative_error,
            average_feval,
        )
    )
data.append(
    [
        "Average",
        "-",
        "-",
        np.nanmean(average_relative_error_list),
        np.nanmean(average_feval_list),
    ]
)
tabulate.tabulate(
    data,
    headers=["Name", "kmin", "kmax", "Average error", "Average func. eval"],
    tablefmt="html",
)

# %%
# Notice that the method cannot perform correctly for the sin function
# at the point
# Indeed, this function is such that :math:`f''(x) = 0` if :math:`x = \pm \pi`.
# In this case, the condition error is infinite and the method
# cannot work.
# Therefore, we make so that the points :math:`\pm \pi` are excluded from the benchmark.
# The same problem appears at the point :math:`x = 0`.
# This point is not included in the test set if the number of points is even
# (e.g. with `number_of_test_points = 100`), but it might appear if the
# number of test points is odd.
