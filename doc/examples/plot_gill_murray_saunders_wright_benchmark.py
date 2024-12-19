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


class GillMurraySaundersWrightMethod:
    def __init__(self, relative_precision, kmin, kmax):
        """
        Create a GillMurraySaundersWright method to compute the approximate first derivative

        Parameters
        ----------
        relative_precision : float, > 0, optional
            The relative precision of evaluation of f.
        kmin : float, kmin > 0
            A minimum bound for the finite difference step of the third derivative.
            If no value is provided, the default is to compute the smallest
            possible kmin using number_of_digits and x.
        kmax : float, kmax > kmin > 0
            A maximum bound for the finite difference step of the third derivative.
            If no value is provided, the default is to compute the largest
            possible kmax using number_of_digits and x.
        """
        self.relative_precision = relative_precision
        self.kmin = kmin
        self.kmax = kmax

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
        algorithm = nd.GillMurraySaundersWright(
            function, x, relative_precision=self.relative_precision
        )
        step, _ = algorithm.compute_step(kmin, kmax)
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
relative_precision = 1.0e-16
method = GillMurraySaundersWrightMethod(relative_precision, kmin, kmax)
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
    method = GillMurraySaundersWrightMethod(relative_precision, kmin, kmax)
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
# Indeed, this function is such that f''(x) = 0 if x = +/-pi.
# In this case, the condition error is infinite and the method
# cannot work.
# Therefore, we make so that the points +/-pi are excluded from the benchmark.
