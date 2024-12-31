#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Benchmark Shi, Xie, Xuan & Nocedal's forward method
===================================================

The goal of this example is to benchmark the :class:`~numericalderivative.ShiXieXuanNocedalForward`
class on a collection of test problems.
These problems are created by the :meth:`~numericalderivative.build_benchmark()` 
static method, which returns a list of problems.

"""

# %%
import numpy as np
import pylab as pl
import tabulate
import numericalderivative as nd

# %%
# Compute the first derivative
# ----------------------------


class ShiXieXuanNocedalForwardMethod:
    def __init__(self, relative_precision, initial_step):
        """
        Create a ShiXieXuanNocedal method to compute the approximate first derivative

        Parameters
        ----------
        relative_precision : float, > 0, optional
            The relative precision of evaluation of f.
        initial_step : float, > 0
            The initial step in the algorithm.
        """
        self.relative_precision = relative_precision
        self.initial_step = initial_step

    def compute_first_derivative(self, function, x):
        """
        Compute the first derivative using ShiXieXuanNocedal

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
        absolute_precision = abs(function(x)) * self.relative_precision
        algorithm = nd.ShiXieXuanNocedalForward(
            function,
            x,
            absolute_precision,
        )
        step, _ = algorithm.find_step(self.initial_step)
        f_prime_approx = algorithm.compute_first_derivative(step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        return f_prime_approx, number_of_function_evaluations


# %%
# The next example computes the approximate derivative on the
# :class:`~numericalderivative.ExponentialProblem`.

# %%
initial_step = 1.0e0
problem = nd.ExponentialProblem()
print(problem)
function = problem.get_function()
x = problem.get_x()
algorithm = nd.ShiXieXuanNocedalForward(
    function,
    x,
    verbose=True,
)
second_derivative = problem.get_second_derivative()
second_derivative_value = second_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeForward.compute_step(
    second_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))
function = problem.get_function()
first_derivative = problem.get_first_derivative()
x = 1.0
relative_precision = 1.0e-15
absolute_precision = abs(function(x)) * relative_precision
method = ShiXieXuanNocedalForwardMethod(absolute_precision, initial_step)
(
    f_prime_approx,
    number_of_function_evaluations,
) = method.compute_first_derivative(function, x)
first_derivative_value = first_derivative(x)
absolute_error = abs(f_prime_approx - first_derivative_value)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)

# %%
# Perform the benchmark
# ---------------------


# %%
print("+ Benchmark on several points")
number_of_test_points = 21  # This number of test points is odd
problem = nd.PolynomialProblem()
print(problem)
interval = problem.get_interval()
function = problem.get_function()
first_derivative = problem.get_first_derivative()
initial_step = 1.0e2
test_points = np.linspace(interval[0], interval[1], number_of_test_points)
relative_precision = 1.0e-15
absolute_precision = abs(function(x)) * relative_precision
method = ShiXieXuanNocedalForwardMethod(absolute_precision, initial_step)
average_relative_error, average_feval, data = nd.benchmark_method(
    function,
    first_derivative,
    test_points,
    method.compute_first_derivative,
    verbose=True,
)
print("Average relative error =", average_relative_error)
print("Average number of function evaluations =", average_feval)
tabulate.tabulate(data, headers=["x", "Rel. err.", "F. Eval."], tablefmt="html")

# %%
# Notice that the method does not perform correctly for the point
# :math:`x = 0` for the polynomial problem.
# This test point appears only if the number of test points is odd,
# because the test interval is symmetric with respect to :math:`x = 0`.
# For this problem, the method does not perform correctly
# because the value of the function is zero at :math:`x = 0`.
# The method can perform correctly in this case, if it is
# provided a consistent value of the absolute error of the function
# value.
# Here, we compute the absolute error depending on the relative
# error and the absolute value of the value of the function.
# If the value of the function is zero, then the computed
# absolute error is zero, which produces a failure of the method.

# %%
# Map from the problem name to the initial step.

# %%
initial_step_map = {
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
    "SXXN2": 1.0e0,
    "SXXN3": 1.0e0,
    "SXXN4": 1.0e0,
    "Oliver1": 1.0e0,
    "Oliver2": 1.0e0,
    "Oliver3": 1.0e-3,
}

# %%
# The next script evaluates a collection of benchmark problems
# using the :class:`~numericalderivative.ShiXieXuanNocedalForward` class.

# %%
number_of_test_points = 100  # This value can significantly change the results
data = []
function_list = nd.build_benchmark()
number_of_functions = len(function_list)
average_absolute_error_list = []
average_feval_list = []
relative_precision = 1.0e-15
delta_x = 1.0e-9
for i in range(number_of_functions):
    problem = function_list[i]
    name = problem.get_name()
    initial_step = initial_step_map[name]
    function = problem.get_function()
    first_derivative = problem.get_first_derivative()
    interval = problem.get_interval()
    lower_x_bound, upper_x_bound = problem.get_interval()
    print(f"Function #{i}, {name}")
    if name == "sin":
        # Change the lower and upper bound so that the points +/-pi
        # are excluded (see below for details).
        lower_x_bound += delta_x
        upper_x_bound -= delta_x
    test_points = np.linspace(lower_x_bound, upper_x_bound, number_of_test_points)
    method = ShiXieXuanNocedalForwardMethod(relative_precision, initial_step)
    average_relative_error, average_feval, _ = nd.benchmark_method(
        function, first_derivative, test_points, method.compute_first_derivative
    )
    average_absolute_error_list.append(average_relative_error)
    average_feval_list.append(average_feval)
    data.append(
        (
            name,
            average_relative_error,
            average_feval,
        )
    )
data.append(
    [
        "Average",
        np.nanmean(average_absolute_error_list),
        np.nanmean(average_feval_list),
    ]
)
tabulate.tabulate(
    data,
    headers=["Name", "Average rel. error", "Average func. eval"],
    tablefmt="html",
)

# %%
# Notice that the method cannot perform correctly for the sin function
# at the point
# Indeed, this function is such that :math:`f''(x) = 0` if :math:`x = \pm \pi`.
# In this case, the test ratio and the method cannot work.
# Therefore, we make so that the points :math:`\pm \pi` are excluded from the benchmark.
# The same problem appears at the point :math:`x = 0`.
# This point is not included in the test set if the number of points is even
# (e.g. with `number_of_test_points = 100`), but it might appear if the
# number of test points is odd.
