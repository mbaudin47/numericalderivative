#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Benchmark Stepleman & Winarsky's method
=======================================

The goal of this example is to benchmark the :class:`~numericalderivative.SteplemanWinarsky`
class on a collection of test problems.
These problems are created by the :meth:`~numericalderivative.build_benchmark()` 
static method, which returns a list of problems.

References
----------
- Adaptive numerical differentiation
  R. S. Stepleman and N. D. Winarsky
  Journal: Math. Comp. 33 (1979), 1257-1264 
"""

# %%
import numpy as np
import pylab as pl
import tabulate
import numericalderivative as nd

# %%
# Compute the first derivative
# ----------------------------

# %%
# The next function computes the approximate first derivative from finite
# differences using Stepleman & Winarsky's method.


class SteplemanWinarskyMethod:
    def __init__(self, initial_step):
        """
        Create a SteplemanWinarsky method to compute the approximate first derivative

        Parameters
        ----------
        initial_step : float, > 0
            A initial step.
        """
        self.initial_step = initial_step

    def compute_first_derivative(self, function, x):
        """
        Compute the first derivative using SteplemanWinarsky

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
        algorithm = nd.SteplemanWinarsky(function, x)
        step, _ = algorithm.find_step(
            self.initial_step,
        )
        f_prime_approx = algorithm.compute_first_derivative(step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        return f_prime_approx, number_of_function_evaluations


# %%
# The next script is a simple use of the :class:`~numericalderivative.SteplemanWinarsky` class.

# %%
problem = nd.ExponentialProblem()
print(problem)
function = problem.get_function()
x = problem.get_x()
algorithm = nd.SteplemanWinarsky(
    function,
    x,
    verbose=True,
)
third_derivative = problem.get_third_derivative()
third_derivative_value = third_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
    third_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))

initial_step, iterations = algorithm.find_initial_step(
    1.0e-7,
    1.0e1,
)
print("Pas initial = ", initial_step, ", iterations = ", iterations)
lost_digits = algorithm.number_of_lost_digits(initial_step)
print("lost_digits = ", lost_digits)
initial_step = 1.0e1
function = problem.get_function()
first_derivative = problem.get_first_derivative()
x = 1.0
method = SteplemanWinarskyMethod(initial_step)
f_prime_approx, number_of_function_evaluations = method.compute_first_derivative(
    function, x
)
f_prime_exact = first_derivative(x)
absolute_error = abs(f_prime_approx - f_prime_exact)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)

# %%
# Perform the benchmark
# ---------------------


# %%
# The next example computes the approximate derivative on the
# :class:`~numericalderivative.ExponentialProblem` on a set of points.

# %%
number_of_test_points = 20
initial_step = 1.0e-1
problem = nd.ExponentialProblem()
function = problem.get_function()
first_derivative = problem.get_first_derivative()
interval = problem.get_interval()
test_points = np.linspace(interval[0], interval[1], number_of_test_points)
method = SteplemanWinarskyMethod(initial_step)
average_relative_error, average_feval, data = nd.benchmark_method(
    function, first_derivative, test_points, method.compute_first_derivative, True
)
print("Average error =", average_relative_error)
print("Average number of function evaluations =", average_feval)
tabulate.tabulate(data, headers=["x", "Rel. err.", "F. Eval."], tablefmt="html")


# %%
# Map from the problem name to the initial step.

# %%
initial_step_map = {
    "polynomial": 1.0,
    "inverse": 1.0e-3,
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
# using the :class:`~numericalderivative.SteplemanWinarsky` class.

# %%
number_of_test_points = 100
data = []
function_list = nd.build_benchmark()
number_of_functions = len(function_list)
average_relative_error_list = []
average_feval_list = []
for i in range(number_of_functions):
    problem = function_list[i]
    name = problem.get_name()
    initial_step = initial_step_map[name]
    function = problem.get_function()
    first_derivative = problem.get_first_derivative()
    interval = problem.get_interval()
    test_points = np.linspace(interval[0], interval[1], number_of_test_points)
    print(f"Function #{i}, {name}")
    method = SteplemanWinarskyMethod(initial_step)
    average_relative_error, average_feval, _ = nd.benchmark_method(
        function, first_derivative, test_points, method.compute_first_derivative
    )
    average_relative_error_list.append(average_relative_error)
    average_feval_list.append(average_feval)
    data.append(
        (
            name,
            initial_step,
            average_relative_error,
            average_feval,
        )
    )
data.append(
    [
        "Average",
        "-",
        np.nanmean(average_relative_error_list),
        np.nanmean(average_feval_list),
    ]
)
tabulate.tabulate(
    data,
    headers=["Name", "initial_step", "Average rel. error", "Average func. eval"],
    tablefmt="html",
)

# %%
