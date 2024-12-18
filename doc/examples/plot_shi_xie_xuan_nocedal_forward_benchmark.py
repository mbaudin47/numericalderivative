#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Benchmark Shi, Xie, Xuan & Nocedal's method
===========================================

The goal of this example is to problem the :class:`~numericalderivative.ShiXieXuanNocedalForward`
class on a collection of test problems.
These problems are created by the :meth:`~numericalderivative.BuildBenchmark()` 
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


# %%
def compute_first_derivative(
    function,
    x,
    first_derivative,
    initial_step,
    relative_precision=1.0e-15,
    verbose=False,
):
    """
    Compute the approximate derivative from finite differences using Shi, Xie, Xuan & Nocedal's method

    Uses bisection to find the approximate optimal step for the first
    derivative.

    Parameters
    ----------
    function : function
        The function.
    x : float
        The point where the derivative is to be evaluated
    first_derivative : function
        The exact first derivative of the function.
    initial_step : float, > 0
        A initial step.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    absolute_error : float, > 0
        The absolute error between the approximate first derivative
        and the true first derivative.

    feval : int
        The number of function evaluations.
    """

    absolute_precision = abs(function(x)) * relative_precision
    try:
        algorithm = nd.ShiXieXuanNocedalForward(function, x, absolute_precision, verbose=verbose)
        step, _ = algorithm.compute_step(initial_step)
        f_prime_approx = algorithm.compute_first_derivative(step)
        f_prime_exact = first_derivative(x)
        feval = algorithm.get_number_of_function_evaluations()
        absolute_error = abs(f_prime_approx - f_prime_exact)
    except:
        absolute_error = np.nan
        feval = np.nan
    return absolute_error, feval


# %%
# Test
x = 1.0
initial_step = 1.0e0
problem = nd.ExponentialProblem()
function = problem.get_function()
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
(
    absolute_error,
    number_of_function_evaluations,
) = compute_first_derivative(
    function,
    x,
    first_derivative,
    initial_step,
    verbose=True,
)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)

# %%
# Perform the benchmark
# ---------------------


# %%
def benchmark_method(
    function, derivative_function, test_points, initial_step, verbose=False
):
    """
    Apply Stepleman & Winarsky method to compute the approximate first
    derivative using finite difference formula.

    Parameters
    ----------
    f : function
        The function.
    derivative_function : function
        The exact first derivative of the function
    test_points : list(float)
        The list of x points where the problem must be performed.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    absolute_error : float, > 0
        The absolute error between the approximate first derivative
        and the true first derivative.

    feval : int
        The number of function evaluations.

    """
    number_of_test_points = len(test_points)
    relative_error_array = np.zeros(number_of_test_points)
    feval_array = np.zeros(number_of_test_points)
    for i in range(number_of_test_points):
        x = test_points[i]
        (
            absolute_error,
            number_of_function_evaluations,
        ) = compute_first_derivative(
            function, x, derivative_function, initial_step, verbose=verbose
        )
        if verbose:
            print(
                f"x = {x}, "
                f"abs. error = {absolute_error:.3e}, "
                f"Func. eval. = {number_of_function_evaluations}"
            )
        relative_error = absolute_error / abs(derivative_function(x))
        relative_error_array[i] = relative_error
        feval_array[i] = number_of_function_evaluations

    average_relative_error = np.mean(relative_error_array)
    average_feval = np.mean(feval_array)
    if verbose:
        print("Average rel. error =", average_relative_error)
        print("Average number of function evaluations =", average_feval)
    return average_relative_error, average_feval


# %%
print("+ Benchmark on several points")
number_of_test_points = 31
problem = nd.PolynomialProblem()
interval = problem.get_interval()
function = problem.get_function()
first_derivative = problem.get_first_derivative()
initial_step = 1.0e2
test_points = np.linspace(interval[0], interval[1], number_of_test_points)
average_relative_error, average_feval = benchmark_method(
    function, first_derivative, test_points, initial_step, verbose=True
)

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
    "SXXN2": 1.0e0,  # Fails
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
number_of_test_points = 100
data = []
function_list = nd.BuildBenchmark()
number_of_functions = len(function_list)
average_absolute_error_list = []
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
    average_relative_error, average_feval = benchmark_method(
        function, first_derivative, test_points, initial_step
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
