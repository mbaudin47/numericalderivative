#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Benchmark Stepleman & Winarsky's method
=======================================

Find a step which is near to optimal for a centered finite difference 
formula.

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
def compute_first_derivative_SW(
    f,
    x,
    initial_step,
    f_prime,
    beta=4.0,
    verbose=False,
):
    """
    Compute the approximate derivative from finite differences

    Uses bisection to find the approximate optimal step for the first
    derivative.

    Parameters
    ----------
    f : function
        The function.
    x : float
        The point where the derivative is to be evaluated
    initial_step : float, > 0
        A initial step.
    f_prime : function
        The exact first derivative of the function.
    beta : float, > 1.0
        The reduction factor of h at each iteration.
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
    try:
        algorithm = nd.SteplemanWinarsky(f, x, verbose=verbose)
        step, _ = algorithm.compute_step(
            initial_step,
            beta=beta,
        )
        f_prime_approx = algorithm.compute_first_derivative(step)
        feval = algorithm.get_number_of_function_evaluations()
        absolute_error = abs(f_prime_approx - f_prime(x))
    except:
        absolute_error = np.nan
        feval = np.nan
    return absolute_error, feval


# %%
# Test
x = 1.0
benchmark = nd.ExponentialProblem()
algorithm = nd.SteplemanWinarsky(
    benchmark.function,
    x,
    verbose=True,
)
third_derivative_value = benchmark.third_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
    third_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))

h0, iterations = algorithm.search_step_with_bisection(
    1.0e-7,
    1.0e1,
)
print("Pas initial = ", h0, ", iterations = ", iterations)
lost_digits = algorithm.number_of_lost_digits(h0)
print("lost_digits = ", lost_digits)

initial_step = 1.0e1
x = 1.0
(
    absolute_error,
    number_of_function_evaluations,
) = compute_first_derivative_SW(
    benchmark.function,
    x,
    initial_step,
    benchmark.first_derivative,
    beta=10.0,
    verbose=True,
)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)


# %%
def benchmark_SteplemanWinarsky_method(
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
        The list of x points where the benchmark must be performed.
    initial_step : float, > 0
        The initial step.
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
        ) = compute_first_derivative_SW(
            function,
            x,
            initial_step,
            derivative_function,
        )
        relative_error = absolute_error / abs(derivative_function(x))
        if verbose:
            print(
                "x = %.3f, abs. error = %.3e, rel. error = %.3e, Func. eval. = %d"
                % (x, absolute_error, relative_error, number_of_function_evaluations)
            )
        relative_error_array[i] = relative_error
        feval_array[i] = number_of_function_evaluations

    average_relative_error = np.mean(relative_error_array)
    average_feval = np.mean(feval_array)
    if verbose:
        print("Average error =", average_relative_error)
        print("Average number of function evaluations =", average_feval)
    return average_relative_error, average_feval


# %%
print("+ Benchmark on several points")
number_of_test_points = 100
test_points = np.linspace(0.01, 12.2, number_of_test_points)
initial_step = 1.0e-1
benchmark = nd.ExponentialProblem()
average_relative_error, average_feval = benchmark_SteplemanWinarsky_method(
    benchmark.function, benchmark.first_derivative, test_points, initial_step, True
)

# %%
# Map from the problem name to the initial step

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
# Benchmark SteplemanWinarsky
number_of_test_points = 100
data = []
function_list = nd.BuildBenchmark()
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
    average_relative_error, average_feval = benchmark_SteplemanWinarsky_method(
        function, first_derivative, test_points, initial_step
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
    headers=["Name", "h0", "Average rel. error", "Average func. eval"],
    tablefmt="html",
)

# %%
